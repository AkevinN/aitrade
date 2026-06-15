"""
CNN 选股编排器（runner.py）。

本模块是整个 CNN 选股特性的**编排心脏**：它把已经各自实现并测试过的原语
（universe 发现、Profiler 画像、CNN 代理指标、综合分合成、WF/OOS 实证、
绝对 edge 门禁、产物存储）按"两阶段漏斗"串联起来，**不重写任何判定逻辑**：

- **Tier-1（廉价预筛，纯只读）**：对 ``discover_universe`` 发现的候选池逐只
  调用 ``Profiler.profile``（``with_suggestion=False, persist=False``）算画像，
  另取**同一裁剪窗口**抽 close/log-return 数组喂给三个 CNN 代理指标，
  再经 ``compute_fitness_score`` 合成 CNN_Fitness_Score（Requirement 2）。
- **漏斗**：按 fitness_score 降序，取满足 ``available`` 且综合置信度
  ``>= min_confidence`` 的前 ``top_k`` 只入围 Tier-2，并受 ``tier2_cap``
  上限保护（超额按分截断并 ``log``，不静默丢弃；Requirement 4）。
- **Tier-2（慢/重，副作用隔离）**：对入围标的构造 ``CNNWalkForwardRequest``
  （``end <= as_of``，红线无前视，Requirement 9.3），调用
  ``run_walk_forward_evaluate(store=隔离store)`` 跑 WF/OOS，再经 ``derive_edge``
  派生绝对 edge 结论（Requirement 5）。

鲁棒性（Property 10 / Requirement 2.6 / 5.4）：单只标的在 Tier-1 或 Tier-2
抛异常时降级为该行的"失败"记录并继续其余标的；universe 为空时返回结构化
"无可选标的"结果而非抛异常（Requirement 1.5）。

时间隔离（红线）：Tier-1 经 ``Profiler`` 的 ``clip_to_as_of`` 物理裁剪，
代理指标输入数组取自**同一**裁剪窗口；Tier-2 强制 ``end <= as_of``。

结论恒草稿：``ScreeningResult.status == "draft"``，不写方案、不晋级模型、
不提交训练（Requirement 11）。
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

from aitrade.cnn.governance import run_walk_forward_evaluate
from aitrade.models.governance import CNNTrainingParams, CNNWalkForwardRequest
from aitrade.models.screening import CNNScreeningRequest
from aitrade.profiling.loader import _load_window_frame, load_local_range
from aitrade.profiling.profiler import Profiler
from aitrade.profiling.rules import DEFAULT_RULES as _PROFILING_RULES
from aitrade.profiling.rules import confidence_for
from aitrade.profiling.types import MetricValue, SymbolProfile
from aitrade.screening.edge import derive_edge
from aitrade.screening.proxy_metrics import (
    nonlinearity,
    pattern_recurrence,
    temporal_stability,
)
from aitrade.screening.rules import DEFAULT_SCREENING_RULES, ScreeningRules
from aitrade.screening.scoring import compute_fitness_score
from aitrade.screening.store import ScreeningStore, build_screening_governance_store
from aitrade.screening.types import (
    LeaderboardRow,
    ScreeningResult,
    Tier1Score,
    Tier2Verdict,
)
from aitrade.screening.universe import discover_universe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 置信度等级排序（用于 min_confidence 过滤；与 profiling.types.ConfidenceLevel 对齐）
# ---------------------------------------------------------------------------

#: 置信度等级序数：值越大越可信。用于"综合置信度 >= min_confidence"的可比较判定。
_CONFIDENCE_ORDER: dict[str, int] = {
    "insufficient": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}

#: Tier-1 阶段占总进度的比例；其余留给 Tier-2（run_tier2=False 时 Tier-1 直接到 100%）。
_TIER1_PROGRESS_SHARE = 0.5


def _confidence_rank(level: str) -> int:
    """把置信度等级映射为可比较的序数。

    Args:
        level: 置信度等级字符串（"insufficient"/"low"/"medium"/"high"）；
            未知值按最低档 0 处理（保守降级）。

    Returns:
        序数，值越大表示置信度越高。
    """
    return _CONFIDENCE_ORDER.get(level, 0)


def _close_and_returns(df: Any) -> tuple[np.ndarray, np.ndarray]:
    """从已裁剪窗口 frame 抽取收盘价与对数收益数组（镜像 Profiler 口径）。

    复刻 ``Profiler._close_array`` / ``Profiler._log_returns`` 的派生口径，
    保证代理指标用的数组与 Profiler 内部画像用的数组**完全同源、同裁剪窗口**：
    收盘价取 ``close`` 列转 float64；对数收益对"有限且为正"的价格做 ``diff(log())``。

    Args:
        df: 已经 ``clip_to_as_of`` 裁剪到 ``as_of`` 的窗口 polars DataFrame。

    Returns:
        ``(close, log_returns)`` 二元组，均为一维 float64 numpy 数组；
        缺 ``close`` 列或有效价格不足 2 个时对应数组为空。
    """
    if "close" not in df.columns:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    prices = np.asarray(df["close"].to_list(), dtype=np.float64)
    finite = prices[np.isfinite(prices) & (prices > 0)]
    if finite.size < 2:
        return prices, np.asarray([], dtype=np.float64)
    returns = np.diff(np.log(finite))
    return prices, returns


def _build_proxies(close: np.ndarray, returns: np.ndarray) -> dict[str, MetricValue]:
    """对一只标的的价格/收益数组计算三个 CNN 代理指标并包装为 MetricValue。

    每个代理指标返回 ``MetricResult(value, effective_sample)``，本函数据其
    ``effective_sample`` 经 ``confidence_for`` 分档置信度（复用 profiling 的
    ``DEFAULT_RULES``，与画像同一套有效性下限口径），并在 ``insufficient`` 或
    ``value is None`` 时把 value 置 None、补降级 note（Property 5，与
    ``Profiler._metric`` 的抑制策略一致）。

    Args:
        close: 收盘价数组（喂给 ``pattern_recurrence``）。
        returns: 对数收益数组（喂给 ``nonlinearity`` / ``temporal_stability``）。

    Returns:
        ``{"nonlinearity"|"pattern_recurrence"|"temporal_stability": MetricValue}``，
        供 ``compute_fitness_score`` 消费。
    """
    raw = {
        "nonlinearity": nonlinearity(returns),
        "pattern_recurrence": pattern_recurrence(close),
        "temporal_stability": temporal_stability(returns),
    }
    proxies: dict[str, MetricValue] = {}
    for key, result in raw.items():
        confidence = confidence_for(key, result.effective_sample, _PROFILING_RULES)
        value = result.value
        note: str | None = None
        if confidence == "insufficient" or value is None:
            value = None
            note = "insufficient_sample"
        proxies[key] = MetricValue(
            key=key,
            value=value,
            effective_sample=result.effective_sample,
            confidence=confidence,
            note=note,
        )
    return proxies


@dataclass(frozen=True)
class Tier2Window:
    """单只标的解析后的 Tier-2 评估窗口与训练超参（统一真相源）。

    由 ``ScreeningRunner._resolve_tier2_window`` 计算并被**数据充足性预检**与
    ``_build_wf_request`` 共同消费，确保两者口径一致（避免分叉导致预检与真正
    评估窗口不符）。``start`` 已被夹进本地数据范围（``max(desired_start, local_start)``），
    ``end`` 已被夹到 ``min(as_of, local_end)``。

    **不变量（由解析器强制，非调用方责任）**：任何成功返回的 ``Tier2Window``
    必然满足 ``start <= end <= as_of``，且 ``end >= local_start``（即本地数据与
    评估窗口存在非空交集）。无法满足此不变量的情形（本地无数据、``as_of`` 早于
    本地数据起点等）解析器直接返回 ``None``，不构造对象。

    Attributes:
        start: 评估窗口左界（含），已夹进本地可用范围，必然 ``<= end``。
        end: 评估窗口右界（含），已夹到 ``min(as_of, 本地最新)``，必然 ``<= as_of``
            且 ``>= local_start``（保证窗口与本地数据有交集）。
        train_days: 解析后的每折训练窗口天数（请求覆盖优先，否则规则默认）。
        fold_test_days: 解析后的单折测试集天数。
        step_days: WF 游标步进天数（取自规则）。
        eval_window_days: 解析后的评估窗口长度天数（用于无 ``eval_start`` 时回推 ``start``）。
        n_seeds: 解析后的每折训练种子数。
        epochs: 每次训练 epoch 数（取自规则）。
        local_start: 本地数据最早日期（用于预检/审计）。
        local_end: 本地数据最晚日期（用于预检/审计）。
    """

    start: date
    end: date
    train_days: int
    fold_test_days: int
    step_days: int
    eval_window_days: int
    n_seeds: int
    epochs: int
    local_start: date
    local_end: date


class ScreeningRunner:
    """CNN 选股两阶段漏斗编排器。

    串联 universe 发现 → Tier-1 批量画像打分 → top-K 入围 → Tier-2 WF/OOS 实证
    → 榜单装配。仅做编排与最小粘合，不重写任何画像/WF 判定逻辑（复用优先）。

    Tier-1 纯只读；Tier-2 通过隔离的 screening governance store 落盘，绝不污染
    生产治理产物（Requirement 10）。
    """

    def __init__(
        self,
        lab: Any,
        rules: ScreeningRules | None = None,
        store: ScreeningStore | None = None,
    ) -> None:
        """初始化编排器。

        Args:
            lab: AlphaLab 实例（或鸭子兼容对象），透传给 ``discover_universe`` 与
                ``Profiler``；Tier-1 仅经其只读接口访问行情。
            rules: 选股规则（权重/漏斗/Tier-2 超参/edge 阈值）；
                ``None`` 时使用 ``DEFAULT_SCREENING_RULES``。
            store: 选股产物存储；``None`` 时使用默认 ``ScreeningStore()``
                （写 ``config.SCREENING_PATH``）。仅在 ``req.persist=True`` 时写入。
        """
        self.lab = lab
        self.rules = rules or DEFAULT_SCREENING_RULES
        self.store = store or ScreeningStore()

    # ------------------------------------------------------------------
    # Tier-1：单标的打分
    # ------------------------------------------------------------------

    def _score_symbol(self, vt_symbol: str, req: CNNScreeningRequest) -> tuple[Tier1Score, datetime | None]:
        """对单只标的计算 Tier-1 画像 + 代理指标并合成 CNN_Fitness_Score。

        流程：调用 ``Profiler.profile``（只读、不持久化）得画像；另取**同一**
        裁剪窗口（``_load_window_frame``，与 Profiler 内部同源）抽 close/log-return
        数组算三个代理指标；最后 ``compute_fitness_score`` 合成综合分。

        不在此处兜底异常：异常由 ``run`` 的逐只 try/except 捕获并降级
        （Requirement 2.6 / Property 10）。

        Args:
            vt_symbol: 候选标的代码。
            req: 选股请求，提供 ``interval``/``as_of``/``lookback_days``。

        Returns:
            ``(tier1_score, effective_right_bound)`` 二元组；
            ``effective_right_bound`` 为该标的画像实际数据右边界（无数据时 None），
            用于审计无前视（Requirement 6.4）。
        """
        profile: SymbolProfile = Profiler(self.lab).profile(
            vt_symbol=vt_symbol,
            interval=req.interval,
            as_of=req.as_of,
            lookback_days=req.lookback_days,
            with_suggestion=False,
            persist=False,
        )

        # 数据不可用时无需再取窗口算代理：直接交给 compute_fitness_score 出不可用行。
        if not profile.available:
            return (
                compute_fitness_score(profile, {}, self.rules),
                profile.input.effective_right_bound,
            )

        # 复用 Profiler 同一裁剪窗口（_load_window_frame 内部再过一遍 clip_to_as_of），
        # 保证代理指标输入数组与画像同源、且物理上不含 datetime > as_of 的点（Property 1）。
        as_of = req.as_of.replace(tzinfo=None) if req.as_of.tzinfo is not None else req.as_of
        df = _load_window_frame(self.lab, vt_symbol, req.interval, as_of, req.lookback_days)
        if df is None:
            proxies: dict[str, MetricValue] = {}
        else:
            close, returns = _close_and_returns(df)
            proxies = _build_proxies(close, returns)

        score = compute_fitness_score(profile, proxies, self.rules)
        return score, profile.input.effective_right_bound

    # ------------------------------------------------------------------
    # 漏斗：入围选择
    # ------------------------------------------------------------------

    def _select_promoted(
        self,
        scores: list[Tier1Score],
        req: CNNScreeningRequest,
    ) -> tuple[list[Tier1Score], dict[str, str]]:
        """按分降序选出满足可用性 + 置信度的 top_k 入围标的（Requirement 4 / Property 6）。

        资格条件（缺一不可）：
        - ``available is True`` 且 ``fitness_score is not None``；
        - 综合置信度 ``overall_confidence >= req.min_confidence``（按
          insufficient<low<medium<high 排序比较）。

        资格者按 ``fitness_score`` 降序取前 ``req.top_k``，再受 ``rules.tier2_cap``
        上限保护：实际入围数超过 cap 时按分截断，被丢弃者经 ``logger.warning``
        记录（不静默丢弃，Requirement 4.5）。

        Args:
            scores: 全池 Tier-1 打分列表（任意顺序）。
            req: 选股请求，提供 ``top_k`` 与 ``min_confidence``。

        Returns:
            ``(promoted, not_promoted_reasons)``：

            - ``promoted``：入围的 ``Tier1Score`` 列表，已按分降序。
            - ``not_promoted_reasons``：``{vt_symbol: 未入围原因}``，覆盖所有
              **未入围**标的（数据不可用 / 置信度不足 / 分数靠后 / 超出上限截断）。
        """
        min_rank = _confidence_rank(req.min_confidence)
        reasons: dict[str, str] = {}

        qualified: list[Tier1Score] = []
        for s in scores:
            if not s.available or s.fitness_score is None:
                reasons[s.vt_symbol] = s.note or "数据不可用"
                continue
            if _confidence_rank(s.overall_confidence) < min_rank:
                reasons[s.vt_symbol] = (
                    f"置信度不足（{s.overall_confidence} < {req.min_confidence}）"
                )
                continue
            qualified.append(s)

        # 按 fitness_score 降序（None 已被过滤）；同分时按 vt_symbol 升序保证确定性。
        qualified.sort(key=lambda s: (-(s.fitness_score or 0.0), s.vt_symbol))

        # top_k 截断
        top_cut = qualified[: req.top_k]
        for s in qualified[req.top_k :]:
            reasons[s.vt_symbol] = "分数靠后（未进 top_k）"

        # tier2_cap 上限保护（Requirement 4.5）：超额按分截断并 log，不静默丢弃。
        promoted = top_cut[: self.rules.tier2_cap]
        dropped = top_cut[self.rules.tier2_cap :]
        if dropped:
            dropped_syms = [s.vt_symbol for s in dropped]
            logger.warning(
                "Tier-2 入围数超过上限 %d，按 fitness_score 截断丢弃 %d 只：%s",
                self.rules.tier2_cap,
                len(dropped_syms),
                ", ".join(dropped_syms),
            )
            for s in dropped:
                reasons[s.vt_symbol] = f"超出 Tier-2 上限（tier2_cap={self.rules.tier2_cap}）被截断"

        return promoted, reasons

    # ------------------------------------------------------------------
    # Tier-2：窗口解析 + WF 请求构造
    # ------------------------------------------------------------------

    def _resolved_tier2_params(self, req: CNNScreeningRequest) -> dict[str, int]:
        """解析 Tier-2 窗口/训练超参（请求级覆盖优先，None 回退 ScreeningRules）。

        请求字段（``train_days``/``fold_test_days``/``eval_window_days``/``n_seeds``）
        非 None 时覆盖对应规则默认值；``step_days``/``epochs`` 不开放请求级覆盖，
        始终取自规则。本函数**不**触碰本地数据范围，故可在 ``run()`` 起始做
        配置不变量校验时复用（与具体标的无关）。

        Args:
            req: 选股请求，提供可选的 Tier-2 覆盖字段。

        Returns:
            ``{"train_days", "fold_test_days", "step_days", "eval_window_days",
            "n_seeds", "epochs"}`` 全部为已解析的最终整数值。
        """
        return {
            "train_days": req.train_days if req.train_days is not None else self.rules.train_days,
            "fold_test_days": (
                req.fold_test_days if req.fold_test_days is not None else self.rules.fold_test_days
            ),
            "step_days": self.rules.step_days,
            "eval_window_days": (
                req.eval_window_days
                if req.eval_window_days is not None
                else self.rules.eval_window_days
            ),
            "n_seeds": req.n_seeds if req.n_seeds is not None else self.rules.n_seeds,
            "epochs": self.rules.epochs,
        }

    def _resolve_tier2_window(
        self, vt_symbol: str, req: CNNScreeningRequest
    ) -> Tier2Window | None:
        """解析单只标的的 Tier-2 评估窗口（含覆盖参数 + 本地数据夹取），统一供预检与请求构造。

        这是 Tier-2 的**唯一窗口真相源**：数据充足性预检与 ``_build_wf_request``
        都调用本函数，杜绝两处各算一遍导致口径分叉。计算步骤：

        1. 解析超参（``_resolved_tier2_params``，请求覆盖优先）。
        2. 探测本地数据范围 ``load_local_range``；任一端缺失（本地无数据）→ 返回 None，
           由预检判定"数据不足"。
        3. ``end = min(as_of.date(), local_end)``（R9.3/R9.5：不越 as_of、不越本地最新）。
        4. ``desired_start = req.eval_start or (end - eval_window_days)``。
        5. **``start = max(desired_start, local_start)``**——把窗口**夹进**本地可用范围
           （更完整的 R9.5）：薄数据标的不会把 ``start`` 落到数据存在之前，从而
           不会在 governance 侧因第一折训练窗无数据而抛晦涩 load error。
        6. **不变量强制**：若 ``end < local_start``（as_of 早于本地数据起点，完全
           无可用交集）或 ``start > end``（其他无法形成合法区间的情形）→ 返回 None；
           由预检产出专项"无可评估区间"结论。这保证任何返回的 ``Tier2Window``
           必然满足 ``start <= end <= as_of``，docstring 承诺的不变量从解析器本身
           就得到保障，而非依赖调用方额外校验。

        Args:
            vt_symbol: 入围标的代码。
            req: 选股请求，提供 ``as_of``/``interval``/``eval_start`` 与可选覆盖。

        Returns:
            解析后的 ``Tier2Window``（满足 ``start <= end <= as_of``）；
            本地无数据、``as_of`` 早于本地数据起点、或无法形成合法区间时返回 None。
        """
        params = self._resolved_tier2_params(req)
        as_of_date: date = req.as_of.date()

        try:
            local_start, local_end = load_local_range(self.lab, vt_symbol, req.interval)
        except Exception:  # noqa: BLE001 - 本地范围探测失败视同无数据（交预检判定不足）
            return None

        # 任一端缺失 → 本地无数据，无法界定窗口；交预检产出"数据不足"结论。
        if local_start is None or local_end is None:
            return None

        local_start_date = (
            local_start.date() if isinstance(local_start, datetime) else local_start
        )
        local_end_date = local_end.date() if isinstance(local_end, datetime) else local_end

        # end 夹到 min(as_of, 本地最新)：杜绝越过截止时间或本地可用范围（R9.3/R9.5）。
        end = min(as_of_date, local_end_date)

        # 不变量早期检测：若本地数据完全在 as_of 之后（end < local_start），
        # 不存在任何可用交集 → 返回 None，由预检给出专项提示，避免后续产生反向窗口。
        if end < local_start_date:
            return None

        desired_start = req.eval_start or (end - timedelta(days=params["eval_window_days"]))
        # 把窗口左界夹进本地可用范围，避免 start 落到数据存在之前（更完整的 R9.5）。
        start = max(desired_start, local_start_date)

        # 最终防御：若任何路径导致 start > end（理论上已被上方 end<local_start 拦截，
        # 此处作双重保险）→ 同样返回 None，绝不输出反向窗口。
        if start > end:
            return None

        return Tier2Window(
            start=start,
            end=end,
            train_days=params["train_days"],
            fold_test_days=params["fold_test_days"],
            step_days=params["step_days"],
            eval_window_days=params["eval_window_days"],
            n_seeds=params["n_seeds"],
            epochs=params["epochs"],
            local_start=local_start_date,
            local_end=local_end_date,
        )

    def _build_wf_request(
        self, vt_symbol: str, req: CNNScreeningRequest, window: Tier2Window
    ) -> CNNWalkForwardRequest:
        """据已解析的 ``Tier2Window`` 构造 WF/OOS 评估请求，强制 ``end <= as_of``（Property 2）。

        本函数不再自行计算窗口/超参——所有 start/end/train_days/test_days/step_days/
        n_seeds/epochs 均取自 ``window``（由 ``_resolve_tier2_window`` 解析、夹取过），
        与数据充足性预检共用同一窗口，口径一致。

        注意：``eval_window`` 在 ``ScreeningResult`` 中作联合包络回显，
        取所有入围标的 ``start``（最小）/ ``end``（最大）的并集——各标的因本地
        范围差异可能有轻微偏移，包络覆盖其余标的 ``end <= as_of`` 红线仍成立。

        Args:
            vt_symbol: 入围标的代码，作为 ``target_symbol``。
            req: 选股请求，提供 ``interval``/``objective``。
            window: 已解析并夹取过的 Tier-2 窗口与超参。

        Returns:
            构造好的 ``CNNWalkForwardRequest``，``end`` 必然 ``<= as_of``。
        """
        return CNNWalkForwardRequest(
            name=f"screening_{vt_symbol}_{window.end.isoformat()}",
            target_symbol=vt_symbol,
            input_interval=req.interval,
            start=window.start,
            end=window.end,
            train_days=window.train_days,
            test_days=window.fold_test_days,
            step_days=window.step_days,
            n_seeds=window.n_seeds,
            objective=req.objective,
            training_params=CNNTrainingParams(epochs=window.epochs),
        )

    # ------------------------------------------------------------------
    # 主编排
    # ------------------------------------------------------------------

    def run(
        self,
        req: CNNScreeningRequest,
        on_progress: Callable[[float, str], None] | None = None,
    ) -> ScreeningResult:
        """执行一次完整的两阶段漏斗选股，返回结构化 ``ScreeningResult``。

        编排步骤见模块 docstring。任何单只标的失败均降级为该行"失败"记录并继续；
        universe 为空时返回结构化"无可选标的"结果（Requirement 1.5 / Property 10）。

        Args:
            req: 选股请求参数（universe 过滤 / as_of / 漏斗 / Tier-2 超参）。
            on_progress: 两阶段进度回调 ``(percent, message)``；Tier-1 占前
                ``_TIER1_PROGRESS_SHARE``，Tier-2 占其余（run_tier2=False 时
                Tier-1 直接推进到 100%）。可为 None。

        Returns:
            ``ScreeningResult``，``status`` 恒为 ``"draft"``；含 universe 回显、
            排除项、按分降序的榜单、Tier-1 实际数据右边界、Tier-2 评估区间。
        """
        run_id = f"scr_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        created_at = datetime.now()

        # ---- 配置不变量校验（快速失败）：解析后 eval_window_days 须 >= train+test ----
        # 与具体标的无关，故提前到一切计算之前；违反时整任务以可操作的清晰报错退出，
        # 而非静默生成 0 折（那会让每只标的都退化为难懂的 Tier-2 失败）。
        params = self._resolved_tier2_params(req)
        min_needed = params["train_days"] + params["fold_test_days"]
        if params["eval_window_days"] < min_needed:
            raise ValueError(
                f"Tier-2 配置无效：eval_window_days({params['eval_window_days']}) "
                f"< train_days({params['train_days']})+fold_test_days({params['fold_test_days']})"
                f"={min_needed}，无法生成任何折。请调大 eval_window_days 或调小 "
                f"train_days/fold_test_days。"
            )

        input_echo = self._build_input_echo(req)

        # ---- Universe 发现（只读）----
        universe, excluded = discover_universe(
            self.lab,
            interval=req.interval,
            min_bar_count=req.min_bar_count,
            exchange=req.exchange,
            include_symbols=req.include_symbols,
            exclude_symbols=req.exclude_symbols,
        )

        # ---- 空池：结构化上报，不抛异常（R1.5 / Property 10）----
        if not universe:
            if on_progress:
                on_progress(100, "无可选标的（universe 过滤后为空）")
            result = ScreeningResult(
                run_id=run_id,
                created_at=created_at,
                input=input_echo,
                rules_id=self.rules.rules_id,
                universe_size=0,
                excluded=excluded,
                leaderboard=[],
                effective_right_bound=None,
                eval_window=None,
            )
            if req.persist:
                self._safe_persist(result)
            return result

        # ---- Tier-1：逐只打分（单只失败降级，不中断整批）----
        scores: list[Tier1Score] = []
        right_bounds: list[datetime] = []
        total = len(universe)
        for i, vt_symbol in enumerate(universe, start=1):
            try:
                score, erb = self._score_symbol(vt_symbol, req)
                if erb is not None:
                    right_bounds.append(erb)
            except Exception as exc:  # noqa: BLE001 - 单只失败降级（R2.6 / Property 10）
                logger.warning("Tier-1 打分失败 %s：%s", vt_symbol, exc)
                score = Tier1Score(
                    vt_symbol=vt_symbol,
                    fitness_score=None,
                    contributions=[],
                    overall_confidence="insufficient",
                    available=False,
                    note=f"打分失败: {exc}",
                )
            scores.append(score)
            if on_progress:
                pct = _TIER1_PROGRESS_SHARE * 100 * i / total
                on_progress(pct, f"Tier-1 打分 {i}/{total}: {vt_symbol}")

        # ---- 漏斗：选入围 ----
        promoted, not_promoted_reasons = self._select_promoted(scores, req)
        promoted_syms = {s.vt_symbol for s in promoted}

        # ---- Tier-2：逐只 WF/OOS（仅 run_tier2 时；单只失败降级）----
        tier2_by_symbol: dict[str, Tier2Verdict] = {}
        eval_window: dict | None = None
        if req.run_tier2 and promoted:
            gov_store = build_screening_governance_store()
            k = len(promoted)
            eval_starts: list[date] = []
            eval_ends: list[date] = []
            for j, score in enumerate(promoted, start=1):
                vt_symbol = score.vt_symbol

                def _sub_progress(p: float, m: str, _j: int = j, _sym: str = vt_symbol) -> None:
                    """把单只 WF 的 0~100 子进度映射到 Tier-2 总进度区间。"""
                    if on_progress is None:
                        return
                    base = _TIER1_PROGRESS_SHARE * 100
                    span = (1.0 - _TIER1_PROGRESS_SHARE) * 100
                    pct = base + span * ((_j - 1) + p / 100.0) / k
                    on_progress(pct, f"Tier-2 {_j}/{k}: {_sym} - {m}")

                try:
                    # 解析窗口（含覆盖参数 + 本地数据夹取），预检与评估共用同一真相源。
                    window = self._resolve_tier2_window(vt_symbol, req)

                    # ---- 数据充足性预检（在调用 WF 之前）----
                    # 本地无数据、as_of 早于本地数据起点，或夹取后可用天数 < 一折所需
                    # （train+test）→ 直接清晰跳过，绝不调用 run_walk_forward_evaluate
                    # （那会在薄数据下抛晦涩的 load error）。
                    needed = params["train_days"] + params["fold_test_days"]
                    if window is None:
                        # 区分两种无法形成合法窗口的情形，给出针对性提示：
                        # (a) 本地完全无数据 vs. (b) as_of 早于本地数据起点。
                        _skip_note: str
                        try:
                            _ls, _le = load_local_range(self.lab, vt_symbol, req.interval)
                        except Exception:  # noqa: BLE001
                            _ls, _le = None, None
                        if _ls is None or _le is None:
                            _skip_note = (
                                f"数据不足：本地无 {req.interval} 数据，跳过 Tier-2；"
                                f"可补历史或在高级设置调小窗口"
                            )
                        else:
                            # 本地有数据，但 as_of 早于本地数据起点（完全无交集）。
                            _ls_date = _ls.date() if isinstance(_ls, datetime) else _ls
                            _as_of_date: date = req.as_of.date()
                            _skip_note = (
                                f"数据不足：as_of {_as_of_date} 早于本地数据起点 {_ls_date}，"
                                f"无可评估区间，跳过 Tier-2；"
                                f"请将 as_of 调整到 {_ls_date} 之后，或补充更早历史数据"
                            )
                        verdict = Tier2Verdict(
                            vt_symbol=vt_symbol,
                            evaluable=False,
                            edge_ok=False,
                            note=_skip_note,
                        )
                    else:
                        # 防御性 max(0, ...) 确保打印值非负（窗口不变量已由解析器保证，
                        # 此处为安全冗余，杜绝任何路径输出"本地可用 -N 天"）。
                        available_days = max(0, (window.end - window.start).days)
                        if available_days < needed:
                            verdict = Tier2Verdict(
                                vt_symbol=vt_symbol,
                                evaluable=False,
                                edge_ok=False,
                                note=(
                                    f"数据不足：本地可用 {available_days} 天 "
                                    f"< Tier-2 最少需 {needed} 天"
                                    f"（train {params['train_days']} + test {params['fold_test_days']}）；"
                                    f"可补历史或在高级设置调小窗口"
                                ),
                            )
                        else:
                            # 预检通过：从同一窗口构造 WF 请求并评估。
                            wf_req = self._build_wf_request(vt_symbol, req, window)
                            eval_starts.append(wf_req.start)
                            eval_ends.append(wf_req.end)
                            report = run_walk_forward_evaluate(
                                wf_req, on_progress=_sub_progress, store=gov_store
                            )
                            verdict = derive_edge(report, self.rules)
                except Exception as exc:  # noqa: BLE001 - 单只 Tier-2 失败降级（R5.4）
                    logger.warning("Tier-2 评估失败 %s：%s", vt_symbol, exc)
                    verdict = Tier2Verdict(
                        vt_symbol=vt_symbol,
                        evaluable=False,
                        edge_ok=False,
                        note=f"Tier-2 失败: {exc}",
                    )
                tier2_by_symbol[vt_symbol] = verdict
                if on_progress:
                    base = _TIER1_PROGRESS_SHARE * 100
                    span = (1.0 - _TIER1_PROGRESS_SHARE) * 100
                    on_progress(base + span * j / k, f"Tier-2 完成 {j}/{k}: {vt_symbol}")

            if eval_starts and eval_ends:
                # 评估区间回显：取所有入围标的的最早 start / 最晚 end（仍满足 end<=as_of）。
                eval_window = {
                    "start": min(eval_starts).isoformat(),
                    "end": max(eval_ends).isoformat(),
                    "objective": req.objective,
                }
        elif on_progress:
            on_progress(100, "Tier-1 排名完成（未运行 Tier-2）")

        # ---- 装配榜单 ----
        leaderboard = self._build_leaderboard(scores, promoted_syms, tier2_by_symbol)

        effective_right_bound = max(right_bounds) if right_bounds else None

        result = ScreeningResult(
            run_id=run_id,
            created_at=created_at,
            input=input_echo,
            rules_id=self.rules.rules_id,
            universe_size=len(universe),
            excluded=excluded,
            leaderboard=leaderboard,
            effective_right_bound=effective_right_bound,
            eval_window=eval_window,
        )

        if req.persist:
            self._safe_persist(result)

        if on_progress:
            on_progress(100, "选股完成")

        return result

    # ------------------------------------------------------------------
    # 装配 / 回显 / 持久化辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _build_leaderboard(
        scores: list[Tier1Score],
        promoted_syms: set[str],
        tier2_by_symbol: dict[str, Tier2Verdict],
    ) -> list[LeaderboardRow]:
        """把 Tier-1 打分 + Tier-2 结论装配成按分降序的榜单。

        排序键：available=False 的行排末尾；available 行按 ``fitness_score`` 降序，
        同分按 ``vt_symbol`` 升序（确定性）。``rank`` 从 1 连续编号。

        Args:
            scores: 全池 Tier-1 打分列表。
            promoted_syms: 入围 Tier-2 的标的代码集合。
            tier2_by_symbol: ``{vt_symbol: Tier2Verdict}``；未跑 Tier-2 的标的不在其中。

        Returns:
            按分降序、连续编号的 ``LeaderboardRow`` 列表。
        """

        def _sort_key(s: Tier1Score) -> tuple[int, float, str]:
            # available=False / 无分 → 第一元 1 排末尾；其余按 -score 升序（即分降序）。
            if not s.available or s.fitness_score is None:
                return (1, 0.0, s.vt_symbol)
            return (0, -s.fitness_score, s.vt_symbol)

        ordered = sorted(scores, key=_sort_key)
        rows: list[LeaderboardRow] = []
        for rank, s in enumerate(ordered, start=1):
            promoted = s.vt_symbol in promoted_syms
            rows.append(
                LeaderboardRow(
                    rank=rank,
                    tier1=s,
                    promoted_to_tier2=promoted,
                    tier2=tier2_by_symbol.get(s.vt_symbol),
                )
            )
        return rows

    def _build_input_echo(self, req: CNNScreeningRequest) -> dict:
        """构造 ScreeningResult.input 的输入回显字典（供复现与审计）。

        额外回显 ``tier2_config``——本次实际生效的 Tier-2 超参（请求覆盖与规则
        默认解析后的最终值），便于审计/复现：即便用户在请求里只填了部分覆盖，
        产物也能完整记录真正用了哪套窗口/种子（Requirement 6.2）。

        Args:
            req: 选股请求。

        Returns:
            JSON 友好的回显字典，键覆盖请求基础参数、universe 过滤、漏斗、
            Tier-2 超参（含解析后的 ``tier2_config``）与本次所用 ``rules_id``
            （与 types.py 的 input 字段契约一致，自包含以便序列化后单文件可复现）。
        """
        params = self._resolved_tier2_params(req)
        return {
            "name": req.name,
            "interval": req.interval,
            "as_of": req.as_of.isoformat(),
            "lookback_days": req.lookback_days,
            "exchange": req.exchange,
            "min_bar_count": req.min_bar_count,
            "include_symbols": list(req.include_symbols),
            "exclude_symbols": list(req.exclude_symbols),
            "top_k": req.top_k,
            "run_tier2": req.run_tier2,
            "min_confidence": req.min_confidence,
            "objective": req.objective,
            "eval_start": req.eval_start.isoformat() if req.eval_start else None,
            "persist": req.persist,
            "rules_id": self.rules.rules_id,
            # 解析后的 Tier-2 超参（请求覆盖优先，否则规则默认），供复现/审计（R6.2）。
            "tier2_config": {
                "train_days": params["train_days"],
                "fold_test_days": params["fold_test_days"],
                "step_days": params["step_days"],
                "eval_window_days": params["eval_window_days"],
                "n_seeds": params["n_seeds"],
                "epochs": params["epochs"],
                "objective": req.objective,
            },
        }

    def _safe_persist(self, result: ScreeningResult) -> None:
        """尝试持久化 ScreeningResult，写入失败不影响结果返回（Requirement 6.3）。

        Args:
            result: 待持久化的选股产物。
        """
        try:
            self.store.save(result)
        except Exception as exc:  # noqa: BLE001 - 持久化失败仅记录，不影响返回
            logger.warning("选股产物持久化失败 run_id=%s：%s", result.run_id, exc)


def run_cnn_screening_batch(
    req: CNNScreeningRequest,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict:
    """task_manager 入口：构建 AlphaLab、跑选股、返回 JSON 可序列化结果字典。

    本函数是 ``POST /api/cnn/screening/batch`` 端点经 ``task_manager.run_async``
    异步调度的目标：它新建一个指向 ``ALPHA_LAB_PATH`` 的 ``AlphaLab``（复用 API
    层的 ``_get_alpha_lab`` 工厂），用默认 ``ScreeningRules`` / ``ScreeningStore``
    构造 ``ScreeningRunner`` 并执行，最后把 ``ScreeningResult`` 以
    ``model_dump(mode="json")`` 序列化为 JSON 原生结构（datetime → 字符串），
    供 task_manager 写入 ``task.result`` 与归档。

    Args:
        req: 选股请求参数。
        on_progress: task_manager 注入的进度回调 ``(percent, message)``，可为 None。

    Returns:
        ``ScreeningResult.model_dump(mode="json")``——完全 JSON 可序列化的字典。
    """
    from aitrade.api.alpha import _get_alpha_lab

    lab = _get_alpha_lab()
    result = ScreeningRunner(lab).run(req, on_progress)
    return result.model_dump(mode="json")
