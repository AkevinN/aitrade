"""
CNN 模型「真实结构」探查 —— 从 checkpoint 重建真实模型实例并导出结构。

设计要点（真实性保证）：
- 不基于配置字段「推算」结构，而是用 model_config 真正 create_market_cnn(...)
  重建实例，再 load_state_dict(strict=True) 把训练时保存的权重灌进去。
- load_state_dict(strict=True) 是真实性闸门：若 network.py 的结构与 checkpoint
  权重在层名 / 形状上不一致，会直接抛错。能成功加载即证明「展示的结构 == 训练时的模型」。
- 逐层张量形状通过 forward hook 跑一遍 dummy 输入获得，是真实前向计算的结果，
  而非按公式估算。
"""

from __future__ import annotations

import logging
from typing import Any

from .network import create_market_cnn
from .storage import CNN_MODEL_DIR

logger = logging.getLogger(__name__)


def _human_params(n: int) -> str:
    """把参数量格式化为易读字符串（如 12.3K / 1.2M）。

    Args:
        n: 参数数量（整数）。

    Returns:
        带单位后缀的字符串；<1 000 时直接返回数字字符串。
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def describe_cnn_architecture(name: str) -> dict[str, Any]:
    """探查指定 CNN 模型的真实网络结构。

    重建模型实例并用 load_state_dict(strict=True) 加载 checkpoint 权重（真实性闸门），
    再通过 forward hook 跑 dummy 输入获取逐层真实输出形状。
    能成功加载即证明「展示的结构 == 训练时的模型」，否则 verified=False 并附不一致说明。

    Args:
        name: 模型名称（不含 .pt 后缀），对应 CNN_MODEL_DIR/<name>.pt。

    Returns:
        字典，包含：
        - verified (bool): 权重是否严格匹配重建的结构（True=完全一致）。
        - verify_message (str): verified=False 时的不一致说明；一致时为空字符串。
        - forward_error (str): dummy forward 失败时的错误说明；成功为空字符串。
        - module_repr (str): PyTorch 原生模块树（str(model)），层级一目了然。
        - objective (str): 训练目标，"classification" 或 "regression"。
        - input_shapes (dict): 探查时使用的输入张量形状，键为 "x" 和 "group_mask"，
          形状分别为 [1, C, T, S, G] 和 [1, 1, 1, S, G]。
        - output_shape (list[int] | None): 整个模型的输出形状；forward 失败时为 None。
        - total_params (int): 总参数量。
        - total_params_h (str): 易读格式（如 "12.3K"）。
        - trainable_params (int): 可训练参数量。
        - trainable_params_h (str): 易读格式。
        - param_dtype (str): 权重数据类型（如 "float32"）；无参数时为 "—"。
        - layers (list[dict]): 逐层（叶子模块）按 forward 执行顺序的列表，每项含
          name/type/params/params_h/output_shape。

    Raises:
        FileNotFoundError: 模型文件不存在时抛出。
    """
    import torch

    model_path = CNN_MODEL_DIR / f"{name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {name}")

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    mc = checkpoint.get("model_config", {})
    objective = str(checkpoint.get("train_config", {}).get("objective", "classification"))

    in_channels = int(mc.get("in_channels", 1))
    time_steps = int(mc.get("time_steps", 1))
    max_group_width = int(mc.get("max_group_width", 1))
    group_count = int(mc.get("group_count", 1))
    dropout = float(mc.get("dropout", 0.5))

    # 1) 用配置真正重建模型实例
    model = create_market_cnn(
        in_channels,
        time_steps,
        max_group_width,
        group_count,
        dropout,
        objective=objective,
    )

    # 2) 灌入训练时保存的权重 —— 真实性闸门
    verified = True
    verify_message = ""
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        verified = False
        verify_message = "checkpoint 缺少 model_state_dict，无法校验权重与结构是否一致"
    else:
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as exc:  # noqa: BLE001 - 需要把不一致原因透传给前端
            verified = False
            verify_message = (
                f"权重与当前网络结构不完全匹配（network.py 可能已改动）：{exc}"
            )
            # 宽松加载，仍尽量展示能对上的部分
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception:  # noqa: BLE001
                pass

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_dtype = "—"
    for p in model.parameters():
        param_dtype = str(p.dtype).replace("torch.", "")
        break

    # 3) 注册 forward hook，跑一遍 dummy 输入取逐层真实输出形状
    leaf_modules: list[tuple[str, Any]] = [
        (mod_name, module)
        for mod_name, module in model.named_modules()
        if mod_name and len(list(module.children())) == 0
    ]
    captured: dict[str, list[int]] = {}
    handles = []

    def _make_hook(mod_name: str):
        def _hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                captured[mod_name] = list(output.shape)
        return _hook

    for mod_name, module in leaf_modules:
        handles.append(module.register_forward_hook(_make_hook(mod_name)))

    output_shape: list[int] | None = None
    forward_error = ""
    try:
        with torch.no_grad():
            dummy_x = torch.zeros(1, in_channels, time_steps, max_group_width, group_count)
            dummy_mask = torch.ones(1, 1, 1, max_group_width, group_count)
            out = model(dummy_x, dummy_mask)
            if isinstance(out, torch.Tensor):
                output_shape = list(out.shape)
    except Exception as exc:  # noqa: BLE001 - forward 失败不应让接口崩溃
        forward_error = f"前向探查失败，逐层形状不可用：{exc}"
        logger.warning("CNN 结构探查 forward 失败 (%s): %s", name, exc)
    finally:
        for h in handles:
            h.remove()

    # 4) 组装逐层列表（按执行顺序：有 hook 命中的在前，未命中的补在后）
    leaf_param_counts: dict[str, int] = {}
    for mod_name, module in leaf_modules:
        leaf_param_counts[mod_name] = sum(p.numel() for p in module.parameters())

    layers: list[dict[str, Any]] = []
    seen: set[str] = set()
    # captured 的 dict 在 Py3.7+ 保持插入顺序 == 执行顺序
    for mod_name, shape in captured.items():
        module = dict(leaf_modules)[mod_name]
        params = leaf_param_counts.get(mod_name, 0)
        layers.append({
            "name": mod_name,
            "type": type(module).__name__,
            "params": params,
            "params_h": _human_params(params),
            "output_shape": shape,
        })
        seen.add(mod_name)
    for mod_name, module in leaf_modules:
        if mod_name in seen:
            continue
        params = leaf_param_counts.get(mod_name, 0)
        layers.append({
            "name": mod_name,
            "type": type(module).__name__,
            "params": params,
            "params_h": _human_params(params),
            "output_shape": None,
        })

    return {
        "name": name,
        "verified": verified,
        "verify_message": verify_message,
        "forward_error": forward_error,
        "module_repr": str(model),
        "objective": objective,
        "input_shapes": {
            "x": [1, in_channels, time_steps, max_group_width, group_count],
            "group_mask": [1, 1, 1, max_group_width, group_count],
        },
        "output_shape": output_shape,
        "total_params": total_params,
        "total_params_h": _human_params(total_params),
        "trainable_params": trainable_params,
        "trainable_params_h": _human_params(trainable_params),
        "param_dtype": param_dtype,
        "layers": layers,
    }
