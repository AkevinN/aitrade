// 把标的画像的建议档位回填到档位策略（纯函数，便于单测）。
// 单位均为后端口径（元）；不可靠场景（样本不足）不回填。
import type { TickPolicyCfg, T0Profile, T0SegmentedProfile } from '../../types/t0'

/**
 * 把全窗画像的建议 (sell, buy) 档回填到固定档策略。
 *
 * @param policy - 目标策略；非 fixed 原样返回
 * @param prof - 全窗画像（用其 suggested_sell_tick/buy_tick）
 * @returns 回填后的策略（元）
 */
export function applyFixedSuggestion(policy: TickPolicyCfg, prof: T0Profile): TickPolicyCfg {
  if (policy.kind !== 'fixed') return policy
  return { ...policy, sell_tick: prof.suggested_sell_tick, buy_tick: prof.suggested_buy_tick }
}

/**
 * 把分场景画像的建议回填到条件(跳空)策略：高开→首个 `gap`+`gt/ge` 规则、
 * 低开→首个 `gap`+`lt/le` 规则、平开→`default`。
 *
 * 样本不足（`n_days < minDays`）的场景**不回填**（保持该规则/默认原值），避免据不可靠建议下结论。
 * 策略无对应跳空规则时，该场景建议无处可填（仅平开能落到 default）。
 *
 * @param policy - 目标条件策略；非 conditional 原样返回
 * @param segs - 分场景画像（高/低/平开）
 * @param minDays - 场景可用的最少样本天数，默认 5
 * @returns 回填后的策略（元）
 */
export function applyGapSegments(
  policy: TickPolicyCfg,
  segs: T0SegmentedProfile,
  minDays = 5,
): TickPolicyCfg {
  if (policy.kind !== 'conditional') return policy
  const by = new Map(segs.segments.map((s) => [s.regime, s]))
  const usable = (regime: 'high' | 'low' | 'flat'): T0Profile | null => {
    const s = by.get(regime)
    return s && s.n_days >= minDays ? s.profile : null
  }
  const high = usable('high')
  const low = usable('low')
  const flat = usable('flat')

  let highDone = false
  let lowDone = false
  const rules = policy.rules.map((r) => {
    if (high && !highDone && r.lhs === 'gap' && (r.op === 'gt' || r.op === 'ge')) {
      highDone = true
      return { ...r, sell_tick: high.suggested_sell_tick, buy_tick: high.suggested_buy_tick }
    }
    if (low && !lowDone && r.lhs === 'gap' && (r.op === 'lt' || r.op === 'le')) {
      lowDone = true
      return { ...r, sell_tick: low.suggested_sell_tick, buy_tick: low.suggested_buy_tick }
    }
    return r
  })

  return {
    ...policy,
    rules,
    default_sell_tick: flat ? flat.suggested_sell_tick : policy.default_sell_tick,
    default_buy_tick: flat ? flat.suggested_buy_tick : policy.default_buy_tick,
  }
}
