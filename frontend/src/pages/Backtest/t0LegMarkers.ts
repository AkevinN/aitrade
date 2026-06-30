// 做 T 标定 K 线买卖腿标记：把标定窗日线 + 策略「当前配置档」算成 K 线买卖点 + 逐日 hover 明细。
// 纯函数、无前视（gap 用昨收 close.shift(1)），便于单测。
import type { TickPolicyCfg, T0DailyBar } from '../../types/t0'
import type { OHLCBar, TradeMarker } from '../../components/charts/types'

/** buildLegChart 产物。 */
export interface LegChart {
  /** K 线数据（升序，time=YYYY-MM-DD） */
  bars: OHLCBar[]
  /** 买卖腿标记：卖腿在上、买腿在下（无 text，详情走 hover） */
  markers: TradeMarker[]
  /** 逐日 hover 明细文本，键为 d（YYYY-MM-DD） */
  details: Map<string, string>
  /** 是否支持按配置档标腿；vol_scaled/trend_tilt 档位动态、不支持 → false */
  supported: boolean
}

/** 按比较运算判断 gap 是否命中（与后端 _OPS 一致）。 */
function gapMatches(gap: number, op: string, thr: number): boolean {
  return op === 'gt' ? gap > thr : op === 'ge' ? gap >= thr : op === 'lt' ? gap < thr : gap <= thr
}

/** 当日的（卖档, 买档, 场景名）；vol/trend 等动态档返回 null。 */
function ticksForDay(
  policy: TickPolicyCfg, open: number, prevClose: number | null,
): { sell: number; buy: number; regime: string } | null {
  if (policy.kind === 'fixed') return { sell: policy.sell_tick, buy: policy.buy_tick, regime: '固定档' }
  if (policy.kind === 'conditional') {
    const gap = prevClose ? open / prevClose - 1 : 0
    for (const r of policy.rules) {
      if (r.lhs !== 'gap') continue           // 仅 gap 规则可在前端无前视判定；信号/振幅/动量需历史，跳过
      if (gapMatches(gap, r.op, r.threshold)) {
        return { sell: r.sell_tick, buy: r.buy_tick, regime: r.name || `gap${r.op}${r.threshold}` }
      }
    }
    return { sell: policy.default_sell_tick, buy: policy.default_buy_tick, regime: '默认（平开等）' }
  }
  return null   // vol_scaled / trend_tilt：档位按历史动态算，前端不复算
}

/** 元→分（整数）显示。 */
const fen = (yuan: number): number => Math.round(yuan * 100)

/** 拼一日的 hover 明细文本（多行，纯文本，无 HTML 注入风险）。 */
function dayDetail(
  b: T0DailyBar, prevClose: number | null,
  t: { sell: number; buy: number; regime: string } | null,
  sellFire: boolean, buyFire: boolean,
): string {
  const gapTxt = prevClose ? `${((b.open / prevClose - 1) * 100).toFixed(2)}%` : '—（首日无昨收）'
  const lines = [
    `${b.d}　${t ? t.regime : '动态档'}`,
    `开 ${b.open}　高 ${b.high}　低 ${b.low}　收 ${b.close}`,
    `跳空 ${gapTxt}`,
  ]
  if (t) {
    lines.push(`卖腿 开+${fen(t.sell)}分=${(b.open + t.sell).toFixed(2)} ${sellFire ? '✓触发' : '✗未触'}`)
    lines.push(`买腿 开−${fen(t.buy)}分=${(b.open - t.buy).toFixed(2)} ${buyFire ? '✓触发' : '✗未触'}`)
  } else {
    lines.push('档位按历史动态计算，K 线暂不标腿')
  }
  return lines.join('\n')
}

/**
 * 把标定窗日线 + 策略「当前配置档」算成 K 线买卖腿标记与逐日 hover 明细。
 *
 * 每日按策略当前配置取档（固定档=配置对；条件策略=按当日 gap 首个命中的跳空规则、否则默认档），
 * 卖腿在 `high ≥ open+卖档` 触发、买腿在 `low ≤ open−买档` 触发，分别标在 K 线上下。gap 用昨收，无前视。
 *
 * @param bars - 标定窗逐日 OHLC（升序）
 * @param policy - 被标定的策略（用其当前配置档判定触发）
 * @returns {@link LegChart}：K 线、买卖点、逐日明细、是否支持标腿
 */
export function buildLegChart(bars: T0DailyBar[], policy: TickPolicyCfg): LegChart {
  const chartBars: OHLCBar[] = bars.map((b) => ({
    time: b.d, open: b.open, high: b.high, low: b.low, close: b.close,
  }))
  const markers: TradeMarker[] = []
  const details = new Map<string, string>()
  const dynamic = policy.kind === 'vol_scaled' || policy.kind === 'trend_tilt'

  let prevClose: number | null = null
  for (const b of bars) {
    const t = dynamic ? null : ticksForDay(policy, b.open, prevClose)
    let sellFire = false
    let buyFire = false
    if (t) {
      sellFire = b.high >= b.open + t.sell
      buyFire = b.low <= b.open - t.buy
      if (sellFire) markers.push({ time: b.d, side: 'sell', price: b.open + t.sell })
      if (buyFire) markers.push({ time: b.d, side: 'buy', price: b.open - t.buy })
    }
    details.set(b.d, dayDetail(b, prevClose, t, sellFire, buyFire))
    prevClose = b.close
  }

  return { bars: chartBars, markers, details, supported: !dynamic }
}
