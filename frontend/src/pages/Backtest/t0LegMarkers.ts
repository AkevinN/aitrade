// 做 T 标定 K 线买卖腿标记：把标定窗日线 + 画像「建议档」算成 K 线买卖点 + 逐日 hover 明细。
// 标记口径与上方表格「建议档」一致：固定档用全窗建议；条件策略按当日 gap 所属高/低/平开场景用该场景建议。
// 纯函数、无前视（gap 用昨收 close.shift(1)），便于单测。
import type { T0DailyBar, T0Profile, T0SegmentedProfile } from '../../types/t0'
import type { OHLCBar, TradeMarker } from '../../components/charts/types'

/** 当日采用的（卖档, 买档, 场景名）。 */
export interface DayTicks {
  sell: number
  buy: number
  regime: string
}

/** 逐日档位解析器：给定当日开盘与昨收，返回当日采用的档位；返回 null 表示不标腿（动态档）。 */
export type TickResolver = (open: number, prevClose: number | null) => DayTicks | null

/** buildLegChart 产物。 */
export interface LegChart {
  /** K 线数据（升序，time=YYYY-MM-DD） */
  bars: OHLCBar[]
  /** 买卖腿标记：卖腿在上、买腿在下（无 text，详情走 hover） */
  markers: TradeMarker[]
  /** 逐日 hover 明细文本，键为 d（YYYY-MM-DD） */
  details: Map<string, string>
  /** 是否标了腿（动态档/无解析时为 false） */
  supported: boolean
}

/** 元→分（整数）显示。 */
const fen = (yuan: number): number => Math.round(yuan * 100)

/** 拼一日的 hover 明细文本（多行，纯文本，无 HTML 注入风险）。 */
function dayDetail(
  b: T0DailyBar, prevClose: number | null, t: DayTicks | null, sellFire: boolean, buyFire: boolean,
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
 * 把标定窗日线按 `tickFor` 解析出的逐日档位算成 K 线买卖腿标记与逐日 hover 明细。
 *
 * 卖腿在 `high ≥ open+卖档` 触发（▼上方）、买腿在 `low ≤ open−买档` 触发（▲下方）。gap 用昨收，无前视。
 *
 * @param bars - 标定窗逐日 OHLC（升序）
 * @param tickFor - 逐日档位解析器（见 {@link fixedSuggestionResolver}/{@link gapSuggestionResolver}）
 * @returns {@link LegChart}
 */
export function buildLegChart(bars: T0DailyBar[], tickFor: TickResolver): LegChart {
  const chartBars: OHLCBar[] = bars.map((b) => ({
    time: b.d, open: b.open, high: b.high, low: b.low, close: b.close,
  }))
  const markers: TradeMarker[] = []
  const details = new Map<string, string>()
  let prevClose: number | null = null
  let any = false

  for (const b of bars) {
    const t = tickFor(b.open, prevClose)
    let sellFire = false
    let buyFire = false
    if (t) {
      any = true
      sellFire = b.high >= b.open + t.sell
      buyFire = b.low <= b.open - t.buy
      if (sellFire) markers.push({ time: b.d, side: 'sell', price: b.open + t.sell })
      if (buyFire) markers.push({ time: b.d, side: 'buy', price: b.open - t.buy })
    }
    details.set(b.d, dayDetail(b, prevClose, t, sellFire, buyFire))
    prevClose = b.close
  }

  return { bars: chartBars, markers, details, supported: any }
}

/** 固定档/全窗画像的解析器：每天都用全窗「建议档」（与表格建议一致）。 */
export function fixedSuggestionResolver(prof: T0Profile): TickResolver {
  return () => ({ sell: prof.suggested_sell_tick, buy: prof.suggested_buy_tick, regime: '建议档' })
}

/**
 * 条件(跳空)策略的解析器：按当日 gap 落在高/低/平开哪个场景，用**该场景的建议档**（与分场景表格一致）。
 *
 * @param seg - 分场景画像（含各场景 suggested 档与切分阈值 thresh）
 */
export function gapSuggestionResolver(seg: T0SegmentedProfile): TickResolver {
  const byRegime = new Map(seg.segments.map((s) => [s.regime, s]))
  const t = seg.thresh
  return (open, prevClose) => {
    const gap = prevClose ? open / prevClose - 1 : 0
    const regime = gap > t ? 'high' : gap < -t ? 'low' : 'flat'
    const s = byRegime.get(regime)
    if (!s) return null
    return { sell: s.profile.suggested_sell_tick, buy: s.profile.suggested_buy_tick, regime: s.label }
  }
}

/** 动态档（波动/趋势）解析器：不标腿（档位按历史动态算，画像测不出最优）。 */
export const noMarkerResolver: TickResolver = () => null
