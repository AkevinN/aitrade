import { describe, expect, it } from 'vitest'

import type { SchemeSuggestion, SuggestionItem } from '../types/alpha'
import {
  buildProfilingRequest,
  confidenceStyle,
  formatMetricValue,
  mapSuggestionToFormValues,
  shouldDropObservation,
} from './profiling'

function item(field: string, value: unknown): SuggestionItem {
  return {
    field,
    value,
    reason: 'because',
    based_on_confidence: 'high',
  }
}

function suggestion(items: SuggestionItem[], degraded = false): SchemeSuggestion {
  return {
    status: 'draft',
    interval: '30m',
    vt_symbols: ['600030.SSE'],
    items,
    degraded,
  }
}

describe('profiling utils', () => {
  it('maps four confidence levels', () => {
    expect(confidenceStyle('high')).toMatchObject({ color: 'green', weak: false })
    expect(confidenceStyle('medium')).toMatchObject({ color: 'blue', weak: false })
    expect(confidenceStyle('low')).toMatchObject({ color: 'orange', weak: true })
    expect(confidenceStyle('insufficient')).toMatchObject({ color: 'default', weak: true })
  })

  it('formats null, integers, decimals and dictionaries', () => {
    expect(formatMetricValue(null)).toBe('-')
    expect(formatMetricValue(12)).toBe('12')
    expect(formatMetricValue(0.123456)).toBe('0.1235')
    expect(formatMetricValue({ 0.5: 0.01, 0.9: 0.025 })).toBe('0.5: 0.01, 0.9: 0.025')
    expect(formatMetricValue(0.123456, 'realized_volatility')).toBe('12.35%')
    expect(formatMetricValue(123_456_789, 'avg_turnover')).toBe('1.23 亿')
    expect(formatMetricValue({ 0.5: 0.01, 0.9: 0.025 }, 'amplitude_quantiles')).toBe('0.5: 1%, 0.9: 2.5%')
  })

  it('maps only allowlisted fields and reports unmapped items', () => {
    const result = mapSuggestionToFormValues(
      suggestion([
        item('label_spec.mode', 'oco'),
        item('label_spec.take_profit', 0.03),
        item('label_spec.stop_loss', 0.02),
        item('label_spec.max_hold', 8),
        item('predictor.params.label_type', 'reg'),
        item('strategy.params.family_hint', 'trend_following'),
      ]),
    )
    expect(result.values).toMatchObject({
      label_mode: 'oco',
      oco_take_profit_pct: 3,
      oco_stop_loss_pct: 2,
      oco_max_hold: 8,
      objective: 'regression',
    })
    expect(result.unmapped).toHaveLength(1)
  })

  it('does not map strong hyper-parameters for degraded suggestions', () => {
    const result = mapSuggestionToFormValues(
      suggestion([item('label_spec.mode', 'oco'), item('label_spec.take_profit', 0.03)], true),
    )
    expect(result.values).toEqual({})
    expect(result.unmapped).toHaveLength(2)
  })

  it('checks observation drop threshold boundaries', () => {
    expect(shouldDropObservation(0.59, 0.5)).toBe(true)
    expect(shouldDropObservation(0.6, 0.04)).toBe(true)
    expect(shouldDropObservation(0.6, 0.05)).toBe(false)
  })

  it('builds request and flattens observation groups with dedupe', () => {
    expect(
      buildProfilingRequest({
        targetSymbol: 'AAA.SSE',
        interval: '30m',
        asOf: '2024-01-01T00:00:00',
        lookbackDays: 10,
        observationGroups: [
          { role: 'market', name: 'm', symbols: ['BBB.SSE', 'AAA.SSE'] },
          { role: 'sector', name: 's', symbols: ['BBB.SSE', 'CCC.SSE'] },
        ],
      }),
    ).toMatchObject({
      vt_symbol: 'AAA.SSE',
      interval: '30m',
      as_of: '2024-01-01T00:00:00',
      lookback_days: 10,
      observation_symbols: ['BBB.SSE', 'CCC.SSE'],
      persist: true,
    })
  })

  it('can explicitly disable profile persistence', () => {
    expect(
      buildProfilingRequest({
        targetSymbol: 'AAA.SSE',
        interval: '30m',
        asOf: '2024-01-01T00:00:00',
        lookbackDays: 10,
        persist: false,
      }).persist,
    ).toBe(false)
  })
})
