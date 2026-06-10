import { describe, expect, it } from 'vitest'
import fc from 'fast-check'

import type { ConfidenceLevel, SchemeSuggestion, SuggestionItem } from '../types/alpha'
import {
  CONFIDENCE_ORDER,
  buildProfilingRequest,
  confidenceStyle,
  formatMetricValue,
  isLowConfidenceItem,
  mapSuggestionToFormValues,
  shouldDropObservation,
} from './profiling'

const confidenceArb = fc.constantFrom<ConfidenceLevel>('high', 'medium', 'low', 'insufficient')

function item(field: string, value: unknown, confidence: ConfidenceLevel = 'high'): SuggestionItem {
  return {
    field,
    value,
    reason: 'rule hit',
    based_on_confidence: confidence,
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

describe('profiling pure-function properties', () => {
  // Feature: symbol-profiling-ui, Property 1: 置信度样式覆盖且 insufficient 必弱化
  // Validates: Requirements 3.3
  it('maps every confidence level and weakens insufficient', () => {
    fc.assert(
      fc.property(confidenceArb, (confidence) => {
        const style = confidenceStyle(confidence)
        expect(style.text.length).toBeGreaterThan(0)
        expect(style.color.length).toBeGreaterThan(0)
        if (confidence === 'insufficient') {
          expect(style.weak).toBe(true)
        }
      }),
      { numRuns: 100 },
    )
  })

  // Feature: symbol-profiling-ui, Property 2: 被抑制指标渲染为占位符而非数值
  // Validates: Requirements 3.2
  it('formats null as placeholder', () => {
    fc.assert(
      fc.property(fc.constant(null), (value) => {
        expect(formatMetricValue(value)).toBe('-')
        expect(formatMetricValue(value)).not.toBe('0')
      }),
      { numRuns: 100 },
    )
  })

  // Feature: symbol-profiling-ui, Property 3: 建议→表单映射不越界且只取可映射字段
  // Validates: Requirements 4.4
  it('only maps allowlisted suggestion fields', () => {
    const allowed = new Set([
      'label_mode',
      'oco_take_profit_pct',
      'oco_stop_loss_pct',
      'oco_max_hold',
      'label_horizon',
      'objective',
    ])
    fc.assert(
      fc.property(
        fc.array(
          fc.record({
            field: fc.string(),
            value: fc.oneof(fc.string(), fc.float({ noNaN: true }), fc.boolean()),
            confidence: confidenceArb,
          }),
          { maxLength: 12 },
        ),
        (rows) => {
          const mapped = mapSuggestionToFormValues(
            suggestion(rows.map((row) => item(row.field, row.value, row.confidence))),
          )
          for (const key of Object.keys(mapped.values)) {
            expect(allowed.has(key)).toBe(true)
          }
        },
      ),
      { numRuns: 100 },
    )
  })

  // Feature: symbol-profiling-ui, Property 4: 降级建议不含强超参建议
  // Validates: Requirements 4.3
  it('drops strong hyper-parameter fields when degraded', () => {
    fc.assert(
      fc.property(fc.double({ min: 0.001, max: 0.5, noNaN: true }), (value) => {
        const mapped = mapSuggestionToFormValues(
          suggestion(
            [
              item('label_spec.mode', 'oco'),
              item('label_spec.take_profit', value),
              item('label_spec.stop_loss', value),
              item('label_spec.max_hold', 10),
            ],
            true,
          ),
        )
        expect(mapped.values).toEqual({})
      }),
      { numRuns: 100 },
    )
  })

  // Feature: symbol-profiling-ui, Property 5: 低置信条目被标注
  // Validates: Requirements 4.6
  it('marks low-confidence suggestion items', () => {
    fc.assert(
      fc.property(confidenceArb, (confidence) => {
        const low = CONFIDENCE_ORDER[confidence] <= CONFIDENCE_ORDER.low
        expect(isLowConfidenceItem(item('label_spec.mode', 'oco', confidence))).toBe(low)
      }),
      { numRuns: 100 },
    )
  })

  // Feature: symbol-profiling-ui, Property 6: 可剔除判定的单调性
  // Validates: Requirements 5.3
  it('does not become more likely to drop as coverage and correlation improve', () => {
    fc.assert(
      fc.property(
        fc.float({ min: 0, max: 1, noNaN: true }),
        fc.float({ min: 0, max: 1, noNaN: true }),
        fc.float({ min: 0, max: 1, noNaN: true }),
        fc.float({ min: 0, max: 1, noNaN: true }),
        (aCov, bCov, aCorr, bCorr) => {
          const lowCov = Math.min(aCov, bCov)
          const highCov = Math.max(aCov, bCov)
          const lowCorr = Math.min(aCorr, bCorr)
          const highCorr = Math.max(aCorr, bCorr)
          const bad = shouldDropObservation(lowCov, lowCorr)
          const better = shouldDropObservation(highCov, highCorr)
          if (!bad) {
            expect(better).toBe(false)
          }
        },
      ),
      { numRuns: 100 },
    )
  })

  // Feature: symbol-profiling-ui, Property 7: 画像请求组装正确
  // Validates: Requirements 2.2, 5.1
  it('builds profiling requests with explicit as_of and deduplicated observations', () => {
    fc.assert(
      fc.property(
        fc.array(fc.string({ minLength: 1, maxLength: 8 }), { maxLength: 12 }),
        (symbols) => {
          const req = buildProfilingRequest({
            targetSymbol: 'TARGET',
            interval: '30m',
            asOf: '2024-01-01T09:30:00',
            lookbackDays: 250.9,
            observationGroups: [{ role: 'market', name: 'm', symbols: ['TARGET', ...symbols] }],
          })
          expect(req.as_of).toBe('2024-01-01T09:30:00')
          expect(req.lookback_days).toBe(250)
          expect(req.persist).toBe(true)
          expect(req.observation_symbols).not.toContain('TARGET')
          expect(new Set(req.observation_symbols).size).toBe(req.observation_symbols?.length)
        },
      ),
      { numRuns: 100 },
    )
  })
})
