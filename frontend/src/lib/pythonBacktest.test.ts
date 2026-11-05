import { describe, expect, it } from 'vitest'
import {
  backtestFormError,
  backtestIsActive,
  backtestStatusText,
  defaultBacktestParams,
  exitReasonText,
  fixed,
  istDate,
  mergeBacktestParams,
  serverMessage,
} from './pythonBacktest'

// 2026-11-05 20:00 UTC is already 6 November in India.
const LATE_EVENING_UTC = new Date('2026-11-05T20:00:00Z')

describe('istDate', () => {
  it('uses the Indian calendar date, not the browser zone', () => {
    expect(istDate(LATE_EVENING_UTC)).toBe('2026-11-06')
  })
})

describe('defaultBacktestParams', () => {
  it('covers the last 30 days up to today in India', () => {
    const p = defaultBacktestParams(LATE_EVENING_UTC)
    expect(p.end_date).toBe('2026-11-06')
    expect(p.start_date).toBe('2026-10-07')
    expect(p.capital).toBe(100000)
    expect(p.brokerage_per_order).toBe(20)
    expect(p.slippage_bps).toBe(2)
  })
})

describe('mergeBacktestParams', () => {
  it('keeps the defaults where the run left a value out', () => {
    const merged = mergeBacktestParams(defaultBacktestParams(LATE_EVENING_UTC), {
      start_date: '2026-01-01',
      capital: undefined,
    })
    expect(merged.start_date).toBe('2026-01-01')
    expect(merged.capital).toBe(100000)
  })
})

describe('backtestFormError', () => {
  const ok = { ...defaultBacktestParams(LATE_EVENING_UTC) }
  const check = (patch: object) => backtestFormError({ ...ok, ...patch }, LATE_EVENING_UTC)

  it('accepts the defaults', () => {
    expect(check({})).toBeNull()
  })
  it('accepts a single day', () => {
    expect(check({ start_date: '2026-11-02', end_date: '2026-11-02' })).toBeNull()
  })
  it('refuses missing dates', () => {
    expect(check({ start_date: '' })).toMatch(/start date and an end date/)
  })
  it('refuses the future', () => {
    expect(check({ end_date: '2026-11-07' })).toMatch(/future/)
  })
  it('refuses a start after the end', () => {
    expect(check({ start_date: '2026-11-03', end_date: '2026-11-02' })).toMatch(/on or before/)
  })
  it('refuses more than a year', () => {
    expect(check({ start_date: '2025-01-01', end_date: '2026-11-02' })).toMatch(/at most 366/)
  })
  it('refuses a tiny or missing capital', () => {
    expect(check({ capital: 10 })).toMatch(/Capital/)
    expect(check({ capital: Number.NaN })).toMatch(/Capital/)
  })
  it('refuses negative costs', () => {
    expect(check({ brokerage_per_order: -1 })).toMatch(/Brokerage/)
    expect(check({ slippage_bps: 600 })).toMatch(/Slippage/)
  })
})

describe('labels', () => {
  it('reads statuses and exit reasons the way a trader would', () => {
    expect(backtestStatusText('completed')).toBe('Finished')
    expect(backtestStatusText('stopped')).toBe('Stopped')
    expect(exitReasonText('square-off')).toBe('MIS square-off')
    expect(exitReasonText('stop')).toBe('Stop loss order')
    expect(exitReasonText('strategy')).toBe('Strategy exit')
  })
  it('polls only while a run is in progress', () => {
    expect(backtestIsActive('running')).toBe(true)
    expect(backtestIsActive('stopping')).toBe(true)
    expect(backtestIsActive('completed')).toBe(false)
    expect(backtestIsActive(undefined)).toBe(false)
  })
  it('shows a dash for missing numbers', () => {
    expect(fixed(null)).toBe('-')
    expect(fixed(1.234)).toBe('1.23')
  })
  it('takes the server message from an error when there is one', () => {
    expect(serverMessage({ response: { data: { message: 'Busy' } } }, 'x')).toBe('Busy')
    expect(serverMessage(new Error('boom'), 'Try again')).toBe('Try again')
  })
})
