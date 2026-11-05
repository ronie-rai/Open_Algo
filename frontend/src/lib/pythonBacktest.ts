/**
 * Small pure helpers for the /python backtest dialog and results page.
 */

import type { BacktestParams, BacktestStatus } from '@/types/python-backtest'

/** The longest range the server accepts, in days, both ends included. */
export const MAX_BACKTEST_DAYS = 366

/** A date as YYYY-MM-DD in India, whatever zone the browser is in. */
export function istDate(at: Date): string {
  // en-CA formats as YYYY-MM-DD.
  return new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Kolkata',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).format(at)
}

/** The dialog's starting values: the last 30 days, ending today. */
export function defaultBacktestParams(now: Date = new Date()): BacktestParams {
  const end = istDate(now)
  const start = istDate(new Date(now.getTime() - 30 * 86_400_000))
  return {
    start_date: start,
    end_date: end,
    capital: 100000,
    brokerage_per_order: 20,
    slippage_bps: 2,
  }
}

/** The defaults, overridden by whichever of `initial` are actually set. */
export function mergeBacktestParams(
  defaults: BacktestParams,
  initial?: Partial<BacktestParams>
): BacktestParams {
  const merged = { ...defaults }
  for (const [key, value] of Object.entries(initial ?? {})) {
    if (value !== undefined && value !== null && value !== '') {
      ;(merged as Record<string, unknown>)[key] = value
    }
  }
  return merged
}

function daysBetween(start: string, end: string): number {
  const a = Date.parse(`${start}T00:00:00Z`)
  const b = Date.parse(`${end}T00:00:00Z`)
  return Math.round((b - a) / 86_400_000)
}

/** What is wrong with the form, in words, or null when it can be sent. */
export function backtestFormError(form: BacktestParams, now: Date = new Date()): string | null {
  const datePattern = /^\d{4}-\d{2}-\d{2}$/
  if (!datePattern.test(form.start_date) || !datePattern.test(form.end_date)) {
    return 'Choose a start date and an end date.'
  }
  if (form.end_date > istDate(now)) return 'The end date cannot be in the future.'
  if (form.start_date > form.end_date) return 'The start date must be on or before the end date.'
  if (daysBetween(form.start_date, form.end_date) + 1 > MAX_BACKTEST_DAYS) {
    return `A backtest can cover at most ${MAX_BACKTEST_DAYS} days. Shorten the range.`
  }
  if (!Number.isFinite(form.capital) || form.capital < 1000) {
    return 'Capital must be at least 1,000.'
  }
  if (!Number.isFinite(form.brokerage_per_order) || form.brokerage_per_order < 0) {
    return 'Brokerage per order cannot be negative.'
  }
  if (!Number.isFinite(form.slippage_bps) || form.slippage_bps < 0 || form.slippage_bps > 500) {
    return 'Slippage must be between 0 and 500 basis points.'
  }
  return null
}

const STATUS_TEXT: Record<BacktestStatus, string> = {
  running: 'Running',
  stopping: 'Stopping',
  completed: 'Finished',
  failed: 'Failed',
  stopped: 'Stopped',
}

export function backtestStatusText(status: string): string {
  return STATUS_TEXT[status as BacktestStatus] ?? status
}

/** Whether the page should keep asking the server for news. */
export function backtestIsActive(status: string | undefined): boolean {
  return status === 'running' || status === 'stopping'
}

/** Why a trade ended, as a trader would say it. */
export function exitReasonText(reason: string): string {
  switch (reason) {
    case 'stop':
      return 'Stop loss order'
    case 'target':
      return 'Limit order'
    case 'square-off':
      return 'MIS square-off'
    case 'end of backtest':
      return 'End of test'
    default:
      return 'Strategy exit'
  }
}

/** A number that may be missing, as text: `-` where there is nothing to show. */
export function fixed(value: number | null | undefined, digits = 2): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? '-'
    : value.toFixed(digits)
}

/** Message from an axios error, or the fallback. */
export function serverMessage(error: unknown, fallback: string): string {
  const response = (error as { response?: { data?: { message?: unknown } } })?.response
  const message = response?.data?.message
  return typeof message === 'string' && message.trim() ? message : fallback
}
