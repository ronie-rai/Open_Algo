/**
 * A backtest of a /python strategy: what the page sends to start one and what
 * the server reports back while it runs and once it is done.
 */

export interface BacktestParams {
  start_date: string
  end_date: string
  capital: number
  brokerage_per_order: number
  slippage_bps: number
}

export type BacktestStatus = 'running' | 'stopping' | 'completed' | 'failed' | 'stopped'

export interface BacktestProgress {
  status: string
  sim_time: string
  percent: number
  days_done: number
  trades: number
  orders: number
  equity: number
}

export interface BacktestDay {
  date: string
  pnl: number
  equity: number
}

export interface BacktestMetrics {
  starting_capital: number
  final_equity: number
  net_pnl: number
  gross_pnl: number
  total_costs: number
  return_pct: number
  trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number | null
  avg_win: number | null
  avg_loss: number | null
  profit_factor: number | null
  expectancy: number | null
  largest_win: number | null
  largest_loss: number | null
  max_drawdown: number
  max_drawdown_pct: number
  max_drawdown_time: number | null
  sharpe: number | null
  exposure_pct: number
  trading_days: number
  best_day: BacktestDay | null
  worst_day: BacktestDay | null
}

export interface BacktestTrade {
  symbol: string
  exchange: string
  product: string
  side: 'LONG' | 'SHORT'
  quantity: number
  entry_time: number
  entry_price: number
  exit_time: number
  exit_price: number
  gross_pnl: number
  costs: number
  pnl: number
  exit_reason: string
}

export interface BacktestCurvePoint {
  time: number
  equity: number
  drawdown: number
}

export interface BacktestResult {
  status: BacktestStatus
  message: string
  metrics?: BacktestMetrics
  equity_curve?: BacktestCurvePoint[]
  daily?: BacktestDay[]
  trades?: BacktestTrade[]
  orders_total?: number
  simulated_until?: string
  wall_seconds?: number
  data?: {
    symbols: Record<string, { minute_bars: number; trading_days_in_period: number }>
    sources: string[]
  }
}

export interface BacktestRun {
  run_id: string
  strategy_id: string
  strategy_name?: string
  status: BacktestStatus
  message: string
  config: Partial<BacktestParams> & {
    exchange?: string
    file_name?: string
    created_at?: string
  }
  progress: BacktestProgress | null
  result?: BacktestResult
  log_tail: string
}

export interface BacktestRunSummary {
  run_id: string
  status: BacktestStatus
  created_at: string | null
  start_date: string | null
  end_date: string | null
  capital: number | null
  net_pnl: number | null
  return_pct: number | null
  trades: number | null
  win_rate: number | null
  max_drawdown_pct: number | null
}

export interface BacktestRunList {
  strategy_name: string
  active_run_id: string | null
  runs: BacktestRunSummary[]
}
