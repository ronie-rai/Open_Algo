import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Route, Routes } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { pythonStrategyApi } from '@/api/python-strategy'
import { render, screen, userEvent, waitFor } from '@/test/test-utils'
import type { BacktestRun } from '@/types/python-backtest'
import PythonBacktestResults from './PythonBacktestResults'

vi.mock('@/api/python-strategy', () => ({
  pythonStrategyApi: {
    getBacktestRun: vi.fn(),
    getBacktestRuns: vi.fn(),
    stopBacktest: vi.fn(),
    startBacktest: vi.fn(),
  },
}))
vi.mock('@/utils/toast', () => ({
  showToast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}))
// The chart engine and the code viewer are not what is under test.
vi.mock('@/components/trading/BacktestChart', () => ({
  BacktestChart: () => <div data-testid="chart" />,
}))
vi.mock('@/components/ui/log-viewer', () => ({
  LogViewer: (props: { value: string }) => <pre>{props.value}</pre>,
}))

const api = vi.mocked(pythonStrategyApi)

const finished: BacktestRun = {
  run_id: '20261105-101500-abcdef',
  strategy_id: 's1',
  strategy_name: 'EMA Bot',
  status: 'completed',
  message: '',
  config: { start_date: '2026-10-01', end_date: '2026-10-31', capital: 100000 },
  progress: {
    status: 'finished',
    sim_time: '2026-10-31 15:30',
    percent: 100,
    days_done: 22,
    trades: 2,
    orders: 4,
    equity: 101500,
  },
  log_tail: 'Starting strategy',
  result: {
    status: 'completed',
    message: '',
    metrics: {
      starting_capital: 100000,
      final_equity: 101500,
      net_pnl: 1500,
      gross_pnl: 1600,
      total_costs: 100,
      return_pct: 1.5,
      trades: 2,
      winning_trades: 1,
      losing_trades: 1,
      win_rate: 50,
      avg_win: 2000,
      avg_loss: -500,
      profit_factor: 4,
      expectancy: 750,
      largest_win: 2000,
      largest_loss: -500,
      max_drawdown: 500,
      max_drawdown_pct: 0.5,
      max_drawdown_time: null,
      sharpe: 1.23,
      exposure_pct: 30,
      trading_days: 22,
      best_day: null,
      worst_day: null,
    },
    equity_curve: [],
    daily: [],
    trades: [
      {
        symbol: 'SBIN',
        exchange: 'NSE',
        product: 'MIS',
        side: 'LONG',
        quantity: 10,
        entry_time: 1790000000000,
        entry_price: 800,
        exit_time: 1790000600000,
        exit_price: 1000,
        gross_pnl: 2000,
        costs: 40,
        pnl: 1960,
        exit_reason: 'square-off',
      },
    ],
  },
}

function renderAt(path: string) {
  window.history.pushState(null, '', path)
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <Routes>
        <Route path="/python/backtest/:strategyId/:runId" element={<PythonBacktestResults />} />
      </Routes>
    </QueryClientProvider>
  )
}

beforeEach(() => {
  vi.resetAllMocks()
  api.getBacktestRuns.mockResolvedValue({ strategy_name: 'EMA Bot', active_run_id: null, runs: [] })
})

describe('PythonBacktestResults', () => {
  it('shows the headline numbers and the trades of a finished run', async () => {
    api.getBacktestRun.mockResolvedValue(finished)
    renderAt('/python/backtest/s1/20261105-101500-abcdef')
    expect(await screen.findByText('Backtest: EMA Bot')).toBeInTheDocument()
    expect(screen.getByText('Finished')).toBeInTheDocument()
    expect(screen.getByText('+1.50%')).toBeInTheDocument()
    expect(screen.getByText('Profit factor')).toBeInTheDocument()
    expect(screen.getByText('1.23')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('tab', { name: /Trades/ }))
    expect(await screen.findByText('MIS square-off')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Stop/ })).not.toBeInTheDocument()
  })

  it('shows progress and a Stop button while it runs', async () => {
    api.getBacktestRun.mockResolvedValue({
      ...finished,
      status: 'running',
      result: undefined,
      progress: { ...finished.progress!, percent: 42, sim_time: '2026-10-14 11:05' },
    } as BacktestRun)
    api.stopBacktest.mockResolvedValue({ status: 'success', message: 'The backtest was stopped.' })
    renderAt('/python/backtest/s1/20261105-101500-abcdef')
    expect(await screen.findByText('Running')).toBeInTheDocument()
    expect(screen.getByText(/Simulated time 2026-10-14 11:05 IST/)).toBeInTheDocument()
    expect(screen.queryByText('Profit factor')).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /Stop/ }))
    await waitFor(() =>
      expect(api.stopBacktest).toHaveBeenCalledWith('s1', '20261105-101500-abcdef')
    )
  })

  it('says what went wrong when a run failed', async () => {
    api.getBacktestRun.mockResolvedValue({
      ...finished,
      status: 'failed',
      message: 'No historical data was found for SBIN on NSE.',
      result: { status: 'failed', message: 'No historical data was found for SBIN on NSE.' },
    })
    renderAt('/python/backtest/s1/20261105-101500-abcdef')
    expect(
      await screen.findByText('No historical data was found for SBIN on NSE.')
    ).toBeInTheDocument()
    expect(screen.getByText('Failed')).toBeInTheDocument()
  })
})
