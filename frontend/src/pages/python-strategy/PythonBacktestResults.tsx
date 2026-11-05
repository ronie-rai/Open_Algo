import { useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertTriangle, ArrowLeft, FlaskConical, Loader2, Square } from 'lucide-react'
import { useState } from 'react'
import { Link, useParams } from 'react-router'
import { pythonStrategyApi } from '@/api/python-strategy'
import { BacktestChart } from '@/components/trading/BacktestChart'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { LogViewer } from '@/components/ui/log-viewer'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  backtestIsActive,
  backtestStatusText,
  exitReasonText,
  fixed,
  serverMessage,
} from '@/lib/pythonBacktest'
import { momentText, moneyText, signedMoney, signedPercent } from '@/lib/trading/backtestFormat'
import type { BacktestMetrics, BacktestRun, BacktestRunSummary } from '@/types/python-backtest'
import { showToast } from '@/utils/toast'
import { BacktestDialog } from './BacktestDialog'

const TZ = 'Asia/Kolkata'
const TRADE_ROWS = 500

function tone(value: number | null | undefined): string {
  if (value === null || value === undefined || value === 0) return ''
  return value > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
}

function statusVariant(status: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  if (status === 'completed') return 'default'
  if (status === 'failed') return 'destructive'
  if (status === 'running' || status === 'stopping') return 'secondary'
  return 'outline'
}

function Tile({ label, value, className }: { label: string; value: string; className?: string }) {
  return (
    <Card>
      <CardContent className="pt-4">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className={`text-xl font-semibold tabular-nums ${className ?? ''}`}>{value}</p>
      </CardContent>
    </Card>
  )
}

function MetricTiles({ m }: { m: BacktestMetrics }) {
  return (
    <div className="grid gap-3 grid-cols-2 md:grid-cols-4">
      <Tile label="Net P&L" value={signedMoney(m.net_pnl)} className={tone(m.net_pnl)} />
      <Tile label="Return" value={signedPercent(m.return_pct)} className={tone(m.return_pct)} />
      <Tile label="Trades" value={String(m.trades)} />
      <Tile label="Win rate" value={m.win_rate === null ? '-' : `${fixed(m.win_rate, 1)}%`} />
      <Tile label="Profit factor" value={fixed(m.profit_factor)} />
      <Tile
        label="Max drawdown"
        value={`${moneyText(m.max_drawdown)} (${fixed(m.max_drawdown_pct)}%)`}
        className={m.max_drawdown > 0 ? 'text-red-600 dark:text-red-400' : ''}
      />
      <Tile label="Sharpe (daily, annualised)" value={fixed(m.sharpe)} />
      <Tile label="Total costs" value={moneyText(m.total_costs)} />
    </div>
  )
}

function DetailRows({ m }: { m: BacktestMetrics }) {
  const rows: [string, string][] = [
    ['Starting capital', moneyText(m.starting_capital)],
    ['Final equity', moneyText(m.final_equity)],
    ['Gross P&L (before costs)', signedMoney(m.gross_pnl)],
    ['Winning / losing trades', `${m.winning_trades} / ${m.losing_trades}`],
    ['Average win', m.avg_win === null ? '-' : moneyText(m.avg_win)],
    ['Average loss', m.avg_loss === null ? '-' : moneyText(m.avg_loss)],
    ['Expectancy per trade', m.expectancy === null ? '-' : signedMoney(m.expectancy)],
    ['Largest win', m.largest_win === null ? '-' : moneyText(m.largest_win)],
    ['Largest loss', m.largest_loss === null ? '-' : moneyText(m.largest_loss)],
    ['Max drawdown at', m.max_drawdown_time ? momentText(m.max_drawdown_time / 1000, TZ) : '-'],
    ['Time in the market', `${fixed(m.exposure_pct, 1)}%`],
    ['Trading days', String(m.trading_days)],
    ['Best day', m.best_day ? `${m.best_day.date}  ${signedMoney(m.best_day.pnl)}` : '-'],
    ['Worst day', m.worst_day ? `${m.worst_day.date}  ${signedMoney(m.worst_day.pnl)}` : '-'],
  ]
  return (
    <div className="grid gap-x-8 gap-y-1 sm:grid-cols-2 text-sm">
      {rows.map(([label, value]) => (
        <div key={label} className="flex justify-between border-b border-border/50 py-1">
          <span className="text-muted-foreground">{label}</span>
          <span className="tabular-nums">{value}</span>
        </div>
      ))}
    </div>
  )
}

function RunHistory({
  strategyId,
  runs,
  current,
}: {
  strategyId: string
  runs: BacktestRunSummary[]
  current?: string
}) {
  if (runs.length === 0) {
    return <p className="text-sm text-muted-foreground">No backtests yet.</p>
  }
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Run</TableHead>
            <TableHead>Period</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Net P&L</TableHead>
            <TableHead className="text-right">Return</TableHead>
            <TableHead className="text-right">Trades</TableHead>
            <TableHead className="text-right">Max DD</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((r) => (
            <TableRow key={r.run_id} className={r.run_id === current ? 'bg-muted/50' : ''}>
              <TableCell>
                <Link
                  to={`/python/backtest/${strategyId}/${r.run_id}`}
                  className="text-primary hover:underline"
                >
                  {r.created_at ? r.created_at.replace('T', ' ').slice(0, 16) : r.run_id}
                </Link>
              </TableCell>
              <TableCell className="whitespace-nowrap">
                {r.start_date} to {r.end_date}
              </TableCell>
              <TableCell>
                <Badge variant={statusVariant(r.status)}>{backtestStatusText(r.status)}</Badge>
              </TableCell>
              <TableCell className={`text-right tabular-nums ${tone(r.net_pnl)}`}>
                {r.net_pnl === null ? '-' : signedMoney(r.net_pnl)}
              </TableCell>
              <TableCell className={`text-right tabular-nums ${tone(r.return_pct)}`}>
                {r.return_pct === null ? '-' : signedPercent(r.return_pct)}
              </TableCell>
              <TableCell className="text-right tabular-nums">{r.trades ?? '-'}</TableCell>
              <TableCell className="text-right tabular-nums">
                {r.max_drawdown_pct === null ? '-' : `${fixed(r.max_drawdown_pct)}%`}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function Results({ run }: { run: BacktestRun }) {
  const result = run.result
  const m = result?.metrics
  if (!result || !m) return null
  const trades = result.trades ?? []
  const daily = result.daily ?? []
  const curve = result.equity_curve ?? []

  return (
    <div className="space-y-4">
      <MetricTiles m={m} />
      <Tabs defaultValue="equity">
        <TabsList className="flex-wrap h-auto">
          <TabsTrigger value="equity">Equity</TabsTrigger>
          <TabsTrigger value="drawdown">Drawdown</TabsTrigger>
          <TabsTrigger value="daily">Daily P&L</TabsTrigger>
          <TabsTrigger value="trades">Trades ({trades.length})</TabsTrigger>
          <TabsTrigger value="details">Details</TabsTrigger>
        </TabsList>
        <TabsContent value="equity">
          {curve.length >= 2 ? (
            <BacktestChart points={curve} show="equity" className="h-80" />
          ) : (
            <p className="text-sm text-muted-foreground py-6">
              Not enough points to draw an equity curve.
            </p>
          )}
        </TabsContent>
        <TabsContent value="drawdown">
          {curve.length >= 2 ? (
            <BacktestChart points={curve} show="drawdown" className="h-80" />
          ) : (
            <p className="text-sm text-muted-foreground py-6">No drawdown to show.</p>
          )}
        </TabsContent>
        <TabsContent value="daily">
          <div className="max-h-[28rem] overflow-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Date</TableHead>
                  <TableHead className="text-right">P&L</TableHead>
                  <TableHead className="text-right">Equity at close</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {daily.map((d) => (
                  <TableRow key={d.date}>
                    <TableCell>{d.date}</TableCell>
                    <TableCell className={`text-right tabular-nums ${tone(d.pnl)}`}>
                      {signedMoney(d.pnl)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{moneyText(d.equity)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </TabsContent>
        <TabsContent value="trades">
          {trades.length === 0 ? (
            <p className="text-sm text-muted-foreground py-6">
              The strategy did not complete a trade in this period.
            </p>
          ) : (
            <div className="max-h-[28rem] overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Side</TableHead>
                    <TableHead className="text-right">Qty</TableHead>
                    <TableHead>Entry</TableHead>
                    <TableHead className="text-right">Entry price</TableHead>
                    <TableHead>Exit</TableHead>
                    <TableHead className="text-right">Exit price</TableHead>
                    <TableHead className="text-right">Costs</TableHead>
                    <TableHead className="text-right">P&L</TableHead>
                    <TableHead>Exit by</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {trades.slice(-TRADE_ROWS).map((t) => (
                    <TableRow key={`${t.entry_time}-${t.exit_time}-${t.symbol}-${t.quantity}`}>
                      <TableCell>{t.symbol}</TableCell>
                      <TableCell>{t.side === 'LONG' ? 'Long' : 'Short'}</TableCell>
                      <TableCell className="text-right tabular-nums">{t.quantity}</TableCell>
                      <TableCell className="whitespace-nowrap">
                        {momentText(t.entry_time / 1000, TZ)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {fixed(t.entry_price)}
                      </TableCell>
                      <TableCell className="whitespace-nowrap">
                        {momentText(t.exit_time / 1000, TZ)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {fixed(t.exit_price)}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{fixed(t.costs)}</TableCell>
                      <TableCell className={`text-right tabular-nums ${tone(t.pnl)}`}>
                        {signedMoney(t.pnl)}
                      </TableCell>
                      <TableCell>{exitReasonText(t.exit_reason)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              {trades.length > TRADE_ROWS && (
                <p className="text-xs text-muted-foreground pt-2">
                  Showing the last {TRADE_ROWS} of {trades.length} trades.
                </p>
              )}
            </div>
          )}
        </TabsContent>
        <TabsContent value="details" className="space-y-4">
          <DetailRows m={m} />
          <div className="text-xs text-muted-foreground space-y-1">
            <p>
              How the test works: the script runs unchanged against one-minute bars. Market orders
              fill at the price of the moment plus slippage, limit and stop orders fill when a later
              bar trades through them, and MIS positions are squared off at the square-off time set
              on the Sandbox page. Positions still open at the end are closed at the last price.
            </p>
            <p>
              Costs: brokerage per order (or 0.03% of the order value, whichever is lower), STT of
              0.025% on sells and 0.005% for exchange and other charges. Live streaming is not
              available inside a backtest.
            </p>
            {result.data?.sources?.length ? (
              <p>Data from: {result.data.sources.join(', ')}.</p>
            ) : null}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default function PythonBacktestResults() {
  const { strategyId = '', runId } = useParams<{ strategyId: string; runId?: string }>()
  const queryClient = useQueryClient()
  const [dialogOpen, setDialogOpen] = useState(false)
  const [stopping, setStopping] = useState(false)

  const runQuery = useQuery({
    queryKey: ['python-backtest', strategyId, runId],
    queryFn: () => pythonStrategyApi.getBacktestRun(strategyId, runId as string),
    enabled: Boolean(strategyId && runId),
    refetchInterval: (query) => (backtestIsActive(query.state.data?.status) ? 2000 : false),
  })
  const run = runQuery.data
  const active = backtestIsActive(run?.status)

  const runsQuery = useQuery({
    queryKey: ['python-backtest-runs', strategyId, run?.status],
    queryFn: () => pythonStrategyApi.getBacktestRuns(strategyId),
    enabled: Boolean(strategyId),
  })

  const handleStop = async () => {
    if (!runId) return
    setStopping(true)
    try {
      const response = await pythonStrategyApi.stopBacktest(strategyId, runId)
      showToast.success(response.message || 'The backtest was stopped.', 'pythonStrategy')
    } catch (err) {
      showToast.error(serverMessage(err, 'The backtest could not be stopped.'), 'pythonStrategy')
    } finally {
      setStopping(false)
      await queryClient.invalidateQueries({ queryKey: ['python-backtest', strategyId, runId] })
    }
  }

  const name = run?.strategy_name || runsQuery.data?.strategy_name || strategyId
  const progress = run?.progress
  const percent = run?.status === 'completed' ? 100 : (progress?.percent ?? 0)

  return (
    <div className="container mx-auto py-6 space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="space-y-1">
          <Button variant="ghost" size="sm" asChild className="-ml-3">
            <Link to="/python">
              <ArrowLeft className="h-4 w-4 mr-1" />
              Python Strategies
            </Link>
          </Button>
          <h1 className="text-2xl font-bold tracking-tight">Backtest: {name}</h1>
          {run && (
            <p className="text-sm text-muted-foreground">
              {run.config.start_date} to {run.config.end_date}
              {run.config.capital ? `, capital ${moneyText(run.config.capital)}` : ''}
              {run.config.file_name ? `, ${run.config.file_name}` : ''}
            </p>
          )}
        </div>
        <div className="flex gap-2">
          {active && (
            <Button variant="destructive" onClick={handleStop} disabled={stopping}>
              {stopping ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Square className="h-4 w-4 mr-2" />
              )}
              Stop
            </Button>
          )}
          <Button onClick={() => setDialogOpen(true)} disabled={active}>
            <FlaskConical className="h-4 w-4 mr-2" />
            New backtest
          </Button>
        </div>
      </div>

      {runId && runQuery.isLoading && <Skeleton className="h-40" />}
      {runId && runQuery.isError && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            {serverMessage(runQuery.error, 'This backtest could not be loaded.')}
          </AlertDescription>
        </Alert>
      )}

      {run && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between gap-2">
              <CardTitle className="text-base">Status</CardTitle>
              <Badge variant={statusVariant(run.status)}>{backtestStatusText(run.status)}</Badge>
            </div>
            {run.message && <CardDescription>{run.message}</CardDescription>}
          </CardHeader>
          <CardContent className="space-y-2">
            <Progress value={percent} />
            <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm text-muted-foreground">
              {progress?.sim_time && <span>Simulated time {progress.sim_time} IST</span>}
              <span>{fixed(percent, 0)}% of the period</span>
              {progress && <span>{progress.trades} trades</span>}
              {progress && active && <span>Equity {moneyText(progress.equity)}</span>}
            </div>
          </CardContent>
        </Card>
      )}

      {run && !active && <Results run={run} />}

      {run && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Strategy log</CardTitle>
            <CardDescription>
              The last lines your script printed during the backtest.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <LogViewer value={run.log_tail || 'Nothing printed yet.'} height="320px" followTail />
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Past backtests</CardTitle>
        </CardHeader>
        <CardContent>
          {runsQuery.isLoading ? (
            <Skeleton className="h-24" />
          ) : (
            <RunHistory strategyId={strategyId} runs={runsQuery.data?.runs ?? []} current={runId} />
          )}
        </CardContent>
      </Card>

      <BacktestDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        strategyId={strategyId}
        strategyName={name}
        initial={
          run?.config
            ? {
                start_date: run.config.start_date,
                end_date: run.config.end_date,
                capital: run.config.capital,
                brokerage_per_order: run.config.brokerage_per_order,
                slippage_bps: run.config.slippage_bps,
              }
            : undefined
        }
      />
    </div>
  )
}
