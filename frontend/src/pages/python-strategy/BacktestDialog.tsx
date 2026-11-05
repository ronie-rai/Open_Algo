import { FlaskConical, Loader2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router'
import { pythonStrategyApi } from '@/api/python-strategy'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  backtestFormError,
  defaultBacktestParams,
  mergeBacktestParams,
  serverMessage,
} from '@/lib/pythonBacktest'
import type { BacktestParams } from '@/types/python-backtest'

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
  strategyId: string
  strategyName: string
  /** Start from these values instead of the last 30 days, e.g. "run again". */
  initial?: Partial<BacktestParams>
}

/**
 * Asks for the test period and the cost assumptions, starts the backtest and
 * goes to its results page. The strategy itself runs unchanged.
 */
export function BacktestDialog({ open, onOpenChange, strategyId, strategyName, initial }: Props) {
  const navigate = useNavigate()
  const [form, setForm] = useState<BacktestParams>(() =>
    mergeBacktestParams(defaultBacktestParams(), initial)
  )
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  // biome-ignore lint/correctness/useExhaustiveDependencies: reset only when the dialog opens; `initial` is a fresh object on every parent render
  useEffect(() => {
    if (open) {
      setForm(mergeBacktestParams(defaultBacktestParams(), initial))
      setError(null)
    }
  }, [open])

  const setNumber = (key: keyof BacktestParams) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [key]: e.target.value === '' ? Number.NaN : Number(e.target.value) }))
  const setText = (key: keyof BacktestParams) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((f) => ({ ...f, [key]: e.target.value }))

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    const problem = backtestFormError(form)
    if (problem) {
      setError(problem)
      return
    }
    setSubmitting(true)
    setError(null)
    try {
      const response = await pythonStrategyApi.startBacktest(strategyId, form)
      const runId = response.data?.run_id
      if (response.status !== 'success' || !runId) {
        setError(response.message || 'The backtest could not be started.')
        return
      }
      onOpenChange(false)
      navigate(`/python/backtest/${strategyId}/${runId}`)
    } catch (err) {
      setError(serverMessage(err, 'The backtest could not be started. Try again in a moment.'))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <form onSubmit={submit} className="space-y-4">
          <DialogHeader>
            <DialogTitle>Backtest {strategyName}</DialogTitle>
            <DialogDescription>
              Replays your script, unchanged, on historical one-minute bars. Orders are simulated
              and never reach your broker.
            </DialogDescription>
          </DialogHeader>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="bt-start">Start date</Label>
              <Input
                id="bt-start"
                type="date"
                value={form.start_date}
                onChange={setText('start_date')}
                required
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="bt-end">End date</Label>
              <Input
                id="bt-end"
                type="date"
                value={form.end_date}
                onChange={setText('end_date')}
                required
              />
            </div>
            <div className="space-y-1.5 col-span-2">
              <Label htmlFor="bt-capital">Capital</Label>
              <Input
                id="bt-capital"
                type="number"
                min={1000}
                step="any"
                value={Number.isNaN(form.capital) ? '' : form.capital}
                onChange={setNumber('capital')}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="bt-brokerage">Brokerage per order</Label>
              <Input
                id="bt-brokerage"
                type="number"
                min={0}
                step="any"
                value={Number.isNaN(form.brokerage_per_order) ? '' : form.brokerage_per_order}
                onChange={setNumber('brokerage_per_order')}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="bt-slippage">Slippage (basis points)</Label>
              <Input
                id="bt-slippage"
                type="number"
                min={0}
                max={500}
                step="any"
                value={Number.isNaN(form.slippage_bps) ? '' : form.slippage_bps}
                onChange={setNumber('slippage_bps')}
              />
            </div>
          </div>

          <p className="text-xs text-muted-foreground">
            Brokerage is this amount or 0.03% of the order value, whichever is lower. STT and
            exchange charges are added at an intraday equity rate. Data comes from Historify first,
            then from your broker.
          </p>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={submitting}>
              {submitting ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <FlaskConical className="h-4 w-4 mr-2" />
              )}
              Run backtest
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
