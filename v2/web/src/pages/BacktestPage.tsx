import { useEffect, useRef, useState } from 'react'
import { createChart, type IChartApi, LineSeries } from 'lightweight-charts'
import { runBacktest } from '../api'

export interface BacktestSeed {
  temperature: number
  horizon: number
}

interface EquityPoint {
  idx: number
  cum_log_pnl: number
  cum_ret_pct: number
  position: number
}

interface BacktestResult {
  temperature: number
  horizon: number
  z_threshold: number
  fee_bps: number
  n_windows: number
  trades: number
  longs: number
  shorts: number
  flats: number
  win_rate: number
  total_return_pct: number
  total_log_pnl: number
  sharpe_per_trade: number
  max_drawdown_pct: number
  equity: EquityPoint[]
  error?: string
}

const inputStyle: React.CSSProperties = {
  display: 'block', width: '100%', marginTop: 4,
  background: '#0b0e13', color: 'var(--fg)',
  border: '1px solid #1c2230', borderRadius: 4,
  padding: '6px 10px', fontFamily: 'var(--font-mono)', fontSize: 13,
  height: 32, boxSizing: 'border-box',
}

export function BacktestPage({ seed }: { seed: BacktestSeed | null }) {
  const [temperature, setTemperature] = useState(1.0)
  const [horizon, setHorizon] = useState(30)
  // Default 0 — this model's z-scores are tiny (it was trained for ~22k of
  // 200k planned steps), so anything above ~0.02 filters out every window
  // and the backtest takes 0 trades. Start at 0 (always-on) and the user
  // can dial up to find a confidence threshold that actually fires.
  const [zThreshold, setZThreshold] = useState(0.0)
  const [feeBps, setFeeBps] = useState(1.0)
  const [startFrac, setStartFrac] = useState(0)
  const [endFrac, setEndFrac] = useState(1)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [elapsedMs, setElapsedMs] = useState<number | null>(null)

  const chartRef = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)
  const series = useRef<any>(null)

  // Apply settings handed over from Calibration sweep.
  useEffect(() => {
    if (seed) {
      setTemperature(seed.temperature)
      setHorizon(seed.horizon)
    }
  }, [seed])

  useEffect(() => {
    if (!chartRef.current) return
    const c = createChart(chartRef.current, {
      layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
      grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
      rightPriceScale: { borderColor: '#252d3d' },
      timeScale: { borderColor: '#252d3d', timeVisible: false },
      width: chartRef.current.clientWidth,
      height: 300,
    })
    const s = c.addSeries(LineSeries, { color: '#00d4aa', lineWidth: 2, title: 'cumulative return %' })
    chart.current = c
    series.current = s
    const ro = new ResizeObserver(() => {
      if (chartRef.current) c.resize(chartRef.current.clientWidth, 300)
    })
    ro.observe(chartRef.current)
    return () => { ro.disconnect(); c.remove() }
  }, [])

  const renderEquity = (eq: EquityPoint[]) => {
    if (!series.current) return
    const data = eq.map((p, i) => ({ time: (i + 1) as any, value: p.cum_ret_pct }))
    series.current.setData(data)
    chart.current?.timeScale().fitContent()
  }

  const doRun = async () => {
    setRunning(true)
    setError(null)
    const t0 = performance.now()
    try {
      const res = await runBacktest({
        temperature, horizon, z_threshold: zThreshold,
        start_frac: startFrac, end_frac: endFrac, fee_bps: feeBps,
      })
      if (res.error) {
        setError(res.error)
        setResult(null)
      } else {
        setResult(res)
        renderEquity(res.equity)
      }
      setElapsedMs(Math.round(performance.now() - t0))
    } catch (e: any) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card">
        <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
          Backtest the predictor on the held-out test set
        </div>
        <div style={{ fontSize: 12, color: 'var(--fg-dim)', marginBottom: 14, lineHeight: 1.5 }}>
          For each test window, applies <strong>temperature</strong> to the model's logits, computes the
          standardized expected return over <strong>horizon</strong> bars, then takes a long if z &gt; threshold,
          short if z &lt; −threshold, otherwise flat. PnL is the actual cumulative log return over the
          horizon, minus 2 × fee on round-trip.
        </div>
        <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginBottom: 12, padding: '8px 12px', background: '#0e1620', border: '1px solid #1c2230', borderRadius: 4 }}>
          <strong style={{ color: 'var(--fg)' }}>Heads up:</strong> this model's z-scores are very small (it was trained for ~22k of 200k planned steps).
          Use <code style={{ fontFamily: 'var(--font-mono)', color: '#00d4aa' }}>z_threshold = 0</code> to take a position on every window, then dial up to filter for higher-conviction signals.
          A threshold of 0.05+ will give you 0 trades on this model.
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr) auto', gap: 12, alignItems: 'end' }}>
          <label style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
            Temperature
            <input type="number" step={0.1} min={0.1} value={temperature} onChange={e => setTemperature(parseFloat(e.target.value) || 1)} style={inputStyle} />
          </label>
          <label style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
            Horizon (bars)
            <input type="number" min={1} max={500} value={horizon} onChange={e => setHorizon(parseInt(e.target.value, 10) || 30)} style={inputStyle} />
          </label>
          <label style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
            z threshold
            <input type="number" step={0.01} min={0} value={zThreshold} onChange={e => setZThreshold(parseFloat(e.target.value) || 0)} style={inputStyle} />
          </label>
          <label style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
            Fee (bps)
            <input type="number" step={0.1} min={0} value={feeBps} onChange={e => setFeeBps(parseFloat(e.target.value) || 0)} style={inputStyle} />
          </label>
          <label style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
            Start frac
            <input type="number" step={0.05} min={0} max={1} value={startFrac} onChange={e => setStartFrac(parseFloat(e.target.value) || 0)} style={inputStyle} />
          </label>
          <label style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
            End frac
            <input type="number" step={0.05} min={0} max={1} value={endFrac} onChange={e => setEndFrac(parseFloat(e.target.value) || 1)} style={inputStyle} />
          </label>
          <button onClick={doRun} disabled={running} style={{ height: 32 }}>
            {running ? 'Running…' : 'Run backtest'}
          </button>
        </div>
        {error && <div style={{ color: 'var(--red)', fontSize: 12, marginTop: 10 }}>{error}</div>}
        {seed && (
          <div style={{ marginTop: 10, fontSize: 11, color: 'var(--fg-dim)' }}>
            Loaded T={seed.temperature}, H={seed.horizon} from Calibration sweep.
          </div>
        )}
      </div>

      {/* Stats row — always visible so the user can see what came back, even before they hit Run */}
      {result && (
        <div className="card">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12 }}>
            <Stat label="Trades" value={result.trades.toString()} />
            <Stat label="Longs / Shorts / Flats" value={`${result.longs} / ${result.shorts} / ${result.flats}`} />
            <Stat label="Win rate" value={`${(result.win_rate * 100).toFixed(1)}%`} color={result.win_rate > 0.5 ? '#00d4aa' : '#f5a623'} />
            <Stat label="Total return" value={`${result.total_return_pct >= 0 ? '+' : ''}${result.total_return_pct.toFixed(2)}%`} color={result.total_return_pct >= 0 ? '#00d4aa' : '#f05252'} />
            <Stat label="Sharpe / trade" value={result.sharpe_per_trade.toFixed(3)} />
            <Stat label="Max drawdown" value={`${result.max_drawdown_pct.toFixed(2)}%`} color="#f05252" />
          </div>
          <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginTop: 10 }}>
            Run on {result.n_windows} test windows · T={result.temperature}, H={result.horizon}, z≥{result.z_threshold}, fee={result.fee_bps} bps
            {elapsedMs !== null && ` · backtest took ${(elapsedMs / 1000).toFixed(2)}s`}
          </div>
        </div>
      )}

      <div className="card">
        <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
          Equity curve — cumulative return %
        </div>
        <div ref={chartRef} style={{ width: '100%' }} />
      </div>
    </div>
  )
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div style={{ fontSize: 10, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
      <div style={{ fontSize: 18, fontFamily: 'var(--font-mono)', color: color ?? 'var(--fg)', marginTop: 2 }}>{value}</div>
    </div>
  )
}
