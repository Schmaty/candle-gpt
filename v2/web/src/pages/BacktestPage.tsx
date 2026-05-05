import { useEffect, useRef, useState } from 'react'
import { createChart, type IChartApi, LineSeries } from 'lightweight-charts'
import { runBacktest, getEvalHistory } from '../api'

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
  strategy: string
  annualized_vol: number
  atm_premium_pct: number
  n_windows: number
  trades: number
  longs: number
  shorts: number
  straddles?: number
  flats: number
  win_rate: number
  total_return_pct: number
  total_log_pnl: number
  sharpe_per_trade: number
  max_drawdown_pct: number
  equity: EquityPoint[]
  error?: string
}

const STRATEGIES: { id: string; label: string; description: string }[] = [
  { id: 'spot',          label: 'Spot long/short', description: 'Long if z > threshold, short if z < −threshold. PnL = ±actual cumulative log return − fee.' },
  { id: 'long_call',     label: 'Long ATM calls (bullish only)', description: 'Buy an ATM call when z > threshold. PnL = max(0, ΔS) − premium.' },
  { id: 'long_put',      label: 'Long ATM puts (bearish only)',  description: 'Buy an ATM put when z < −threshold. PnL = max(0, −ΔS) − premium.' },
  { id: 'long_straddle', label: 'Long ATM straddle (volatility)', description: 'Buy a call + put when |z| > threshold. PnL = |ΔS| − 2×premium. Profits on big moves either way.' },
]

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
  const [feeBps, setFeeBps] = useState(0.0)
  const [startFrac, setStartFrac] = useState(0)
  const [endFrac, setEndFrac] = useState(1)
  const [strategy, setStrategy] = useState('spot')
  const [annualizedVol, setAnnualizedVol] = useState(0.6)
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
        strategy, annualized_vol: annualizedVol,
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
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr 1fr', gap: 12, alignItems: 'end', marginBottom: 12 }}>
          <label style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
            Strategy
            <select
              value={strategy}
              onChange={e => setStrategy(e.target.value)}
              style={{ ...inputStyle, padding: '6px 10px' }}
            >
              {STRATEGIES.map(s => (
                <option key={s.id} value={s.id}>{s.label}</option>
              ))}
            </select>
          </label>
          {strategy !== 'spot' && (
            <label style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
              Annualized vol (for premium)
              <input type="number" step={0.05} min={0.01} value={annualizedVol} onChange={e => setAnnualizedVol(parseFloat(e.target.value) || 0.6)} style={inputStyle} />
            </label>
          )}
          <label style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
            Temperature
            <input type="number" step={0.1} min={0.1} value={temperature} onChange={e => setTemperature(parseFloat(e.target.value) || 1)} style={inputStyle} />
          </label>
          <label style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
            Horizon (bars)
            <input type="number" min={1} max={500} value={horizon} onChange={e => setHorizon(parseInt(e.target.value, 10) || 30)} style={inputStyle} />
          </label>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr) auto', gap: 12, alignItems: 'end' }}>
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
        <div style={{ marginTop: 8, fontSize: 11, color: 'var(--fg-dim)' }}>
          {STRATEGIES.find(s => s.id === strategy)?.description}
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
            <Stat
              label={
                result.strategy === 'long_straddle'
                  ? 'Straddles / Flats'
                  : result.strategy === 'long_call'
                    ? 'Calls / Flats'
                    : result.strategy === 'long_put'
                      ? 'Puts / Flats'
                      : 'Longs / Shorts / Flats'
              }
              value={
                result.strategy === 'long_straddle'
                  ? `${result.straddles ?? 0} / ${result.flats}`
                  : result.strategy === 'long_call'
                    ? `${result.longs} / ${result.flats}`
                    : result.strategy === 'long_put'
                      ? `${result.shorts} / ${result.flats}`
                      : `${result.longs} / ${result.shorts} / ${result.flats}`
              }
            />
            <Stat label="Win rate" value={`${(result.win_rate * 100).toFixed(1)}%`} color={result.win_rate > 0.5 ? '#00d4aa' : '#f5a623'} />
            <Stat label="Total return" value={`${result.total_return_pct >= 0 ? '+' : ''}${result.total_return_pct.toFixed(2)}%`} color={result.total_return_pct >= 0 ? '#00d4aa' : '#f05252'} />
            <Stat label="Sharpe / trade" value={result.sharpe_per_trade.toFixed(3)} />
            <Stat label="Max drawdown" value={`${result.max_drawdown_pct.toFixed(2)}%`} color="#f05252" />
          </div>
          <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginTop: 10, lineHeight: 1.6 }}>
            Strategy: <span style={{ color: 'var(--fg)', fontFamily: 'var(--font-mono)' }}>{result.strategy}</span>
            {' · '}
            Run on {result.n_windows} windows · T={result.temperature}, H={result.horizon}, z≥{result.z_threshold}, fee={result.fee_bps} bps
            {result.strategy !== 'spot' && (
              <> · σ={result.annualized_vol.toFixed(2)} → ATM premium <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--fg)' }}>{result.atm_premium_pct.toFixed(3)}%</span></>
            )}
            {elapsedMs !== null && ` · ${(elapsedMs / 1000).toFixed(2)}s`}
          </div>
        </div>
      )}

      <div className="card">
        <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
          Equity curve — cumulative return %
        </div>
        <div ref={chartRef} style={{ width: '100%' }} />
      </div>

      <EvalHistorySection />
    </div>
  )
}

interface HistoryEntry {
  ts: number
  step: number
  best_T: number
  best_H: number
  best_dir_acc: number | null
  backtest: {
    trades: number
    win_rate: number
    total_return_pct: number
    sharpe_per_trade: number
    max_drawdown_pct: number
  }
}

function EvalHistorySection() {
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [runId, setRunId] = useState<string | null>(null)
  const [available, setAvailable] = useState(true)

  const accRef = useRef<HTMLDivElement>(null)
  const accChart = useRef<IChartApi | null>(null)
  const accSeries = useRef<any>(null)

  const winRef = useRef<HTMLDivElement>(null)
  const winChart = useRef<IChartApi | null>(null)
  const winSeries = useRef<any>(null)

  const retRef = useRef<HTMLDivElement>(null)
  const retChart = useRef<IChartApi | null>(null)
  const retSeries = useRef<any>(null)

  const sharpeRef = useRef<HTMLDivElement>(null)
  const sharpeChart = useRef<IChartApi | null>(null)
  const sharpeSeries = useRef<any>(null)

  // Build the four small charts once.
  useEffect(() => {
    const setup = (
      ref: React.RefObject<HTMLDivElement | null>,
      chartRef: React.MutableRefObject<IChartApi | null>,
      seriesRef: React.MutableRefObject<any>,
      color: string,
    ) => {
      if (!ref.current) return
      const c = createChart(ref.current, {
        layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
        grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
        rightPriceScale: { borderColor: '#252d3d' },
        timeScale: { borderColor: '#252d3d', timeVisible: false },
        width: ref.current.clientWidth,
        height: 160,
      })
      const s = c.addSeries(LineSeries, { color, lineWidth: 2, lastValueVisible: false, priceLineVisible: false })
      chartRef.current = c
      seriesRef.current = s
      const ro = new ResizeObserver(() => {
        if (ref.current) c.resize(ref.current.clientWidth, 160)
      })
      ro.observe(ref.current)
      return () => { ro.disconnect(); c.remove() }
    }
    const off1 = setup(accRef, accChart, accSeries, '#00d4aa')
    const off2 = setup(winRef, winChart, winSeries, '#4a90e2')
    const off3 = setup(retRef, retChart, retSeries, '#f5a623')
    const off4 = setup(sharpeRef, sharpeChart, sharpeSeries, '#9b59b6')
    return () => { off1?.(); off2?.(); off3?.(); off4?.() }
  }, [])

  // Poll history every 30s.
  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const res = await getEvalHistory()
        if (cancelled) return
        setAvailable(!!res.available)
        setRunId(res.run_id ?? null)
        const items: HistoryEntry[] = res.entries ?? []
        items.sort((a, b) => a.step - b.step)
        setEntries(items)
        const x = (e: HistoryEntry) => (e.step + 1) as any
        accSeries.current?.setData(items.filter(e => e.best_dir_acc !== null).map(e => ({ time: x(e), value: (e.best_dir_acc as number) * 100 })))
        winSeries.current?.setData(items.map(e => ({ time: x(e), value: e.backtest.win_rate * 100 })))
        retSeries.current?.setData(items.map(e => ({ time: x(e), value: e.backtest.total_return_pct })))
        sharpeSeries.current?.setData(items.map(e => ({ time: x(e), value: e.backtest.sharpe_per_trade })))
        accChart.current?.timeScale().fitContent()
        winChart.current?.timeScale().fitContent()
        retChart.current?.timeScale().fitContent()
        sharpeChart.current?.timeScale().fitContent()
      } catch { /* ignore */ }
    }
    load()
    const id = setInterval(load, 30_000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  const latest = entries[entries.length - 1]
  const first = entries[0]

  return (
    <div className="card">
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 8 }}>
        <span style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Progress over time — auto-evaluated each new training checkpoint
        </span>
        {runId && <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--fg-dim)' }}>{runId}</span>}
        {entries.length > 0 && (
          <span style={{ fontSize: 11, color: 'var(--fg-dim)', marginLeft: 'auto' }}>
            {entries.length} checkpoint{entries.length === 1 ? '' : 's'} evaluated · steps {first?.step.toLocaleString()} → {latest?.step.toLocaleString()}
          </span>
        )}
      </div>
      {!available && (
        <div style={{ fontSize: 12, color: 'var(--fg-dim)' }}>
          No eval history yet. Run <code style={{ fontFamily: 'var(--font-mono)' }}>v2.train.poll_eval</code> alongside training to populate.
        </div>
      )}
      {available && entries.length === 0 && (
        <div style={{ fontSize: 12, color: 'var(--fg-dim)' }}>
          Watching for the first checkpoint — typically lands ~30 min into a fresh run.
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        <MiniPanel label="Best directional accuracy %" color="#00d4aa" innerRef={accRef} latest={latest && latest.best_dir_acc !== null ? `${(latest.best_dir_acc * 100).toFixed(1)}%` : '—'} />
        <MiniPanel label="Backtest win rate %" color="#4a90e2" innerRef={winRef} latest={latest ? `${(latest.backtest.win_rate * 100).toFixed(1)}%` : '—'} />
        <MiniPanel label="Backtest total return %" color="#f5a623" innerRef={retRef} latest={latest ? `${latest.backtest.total_return_pct >= 0 ? '+' : ''}${latest.backtest.total_return_pct.toFixed(2)}%` : '—'} />
        <MiniPanel label="Backtest Sharpe / trade" color="#9b59b6" innerRef={sharpeRef} latest={latest ? latest.backtest.sharpe_per_trade.toFixed(3) : '—'} />
      </div>
    </div>
  )
}

function MiniPanel({ label, color, innerRef, latest }: { label: string; color: string; innerRef: React.RefObject<HTMLDivElement | null>; latest: string }) {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
          <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: color, marginRight: 6 }} />
          {label}
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color }}>{latest}</span>
      </div>
      <div ref={innerRef} style={{ width: '100%' }} />
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
