import { useEffect, useState } from 'react'
import { runBacktest, getEvalHistory, fetchEquity } from '../api'
import { Panel, SLabel, Divider, MBox, Pill } from '../components/dash'

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

const STRATEGIES: { id: string; label: string }[] = [
  { id: 'spot', label: 'spot' },
  { id: 'long_call', label: 'long_call' },
  { id: 'long_put', label: 'long_put' },
  { id: 'long_straddle', label: 'straddle' },
]

// ---------- inline equity SVG (ported from prototype) -------------------

function EqChart({ data, W = 620, H = 180 }: { data: EquityPoint[]; W?: number; H?: number }) {
  const [tip, setTip] = useState<{ i: number; x: number } | null>(null)
  if (!data.length) return null
  const values = data.map(d => d.cum_ret_pct / 100 + 1) // normalize to 1.0 base for shading
  const minV = Math.min(...values)
  const maxV = Math.max(...values)
  const rng = Math.max(maxV - minV, 1e-6)
  const n = data.length
  const px = (i: number) => (i / Math.max(1, n - 1)) * (W - 36) + 16
  const py = (v: number) => H - 26 - ((v - minV) / rng) * (H - 42)
  const line = values.map((v, i) => `${i === 0 ? 'M' : 'L'}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ')
  const area = `${line} L${px(n - 1)},${H - 26} L${px(0)},${H - 26} Z`
  return (
    <div style={{ position: 'relative' }}>
      <svg
        width="100%"
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        style={{ overflow: 'visible' }}
        onMouseMove={e => {
          const r = e.currentTarget.getBoundingClientRect()
          const rx = ((e.clientX - r.left) / r.width) * W
          const i = Math.round(((rx - 16) / (W - 36)) * (n - 1))
          if (i >= 0 && i < n) setTip({ i, x: rx })
        }}
        onMouseLeave={() => setTip(null)}
      >
        <defs>
          <linearGradient id="eqG" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity=".12" />
            <stop offset="100%" stopColor="#38bdf8" stopOpacity="0" />
          </linearGradient>
        </defs>
        {(() => {
          let pk = values[0]
          return values.map((v, i) => {
            if (v > pk) pk = v
            const ddH = ((pk - v) / rng) * (H - 42)
            if (ddH < 2) return null
            return (
              <rect
                key={i}
                x={px(i)}
                y={py(pk)}
                width={W / n + 0.5}
                height={ddH}
                fill="rgba(251,113,133,.07)"
              />
            )
          })
        })()}
        <line
          x1={16}
          x2={W - 20}
          y1={py(1)}
          y2={py(1)}
          stroke="rgba(255,255,255,.08)"
          strokeWidth="1"
          strokeDasharray="3 3"
        />
        <path d={area} fill="url(#eqG)" />
        <path d={line} fill="none" stroke="#38bdf8" strokeWidth="1.5" strokeLinejoin="round" />
        {[minV, 1, maxV].map((v, i) => (
          <text
            key={i}
            x={W - 18}
            y={py(v) + 3}
            fontSize="8"
            fill="#3f3f46"
            fontFamily="'JetBrains Mono',monospace"
            textAnchor="end"
          >
            {((v - 1) * 100 >= 0 ? '+' : '') + ((v - 1) * 100).toFixed(1)}%
          </text>
        ))}
        {tip && (
          <>
            <line
              x1={tip.x}
              x2={tip.x}
              y1={8}
              y2={H - 26}
              stroke="#fbbf24"
              strokeWidth="1"
              strokeDasharray="2 4"
              opacity=".4"
            />
            <circle cx={tip.x} cy={py(values[tip.i])} r="2.5" fill="#38bdf8" />
          </>
        )}
      </svg>
      {tip && (
        <div
          style={{
            position: 'absolute',
            top: 6,
            left: Math.min(tip.x + 8, W - 140),
            background: 'rgba(12,12,14,.97)',
            border: '1px solid rgba(255,255,255,.08)',
            borderRadius: 8,
            padding: '5px 9px',
            fontSize: 9,
            lineHeight: 1.7,
            fontFamily: "'JetBrains Mono',monospace",
            pointerEvents: 'none',
            whiteSpace: 'nowrap',
          }}
        >
          <div style={{ color: '#52525b' }}>trade {tip.i}</div>
          <div style={{ color: '#38bdf8' }}>cum return {data[tip.i].cum_ret_pct.toFixed(2)}%</div>
        </div>
      )}
    </div>
  )
}

// ---------- BacktestPage --------------------------------------------------

export function BacktestPage({ seed }: { seed: BacktestSeed | null }) {
  const [strat, setStrat] = useState('spot')
  const [temperature, setTemperature] = useState(1.0)
  const [horizon, setHorizon] = useState(30)
  const [zThreshold, setZThreshold] = useState(0.0)
  const [feeBps, setFeeBps] = useState(0.0)
  const [annualizedVol, setAnnualizedVol] = useState(0.6)
  const [startFrac, setStartFrac] = useState(0)
  const [endFrac, setEndFrac] = useState(1)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [elapsedMs, setElapsedMs] = useState<number | null>(null)

  useEffect(() => {
    if (seed) {
      setTemperature(seed.temperature)
      setHorizon(seed.horizon)
    }
  }, [seed])

  const doRun = async () => {
    setRunning(true)
    setError(null)
    const t0 = performance.now()
    try {
      const res = await runBacktest({
        temperature,
        horizon,
        z_threshold: zThreshold,
        start_frac: startFrac,
        end_frac: endFrac,
        fee_bps: feeBps,
        strategy: strat,
        annualized_vol: annualizedVol,
      })
      if (res.error) {
        setError(res.error)
        setResult(null)
      } else {
        setResult(res)
      }
      setElapsedMs(Math.round(performance.now() - t0))
    } catch (e: any) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }

  const fmt = (n: number, d = 2) => (n % 1 === 0 ? n.toString() : n.toFixed(d))

  return (
    <div className="cgpt-bt-layout">
      {/* Params */}
      <Panel style={{ padding: 16 }}>
        <SLabel>Parameters</SLabel>
        <div style={{ marginBottom: 12 }}>
          <div
            style={{
              fontSize: 8,
              letterSpacing: '.15em',
              textTransform: 'uppercase',
              color: '#52525b',
              marginBottom: 6,
            }}
          >
            Strategy
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {STRATEGIES.map(s => (
              <button
                key={s.id}
                onClick={() => setStrat(s.id)}
                style={{
                  textAlign: 'left',
                  padding: '5px 10px',
                  borderRadius: 7,
                  fontSize: 9,
                  fontWeight: 700,
                  letterSpacing: '.08em',
                  textTransform: 'uppercase',
                  border: strat === s.id ? '1px solid rgba(125,211,252,.25)' : '1px solid transparent',
                  background: strat === s.id ? 'rgba(125,211,252,.05)' : 'transparent',
                  color: strat === s.id ? '#7dd3fc' : '#52525b',
                  cursor: 'pointer',
                  transition: 'all 150ms',
                }}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
        <Divider />
        {([
          ['Temperature', temperature, setTemperature, 0.1, 3, 0.1],
          ['Horizon', horizon, setHorizon, 1, 200, 1],
          ['Z-Threshold', zThreshold, setZThreshold, 0, 3, 0.1],
          ['Fee bps', feeBps, setFeeBps, 0, 50, 0.5],
          ['Ann. Vol', annualizedVol, setAnnualizedVol, 0.1, 2, 0.05],
          ['Start frac', startFrac, setStartFrac, 0, 1, 0.05],
          ['End frac', endFrac, setEndFrac, 0, 1, 0.05],
        ] as Array<[string, number, (v: number) => void, number, number, number]>).map(
          ([lbl, val, set, mn, mx, st]) => (
            <div key={lbl} style={{ marginBottom: 11 }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: 8,
                  color: '#52525b',
                  letterSpacing: '.12em',
                  textTransform: 'uppercase',
                  marginBottom: 4,
                }}
              >
                <span>{lbl}</span>
                <span style={{ color: '#71717a', fontFamily: "'JetBrains Mono',monospace" }}>{fmt(val)}</span>
              </div>
              <input
                type="range"
                min={mn}
                max={mx}
                step={st}
                value={val}
                onChange={e => set(Number(e.target.value))}
              />
            </div>
          ),
        )}
        <button
          onClick={doRun}
          disabled={running}
          style={{
            width: '100%',
            padding: '9px 0',
            borderRadius: 10,
            marginTop: 4,
            fontSize: 9,
            fontWeight: 700,
            letterSpacing: '.15em',
            textTransform: 'uppercase',
            background: running ? 'rgba(56,189,248,.05)' : 'rgba(56,189,248,.09)',
            color: running ? '#52525b' : '#7dd3fc',
            border: '1px solid rgba(56,189,248,.18)',
            cursor: running ? 'not-allowed' : 'pointer',
            transition: 'all 200ms',
          }}
        >
          {running ? '— running —' : '→ run backtest'}
        </button>
        {seed && (
          <div style={{ marginTop: 8, fontSize: 8, color: '#52525b', letterSpacing: '.1em' }}>
            seed T={seed.temperature}, H={seed.horizon}
          </div>
        )}
        {error && (
          <div style={{ color: '#fb7185', fontSize: 9, marginTop: 8 }}>{error}</div>
        )}
        {elapsedMs !== null && !error && (
          <div style={{ marginTop: 8, fontSize: 8, color: '#3f3f46', letterSpacing: '.1em' }}>
            {(elapsedMs / 1000).toFixed(2)}s elapsed
          </div>
        )}
      </Panel>

      {/* Results */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {result ? (
          <>
            <Panel style={{ padding: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <SLabel>Results</SLabel>
                <Pill>
                  {result.n_windows.toLocaleString()} windows · T={result.temperature} · H={result.horizon}
                </Pill>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12, marginBottom: 14 }}>
                <MBox
                  label="Total Return"
                  value={(result.total_return_pct >= 0 ? '+' : '') + result.total_return_pct.toFixed(2) + '%'}
                  accent={result.total_return_pct >= 0 ? '#4ade80' : '#fb7185'}
                />
                <MBox
                  label="Win Rate"
                  value={(result.win_rate * 100).toFixed(1) + '%'}
                  meter={result.win_rate}
                />
                <MBox label="Sharpe / trade" value={result.sharpe_per_trade.toFixed(3)} />
                <MBox
                  label="Max DD"
                  value={'-' + result.max_drawdown_pct.toFixed(2) + '%'}
                  accent="#fb7185"
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
                <MBox label="Trades" value={result.trades.toLocaleString()} />
                <MBox label="Longs" value={result.longs.toLocaleString()} accent="#4ade80" />
                <MBox label="Shorts" value={result.shorts.toLocaleString()} accent="#fb7185" />
                <MBox label="Flats" value={result.flats.toLocaleString()} />
              </div>
              {result.strategy !== 'spot' && (
                <div style={{ fontSize: 9, color: '#52525b', marginTop: 10, letterSpacing: '.08em' }}>
                  σ={result.annualized_vol.toFixed(2)} → ATM premium {result.atm_premium_pct.toFixed(3)}% · fee {result.fee_bps} bps
                </div>
              )}
            </Panel>
            <Panel style={{ padding: 16 }}>
              <SLabel>Equity Curve</SLabel>
              <EqChart data={result.equity} W={620} H={180} />
            </Panel>
          </>
        ) : (
          <CheckpointEquityPanel />
        )}
        <EvalHistorySection />
      </div>
    </div>
  )
}

// ---------- Loaded-checkpoint equity (formerly the standalone Equity tab) ------

interface CkptEquityPoint { idx: number; cumret: number; position: number }

function CheckpointEquityPanel() {
  const [data, setData] = useState<{ equity: CkptEquityPoint[]; sharpe: number; max_dd: number } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchEquity()
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <Panel style={{ padding: 24 }}>
        <SLabel>Loaded Checkpoint · Eval Equity</SLabel>
        <div style={{ color: '#3f3f46', fontSize: 10 }}>Loading…</div>
      </Panel>
    )
  }

  if (error || !data?.equity?.length) {
    return (
      <Panel style={{ padding: 32, textAlign: 'center' }}>
        <div
          className="float-hint"
          style={{
            fontSize: 9, color: '#2a2a2e',
            letterSpacing: '.15em', textTransform: 'uppercase',
          }}
        >
          Configure parameters and run backtest
        </div>
        <div style={{ marginTop: 8, fontSize: 8, color: '#3f3f46', letterSpacing: '.1em' }}>
          {error
            ? 'No checkpoint equity available'
            : 'Or load a trained checkpoint to see its baseline eval equity here'}
        </div>
      </Panel>
    )
  }

  // Render checkpoint equity inline (mini version of the old EquityPage chart).
  const curve = data.equity.map((e, i) => ({ t: i, v: 1 + e.cumret }))
  const finalCumret = data.equity[data.equity.length - 1]?.cumret ?? 0
  const minV = Math.min(...curve.map(d => d.v))
  const maxV = Math.max(...curve.map(d => d.v))
  const rng = Math.max(maxV - minV, 1e-6)
  const n = curve.length
  const W = 720
  const H = 180
  const pL = 16, pR = 52, pT = 16, pB = 26
  const px = (i: number) => pL + (i / Math.max(1, n - 1)) * (W - pL - pR)
  const py = (v: number) => pT + (H - pT - pB) - ((v - minV) / rng) * (H - pT - pB)
  const line = curve.map((d, i) => `${i === 0 ? 'M' : 'L'}${px(i).toFixed(1)},${py(d.v).toFixed(1)}`).join(' ')

  let peak = 0, maxDD = 0
  for (const p of curve) {
    if (p.v > peak) peak = p.v
    const dd = peak > 0 ? (peak - p.v) / peak : 0
    if (dd > maxDD) maxDD = dd
  }
  const totalRet = finalCumret * 100
  const peakRet = (maxV - 1) * 100

  return (
    <Panel style={{ padding: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, gap: 12, flexWrap: 'wrap' }}>
        <SLabel style={{ marginBottom: 0 }}>Loaded Checkpoint · Eval Equity</SLabel>
        <Pill>baseline · z=0 spot</Pill>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,minmax(0,1fr))', gap: 12, marginBottom: 14 }}>
        <MBox
          label="Total Return"
          value={(totalRet >= 0 ? '+' : '') + totalRet.toFixed(2) + '%'}
          accent={totalRet >= 0 ? '#4ade80' : '#fb7185'}
        />
        <MBox label="Max DD" value={'-' + (data.max_dd * 100).toFixed(2) + '%'} accent="#fb7185" />
        <MBox label="Peak" value={`+${peakRet.toFixed(2)}%`} accent="#7dd3fc" />
        <MBox label="Sharpe" value={data.sharpe.toFixed(2)} accent={data.sharpe > 0 ? '#4ade80' : '#fb7185'} />
      </div>
      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }} preserveAspectRatio="none">
        <defs>
          <linearGradient id="ckptEqLG" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity=".16" />
            <stop offset="100%" stopColor="#38bdf8" stopOpacity="0" />
          </linearGradient>
        </defs>
        {(() => {
          let pk = curve[0].v
          return curve.map((d, i) => {
            if (d.v > pk) pk = d.v
            const ddH = ((pk - d.v) / rng) * (H - pT - pB)
            if (ddH < 2) return null
            return (
              <rect
                key={i}
                x={px(i)} y={py(pk)}
                width={(W - pL - pR) / Math.max(1, n - 1) + 0.5}
                height={ddH}
                fill="rgba(251,113,133,.07)"
              />
            )
          })
        })()}
        <line
          x1={pL} x2={W - pR} y1={py(1)} y2={py(1)}
          stroke="rgba(255,255,255,.08)" strokeWidth="1" strokeDasharray="3 3"
        />
        <path d={`${line} L${px(n - 1)},${py(minV)} L${px(0)},${py(minV)} Z`} fill="url(#ckptEqLG)" />
        <path d={line} fill="none" stroke="#38bdf8" strokeWidth="1.5" strokeLinejoin="round" />
        {[minV, 1, maxV].map((v, i) => (
          <text
            key={i}
            x={W - pR + 6}
            y={py(v) + 3}
            fontSize="8"
            fill="#3f3f46"
            fontFamily="'JetBrains Mono',monospace"
          >
            {((v - 1) * 100 >= 0 ? '+' : '') + ((v - 1) * 100).toFixed(1)}%
          </text>
        ))}
      </svg>
      <div style={{ marginTop: 8, fontSize: 8, color: '#3f3f46', letterSpacing: '.08em', lineHeight: 1.7 }}>
        Saved-eval baseline from the loaded inference checkpoint. Computed peak DD {(maxDD * 100).toFixed(2)}%. Run a backtest above with custom parameters to overlay results.
      </div>
    </Panel>
  )
}

// ---------- Eval history (real API) -------------------------------------

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

function MiniLine({
  data,
  color,
  W = 280,
  H = 70,
}: {
  data: number[]
  color: string
  W?: number
  H?: number
}) {
  if (!data.length) return <div style={{ height: H, color: '#3f3f46', fontSize: 9 }}>no data</div>
  const minV = Math.min(...data)
  const maxV = Math.max(...data)
  const rng = Math.max(maxV - minV, 1e-9)
  const n = data.length
  const px = (i: number) => 6 + (i / Math.max(1, n - 1)) * (W - 12)
  const py = (v: number) => 6 + (1 - (v - minV) / rng) * (H - 12)
  const d = data.map((v, i) => `${i === 0 ? 'M' : 'L'}${px(i).toFixed(1)},${py(v).toFixed(1)}`).join(' ')
  return (
    <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
      <path d={d} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  )
}

function EvalHistorySection() {
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [runId, setRunId] = useState<string | null>(null)
  const [available, setAvailable] = useState(true)

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
      } catch { /* ignore */ }
    }
    load()
    const id = setInterval(load, 30_000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  const latest = entries[entries.length - 1]

  return (
    <Panel style={{ padding: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <SLabel>Eval History · auto per checkpoint</SLabel>
        {runId && <Pill>{runId}</Pill>}
      </div>
      {!available && (
        <div style={{ fontSize: 9, color: '#3f3f46' }}>
          No eval history yet. Run <code style={{ color: '#7dd3fc' }}>v2.train.poll_eval</code> alongside training.
        </div>
      )}
      {available && entries.length === 0 && (
        <div style={{ fontSize: 9, color: '#3f3f46' }}>
          Watching for the first checkpoint…
        </div>
      )}
      {entries.length > 0 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14 }}>
          <div>
            <MBox
              label="Best Dir Acc"
              value={latest && latest.best_dir_acc != null ? `${(latest.best_dir_acc * 100).toFixed(1)}%` : '—'}
              accent="#4ade80"
            />
            <MiniLine
              data={entries
                .filter(e => e.best_dir_acc !== null)
                .map(e => (e.best_dir_acc as number) * 100)}
              color="#4ade80"
            />
          </div>
          <div>
            <MBox
              label="Win Rate"
              value={latest ? `${(latest.backtest.win_rate * 100).toFixed(1)}%` : '—'}
              accent="#7dd3fc"
            />
            <MiniLine data={entries.map(e => e.backtest.win_rate * 100)} color="#7dd3fc" />
          </div>
          <div>
            <MBox
              label="Total Return"
              value={latest ? `${latest.backtest.total_return_pct >= 0 ? '+' : ''}${latest.backtest.total_return_pct.toFixed(2)}%` : '—'}
              accent="#fbbf24"
            />
            <MiniLine data={entries.map(e => e.backtest.total_return_pct)} color="#fbbf24" />
          </div>
          <div>
            <MBox
              label="Sharpe / trade"
              value={latest ? latest.backtest.sharpe_per_trade.toFixed(3) : '—'}
              accent="#c084fc"
            />
            <MiniLine data={entries.map(e => e.backtest.sharpe_per_trade)} color="#c084fc" />
          </div>
        </div>
      )}
    </Panel>
  )
}
