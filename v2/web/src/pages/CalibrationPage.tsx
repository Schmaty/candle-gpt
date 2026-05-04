import { useEffect, useRef, useState } from 'react'
import { createChart, type IChartApi, LineSeries } from 'lightweight-charts'
import { fetchCalibration, runSweep } from '../api'
import type { BacktestSeed } from './BacktestPage'

interface Bucket {
  lo: number
  hi: number
  conf: number
  acc: number
  frac: number
}

interface SweepRow {
  temperature: number
  horizon: number
  dir_acc: number | null
  n_valid: number
  ece: number
  mean_conf: number
  top1_acc: number
}

export function CalibrationPage({ onUseInBacktest }: { onUseInBacktest?: (seed: BacktestSeed) => void }) {
  const [data, setData] = useState<{ buckets: Bucket[]; ece: number } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const chartRef = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)

  // Sweep controls
  const [tempsStr, setTempsStr] = useState('0.5,0.8,1.0,1.5,2.0')
  const [horizonsStr, setHorizonsStr] = useState('1,3,5,10,20,30')
  const [nSamples, setNSamples] = useState(150)
  const [sweepResults, setSweepResults] = useState<SweepRow[] | null>(null)
  const [sweepBest, setSweepBest] = useState<SweepRow | null>(null)
  const [sweepRunning, setSweepRunning] = useState(false)
  const [sweepError, setSweepError] = useState<string | null>(null)
  const [sweepElapsedMs, setSweepElapsedMs] = useState<number | null>(null)

  const doSweep = async () => {
    setSweepRunning(true)
    setSweepError(null)
    const t0 = performance.now()
    try {
      const T_list = tempsStr.split(',').map(s => parseFloat(s.trim())).filter(x => Number.isFinite(x) && x > 0)
      const H_list = horizonsStr.split(',').map(s => parseInt(s.trim(), 10)).filter(x => Number.isFinite(x) && x > 0)
      if (!T_list.length || !H_list.length) {
        throw new Error('Need at least one positive temperature and horizon.')
      }
      const res = await runSweep(T_list, H_list, nSamples)
      setSweepResults(res.results as SweepRow[])
      setSweepBest((res.best as SweepRow) ?? null)
      setSweepElapsedMs(Math.round(performance.now() - t0))
    } catch (e: any) {
      setSweepError(e.message)
    } finally {
      setSweepRunning(false)
    }
  }

  useEffect(() => {
    fetchCalibration()
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!chartRef.current || !data?.buckets?.length) return

    const c = createChart(chartRef.current, {
      layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
      grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
      rightPriceScale: { borderColor: '#252d3d' },
      timeScale: { borderColor: '#252d3d', timeVisible: false },
      width: chartRef.current.clientWidth,
      height: 280,
    })
    chart.current = c

    // Actual accuracy per confidence bucket (teal line)
    const actualSeries = c.addSeries(LineSeries, { color: '#00d4aa', lineWidth: 2, title: 'actual' })
    // Perfect calibration diagonal (red dashed)
    const idealSeries = c.addSeries(LineSeries, { color: '#f05252', lineWidth: 1, lineStyle: 2, title: 'ideal' })

    // Use bucket index as monotonically increasing time (lw-charts requires this)
    const actualData = data.buckets.map((b, i) => ({ time: i + 1 as any, value: b.acc }))
    const idealData = data.buckets.map((b, i) => ({ time: i + 1 as any, value: b.conf }))

    actualSeries.setData(actualData)
    idealSeries.setData(idealData)
    c.timeScale().fitContent()

    const ro = new ResizeObserver(() => {
      if (chartRef.current) c.resize(chartRef.current.clientWidth, 280)
    })
    ro.observe(chartRef.current)

    return () => { ro.disconnect(); c.remove() }
  }, [data])

  if (loading) return <div style={{ color: 'var(--fg-dim)', padding: 24 }}>Loading calibration…</div>
  if (error) return <div style={{ color: 'var(--red)', padding: 24 }}>{error}</div>

  if (!data?.buckets?.length) {
    return (
      <div className="card" style={{ maxWidth: 480, marginTop: 32 }}>
        <div style={{ color: 'var(--fg-dim)', fontSize: 13 }}>
          No calibration data available. Run evaluation after training.
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="metric">
        <div className="label">ECE (Expected Calibration Error)</div>
        <div className="value">{data.ece.toFixed(4)}</div>
      </div>

      <div className="card">
        <div style={{ fontSize: 12, color: 'var(--fg-dim)', marginBottom: 8 }}>
          <span style={{ color: '#00d4aa' }}>● Actual accuracy</span> per confidence bucket ·{' '}
          <span style={{ color: '#f05252' }}>— Perfect calibration</span>
        </div>
        <div ref={chartRef} style={{ width: '100%' }} />
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table>
          <thead>
            <tr>
              <th>Confidence range</th>
              <th>Avg conf</th>
              <th>Avg acc</th>
              <th>Gap</th>
              <th>Fraction</th>
            </tr>
          </thead>
          <tbody>
            {data.buckets.map((b, i) => {
              const gap = b.conf - b.acc
              return (
                <tr key={i}>
                  <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                    [{b.lo.toFixed(1)}, {b.hi.toFixed(1)})
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: '#00d4aa' }}>{b.conf.toFixed(3)}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: '#f5a623' }}>{b.acc.toFixed(3)}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: Math.abs(gap) > 0.1 ? 'var(--red)' : 'var(--fg-dim)', fontSize: 12 }}>
                    {gap >= 0 ? '+' : ''}{gap.toFixed(3)}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--fg-dim)', fontSize: 12 }}>
                    {(b.frac * 100).toFixed(1)}%
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Settings sweep */}
      <div className="card">
        <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
          Settings sweep — find the (temperature, horizon) that maximizes directional accuracy
        </div>
        <div style={{ fontSize: 12, color: 'var(--fg-dim)', marginBottom: 12, lineHeight: 1.5 }}>
          For each pair, samples N test windows, applies the temperature to the model's logits, and checks
          whether the predicted return's sign matches the actual cumulative return over <em>horizon</em> bars.
          One forward pass per window — temperatures rescale the same logits.
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr auto', gap: 12, alignItems: 'end', marginBottom: 12 }}>
          <label style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
            Temperatures (comma-separated)
            <input value={tempsStr} onChange={e => setTempsStr(e.target.value)} style={inputStyle} />
          </label>
          <label style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
            Horizons in bars (comma-separated)
            <input value={horizonsStr} onChange={e => setHorizonsStr(e.target.value)} style={inputStyle} />
          </label>
          <label style={{ fontSize: 11, color: 'var(--fg-dim)' }}>
            Samples per pair
            <input type="number" min={20} max={1000} value={nSamples} onChange={e => setNSamples(parseInt(e.target.value, 10) || 150)} style={inputStyle} />
          </label>
          <button onClick={doSweep} disabled={sweepRunning} style={{ height: 32 }}>
            {sweepRunning ? 'Running…' : 'Run sweep'}
          </button>
        </div>
        {sweepError && <div style={{ color: 'var(--red)', fontSize: 12, marginBottom: 8 }}>{sweepError}</div>}
        {sweepBest && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, padding: '10px 14px', background: '#0e1620', border: '1px solid #1c2230', borderRadius: 6, marginBottom: 12 }}>
            <span style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Best</span>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--fg)' }}>
              T = <span style={{ color: '#00d4aa' }}>{sweepBest.temperature}</span>
              {'  ·  '}
              H = <span style={{ color: '#00d4aa' }}>{sweepBest.horizon}</span>
              {'  ·  '}
              dir_acc = <span style={{ color: '#00d4aa' }}>{(sweepBest.dir_acc! * 100).toFixed(1)}%</span>
              {'  '}
              <span style={{ color: 'var(--fg-dim)', fontSize: 12 }}>(n={sweepBest.n_valid})</span>
            </span>
            {onUseInBacktest && (
              <button
                onClick={() => onUseInBacktest({ temperature: sweepBest.temperature, horizon: sweepBest.horizon })}
                style={{ marginLeft: 'auto' }}
              >
                Use in backtest →
              </button>
            )}
          </div>
        )}
        {sweepResults && sweepResults.length > 0 && (
          <SweepHeatmap rows={sweepResults} onPick={onUseInBacktest} />
        )}
        {sweepElapsedMs !== null && (
          <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginTop: 8 }}>
            Sweep took {(sweepElapsedMs / 1000).toFixed(2)}s.
          </div>
        )}
      </div>
    </div>
  )
}

const inputStyle: React.CSSProperties = {
  display: 'block', width: '100%', marginTop: 4,
  background: '#0b0e13', color: 'var(--fg)',
  border: '1px solid #1c2230', borderRadius: 4,
  padding: '6px 10px', fontFamily: 'var(--font-mono)', fontSize: 13,
  height: 32, boxSizing: 'border-box',
}

function SweepHeatmap({
  rows,
  onPick,
}: {
  rows: SweepRow[]
  onPick?: (seed: BacktestSeed) => void
}) {
  const temperatures = Array.from(new Set(rows.map(r => r.temperature))).sort((a, b) => a - b)
  const horizons = Array.from(new Set(rows.map(r => r.horizon))).sort((a, b) => a - b)
  const accs = rows.filter(r => r.dir_acc !== null).map(r => r.dir_acc as number)
  const minAcc = accs.length ? Math.min(...accs) : 0
  const maxAcc = accs.length ? Math.max(...accs) : 1
  const denom = Math.max(1e-6, maxAcc - minAcc)
  const cellColor = (acc: number | null) => {
    if (acc === null) return '#1c2230'
    const t = (acc - minAcc) / denom    // 0..1, 1 = best
    // 0.5 = neutral gray; map onto a red→teal gradient with 0.5 anchored to no-skill.
    const dist = Math.abs(acc - 0.5) * 2 // 0..1 from neutral
    const isUp = acc > 0.5
    const baseR = isUp ? 0  : 240
    const baseG = isUp ? 212 : 82
    const baseB = isUp ? 170 : 82
    const a = Math.min(0.9, 0.15 + dist * 0.75)
    return `rgba(${baseR}, ${baseG}, ${baseB}, ${a})`
  }
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'separate', borderSpacing: 0, width: 'auto' }}>
        <thead>
          <tr>
            <th style={{ fontSize: 11, color: 'var(--fg-dim)', padding: '6px 10px', textAlign: 'right' }}>T \\ H →</th>
            {horizons.map(h => (
              <th key={h} style={{ fontSize: 12, fontFamily: 'var(--font-mono)', padding: '6px 10px', color: 'var(--fg-dim)' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {temperatures.map(T => (
            <tr key={T}>
              <td style={{ fontSize: 12, fontFamily: 'var(--font-mono)', padding: '6px 10px', color: 'var(--fg-dim)', textAlign: 'right' }}>{T}</td>
              {horizons.map(H => {
                const r = rows.find(x => x.temperature === T && x.horizon === H)
                if (!r) return <td key={H} />
                const acc = r.dir_acc
                return (
                  <td
                    key={H}
                    title={acc !== null ? `T=${T}, H=${H}\ndir_acc = ${(acc * 100).toFixed(2)}%\nn = ${r.n_valid}\ntop1 = ${(r.top1_acc * 100).toFixed(2)}%\nmean conf = ${(r.mean_conf * 100).toFixed(2)}%` : 'no data'}
                    onClick={() => onPick && acc !== null && onPick({ temperature: T, horizon: H })}
                    style={{
                      padding: '8px 12px',
                      fontFamily: 'var(--font-mono)',
                      fontSize: 12,
                      background: cellColor(acc),
                      color: acc !== null && acc > 0.6 ? '#0b0e13' : 'var(--fg)',
                      cursor: onPick && acc !== null ? 'pointer' : 'default',
                      borderRadius: 3,
                      textAlign: 'center',
                      minWidth: 64,
                    }}
                  >
                    {acc !== null ? `${(acc * 100).toFixed(1)}%` : '—'}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginTop: 6 }}>
        Click any cell to send those settings to the Backtest tab. 50% = no directional skill; greener is better.
      </div>
    </div>
  )
}
