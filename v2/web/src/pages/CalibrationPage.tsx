import { useEffect, useRef, useState } from 'react'
import { createChart, type IChartApi, HistogramSeries } from 'lightweight-charts'
import { fetchCalibration } from '../api'

interface Bucket {
  lo: number
  hi: number
  conf: number
  acc: number
  frac: number
}

export function CalibrationPage() {
  const [data, setData] = useState<{ buckets: Bucket[]; ece: number } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const confChartRef = useRef<HTMLDivElement>(null)
  const accChartRef = useRef<HTMLDivElement>(null)
  const confChart = useRef<IChartApi | null>(null)
  const accChart = useRef<IChartApi | null>(null)

  useEffect(() => {
    fetchCalibration()
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const chartOpts = (ref: React.RefObject<HTMLDivElement | null>) => ({
    layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
    grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
    rightPriceScale: { borderColor: '#252d3d' },
    timeScale: { borderColor: '#252d3d', timeVisible: false },
    width: ref.current?.clientWidth ?? 400,
    height: 180,
  })

  useEffect(() => {
    if (!confChartRef.current || !accChartRef.current || !data?.buckets?.length) return

    const cc = createChart(confChartRef.current, chartOpts(confChartRef))
    const ac = createChart(accChartRef.current, chartOpts(accChartRef))
    confChart.current = cc
    accChart.current = ac

    const confSeries = cc.addSeries(HistogramSeries, { color: '#00d4aa' })
    const accSeries = ac.addSeries(HistogramSeries, { color: '#f5a623' })

    // Use bucket index as time (lightweight-charts requires monotonically increasing time)
    const confData = data.buckets.map((b, i) => ({ time: i + 1 as any, value: b.conf }))
    const accData = data.buckets.map((b, i) => ({ time: i + 1 as any, value: b.acc }))

    confSeries.setData(confData)
    accSeries.setData(accData)
    cc.timeScale().fitContent()
    ac.timeScale().fitContent()

    const ro1 = new ResizeObserver(() => {
      if (confChartRef.current) cc.resize(confChartRef.current.clientWidth, 180)
    })
    const ro2 = new ResizeObserver(() => {
      if (accChartRef.current) ac.resize(accChartRef.current.clientWidth, 180)
    })
    if (confChartRef.current) ro1.observe(confChartRef.current)
    if (accChartRef.current) ro2.observe(accChartRef.current)

    return () => { ro1.disconnect(); ro2.disconnect(); cc.remove(); ac.remove() }
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

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="card">
          <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
            <span style={{ color: '#00d4aa' }}>● Avg Confidence</span> by bucket
          </div>
          <div ref={confChartRef} style={{ width: '100%' }} />
        </div>
        <div className="card">
          <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
            <span style={{ color: '#f5a623' }}>● Avg Accuracy</span> by bucket
          </div>
          <div ref={accChartRef} style={{ width: '100%' }} />
        </div>
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
    </div>
  )
}
