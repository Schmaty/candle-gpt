import { useEffect, useRef, useState } from 'react'
import { createChart, type IChartApi, LineSeries } from 'lightweight-charts'
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

  const chartRef = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)

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
    </div>
  )
}
