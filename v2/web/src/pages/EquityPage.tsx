import { useEffect, useRef, useState } from 'react'
import { createChart, type IChartApi, LineSeries } from 'lightweight-charts'
import { fetchEquity } from '../api'

interface EquityPoint {
  idx: number
  cumret: number
  position: number
}

export function EquityPage() {
  const chartRef = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)
  const [data, setData] = useState<{ equity: EquityPoint[]; sharpe: number; max_dd: number } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchEquity()
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!chartRef.current || !data?.equity?.length) return

    const c = createChart(chartRef.current, {
      layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
      grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
      rightPriceScale: { borderColor: '#252d3d' },
      timeScale: { borderColor: '#252d3d', timeVisible: false },
      width: chartRef.current.clientWidth,
      height: 320,
    })
    chart.current = c

    const equitySeries = c.addSeries(LineSeries, { color: '#00d4aa', lineWidth: 2, title: 'equity' })
    const baselineSeries = c.addSeries(LineSeries, {
      color: '#252d3d', lineWidth: 1, lineStyle: 2 /* dashed */,
    })

    const equityData = data.equity.map((e, i) => ({ time: (i + 1) as any, value: e.cumret }))
    const baselineData = data.equity.map((_, i) => ({ time: (i + 1) as any, value: 0 }))

    equitySeries.setData(equityData)
    baselineSeries.setData(baselineData)
    c.timeScale().fitContent()

    const ro = new ResizeObserver(() => {
      if (chartRef.current) c.resize(chartRef.current.clientWidth, 320)
    })
    ro.observe(chartRef.current)

    return () => { ro.disconnect(); c.remove() }
  }, [data])

  if (loading) return <div style={{ color: 'var(--fg-dim)', padding: 24 }}>Loading equity curve…</div>
  if (error) return <div style={{ color: 'var(--red)', padding: 24 }}>{error}</div>

  if (!data?.equity?.length) {
    return (
      <div className="card" style={{ maxWidth: 480, marginTop: 32 }}>
        <div style={{ color: 'var(--fg-dim)', fontSize: 13 }}>
          No equity data. Run evaluation after training to generate predictions.
        </div>
      </div>
    )
  }

  const finalCumret = data.equity[data.equity.length - 1]?.cumret ?? 0
  const isPositive = finalCumret >= 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Metrics */}
      <div className="metric-row">
        <div className="metric">
          <div className="label">Annualized Sharpe</div>
          <div className="value" style={{ color: data.sharpe > 0 ? 'var(--green)' : 'var(--red)' }}>
            {data.sharpe.toFixed(2)}
          </div>
        </div>
        <div className="metric">
          <div className="label">Max Drawdown</div>
          <div className="value" style={{ color: 'var(--red)' }}>
            {(data.max_dd * 100).toFixed(2)}%
          </div>
        </div>
        <div className="metric">
          <div className="label">Total Return</div>
          <div className="value" style={{ color: isPositive ? 'var(--green)' : 'var(--red)' }}>
            {isPositive ? '+' : ''}{(finalCumret * 100).toFixed(2)}%
          </div>
        </div>
        <div className="metric">
          <div className="label">Trades</div>
          <div className="value">{(data.equity.length - 1).toLocaleString()}</div>
        </div>
      </div>

      {/* Chart */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '12px 16px 0', fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Simulated equity curve — <span style={{ color: '#00d4aa' }}>● cumulative log-return</span>
        </div>
        <div ref={chartRef} style={{ width: '100%' }} />
      </div>

      {/* Disclaimer */}
      <div style={{ fontSize: 11, color: 'var(--fg-dim)', lineHeight: 1.6 }}>
        Strategy: long if pred_ret {'>'} +0.02%, short if pred_ret {'<'} −0.02%, else flat.
        Simulated on test set predictions. Not financial advice.
      </div>
    </div>
  )
}
