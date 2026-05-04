import { useEffect, useRef, useState } from 'react'
import { createChart, type IChartApi, CandlestickSeries, LineSeries } from 'lightweight-charts'
import { fetchCandles } from '../api'

interface Candle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

interface Prediction {
  top5_rets: number[]
  top5_probs: number[]
  probs: number[]
}

export function LivePage() {
  const chartRef = useRef<HTMLDivElement>(null)
  const chart = useRef<IChartApi | null>(null)
  const candleSeries = useRef<any>(null)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [lastPrice, setLastPrice] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [limit, setLimit] = useState(300)

  useEffect(() => {
    if (!chartRef.current) return
    const c = createChart(chartRef.current, {
      layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
      grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
      rightPriceScale: { borderColor: '#252d3d' },
      timeScale: { borderColor: '#252d3d', timeVisible: true, secondsVisible: false },
      width: chartRef.current.clientWidth,
      height: 380,
    })
    const cs = c.addSeries(CandlestickSeries, {
      upColor: '#00d4aa', downColor: '#f05252',
      borderUpColor: '#00d4aa', borderDownColor: '#f05252',
      wickUpColor: '#00d4aa', wickDownColor: '#f05252',
    })
    chart.current = c
    candleSeries.current = cs

    const ro = new ResizeObserver(() => {
      if (chartRef.current) c.resize(chartRef.current.clientWidth, 380)
    })
    ro.observe(chartRef.current)
    return () => { ro.disconnect(); c.remove() }
  }, [])

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchCandles(limit)
      const candles: Candle[] = data.candles ?? []
      if (candles.length > 0) {
        const chartData = candles.map(c => ({
          time: c.time as any,
          open: c.open, high: c.high, low: c.low, close: c.close,
        }))
        candleSeries.current?.setData(chartData)
        chart.current?.timeScale().fitContent()
        setLastPrice(candles[candles.length - 1].close)
      }
      setPrediction(data.prediction ?? null)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [limit])

  useEffect(() => {
    const id = setInterval(load, 30_000)
    return () => clearInterval(id)
  }, [limit])

  const ret2pct = (r: number) => `${r >= 0 ? '+' : ''}${(r * 100).toFixed(3)}%`

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Toolbar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 18, color: 'var(--accent)' }}>
          BTC/USDT {lastPrice != null ? `$${lastPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : ''}
        </span>
        {[100, 200, 300, 500].map(l => (
          <button key={l} className={limit === l ? 'active' : ''} onClick={() => setLimit(l)}>{l}m</button>
        ))}
        <button onClick={load} disabled={loading}>{loading ? 'Loading…' : 'Refresh'}</button>
        {error && <span style={{ color: 'var(--red)', fontSize: 12 }}>{error}</span>}
      </div>

      {/* Chart */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div ref={chartRef} style={{ width: '100%' }} />
      </div>

      {/* Prediction */}
      {prediction ? (
        <div className="card">
          <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>
            Next-bar prediction — top 5 return bins
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {prediction.top5_rets.map((ret, i) => {
              const prob = prediction.top5_probs[i]
              const isPositive = ret > 0
              const barColor = isPositive ? 'var(--accent)' : ret < 0 ? 'var(--red)' : 'var(--fg-dim)'
              return (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: barColor, minWidth: 80 }}>
                    {ret2pct(ret)}
                  </span>
                  <div style={{ flex: 1, height: 16, background: 'var(--bg-elevated)', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${(prob * 100).toFixed(1)}%`, background: barColor, opacity: 0.8 }} />
                  </div>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--fg-dim)', minWidth: 50 }}>
                    {(prob * 100).toFixed(2)}%
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      ) : (
        <div className="card" style={{ color: 'var(--fg-dim)', fontSize: 13 }}>
          No model loaded — predictions unavailable. Train a model first.
        </div>
      )}
    </div>
  )
}
