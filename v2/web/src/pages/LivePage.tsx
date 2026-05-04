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
  bin_centers: number[]
  p_up: number
  p_down: number
  p_flat: number
  flat_eps: number
  expected_ret: number
  expected_close: number
  last_close: number
  entropy_bits: number
  max_entropy_bits: number
  confidence: number
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
  const ret2bps = (r: number) => `${r >= 0 ? '+' : ''}${(r * 10000).toFixed(2)} bps`
  const fmtPrice = (p: number) => `$${p.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`

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
        <PredictionCard p={prediction} ret2pct={ret2pct} ret2bps={ret2bps} fmtPrice={fmtPrice} />
      ) : (
        <div className="card" style={{ color: 'var(--fg-dim)', fontSize: 13 }}>
          No model loaded — predictions unavailable. Train a model first.
        </div>
      )}
    </div>
  )
}

function PredictionCard({
  p,
  ret2pct,
  ret2bps,
  fmtPrice,
}: {
  p: Prediction
  ret2pct: (r: number) => string
  ret2bps: (r: number) => string
  fmtPrice: (p: number) => string
}) {
  const direction =
    p.p_up > p.p_down && p.p_up > p.p_flat ? { label: 'UP',   color: 'var(--accent)' } :
    p.p_down > p.p_up && p.p_down > p.p_flat ? { label: 'DOWN', color: 'var(--red)' } :
                                                { label: 'FLAT', color: 'var(--fg-dim)' }

  const conf = p.confidence
  const confLabel =
    conf < 0.05 ? 'very low — model unsure' :
    conf < 0.15 ? 'low' :
    conf < 0.30 ? 'moderate' :
    conf < 0.55 ? 'high' :
                  'very high'

  const priceDelta = p.expected_close - p.last_close
  const priceDeltaSign = priceDelta >= 0 ? '+' : ''

  const pctRow = (label: string, prob: number, color: string) => (
    <div style={{ display: 'grid', gridTemplateColumns: '60px 1fr 70px', gap: 10, alignItems: 'center' }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color }}>{label}</span>
      <div style={{ height: 12, background: 'var(--bg-elevated)', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${(prob * 100).toFixed(1)}%`, background: color, opacity: 0.85 }} />
      </div>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--fg)', textAlign: 'right' }}>
        {(prob * 100).toFixed(1)}%
      </span>
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="card">
        <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>
          Next-bar prediction
        </div>

        {/* Headline */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 18 }}>
          <div>
            <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginBottom: 4 }}>Direction (1 min ahead)</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
              <span style={{ fontSize: 26, fontWeight: 700, color: direction.color, fontFamily: 'var(--font-mono)' }}>
                {direction.label}
              </span>
              <span style={{ fontSize: 13, color: 'var(--fg-dim)' }}>
                conf {(conf * 100).toFixed(1)}% — {confLabel}
              </span>
            </div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginBottom: 4 }}>Expected next close</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
              <span style={{ fontSize: 22, fontWeight: 600, color: 'var(--fg)', fontFamily: 'var(--font-mono)' }}>
                {fmtPrice(p.expected_close)}
              </span>
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 13,
                color: priceDelta > 0 ? 'var(--accent)' : priceDelta < 0 ? 'var(--red)' : 'var(--fg-dim)',
              }}>
                {priceDeltaSign}{priceDelta.toFixed(2)} ({ret2bps(p.expected_ret)})
              </span>
            </div>
            <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginTop: 2 }}>
              from {fmtPrice(p.last_close)}
            </div>
          </div>
        </div>

        {/* Direction breakdown */}
        <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Aggregated probability  <span style={{ textTransform: 'none', letterSpacing: 0 }}>· “flat” = |return| &lt; {(p.flat_eps * 100).toFixed(2)}%</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 18 }}>
          {pctRow('Up',   p.p_up,   'var(--accent)')}
          {pctRow('Flat', p.p_flat, 'var(--fg-dim)')}
          {pctRow('Down', p.p_down, 'var(--red)')}
        </div>

        {/* Histogram */}
        <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Full distribution — {p.probs.length} return bins (left = most negative, right = most positive)
        </div>
        <ProbHistogram probs={p.probs} centers={p.bin_centers} flatEps={p.flat_eps} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--fg-dim)', marginTop: 4 }}>
          <span>{ret2pct(p.bin_centers[0])}</span>
          <span>0</span>
          <span>{ret2pct(p.bin_centers[p.bin_centers.length - 1])}</span>
        </div>

        {/* Top-5 detail */}
        <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginTop: 18, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Top 5 most-likely bins
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {p.top5_rets.map((ret, i) => {
            const prob = p.top5_probs[i]
            const color = ret > p.flat_eps ? 'var(--accent)' : ret < -p.flat_eps ? 'var(--red)' : 'var(--fg-dim)'
            return (
              <div key={i} style={{ display: 'grid', gridTemplateColumns: '90px 50px 1fr 60px', gap: 10, alignItems: 'center' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color }}>
                  {ret2bps(ret)}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--fg-dim)' }}>
                  {ret2pct(ret)}
                </span>
                <div style={{ height: 10, background: 'var(--bg-elevated)', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${(prob * 100).toFixed(1)}%`, background: color, opacity: 0.8 }} />
                </div>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--fg)', textAlign: 'right' }}>
                  {(prob * 100).toFixed(2)}%
                </span>
              </div>
            )
          })}
        </div>

        {/* Footer caveats */}
        <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginTop: 16, lineHeight: 1.6 }}>
          Model entropy {p.entropy_bits.toFixed(2)} / {p.max_entropy_bits.toFixed(2)} bits.
          A freshly-trained model with confidence under 10% is essentially saying “I don’t know.”
          Expected close treats the bin distribution as a probability over log-returns.
        </div>
      </div>
    </div>
  )
}

function ProbHistogram({ probs, centers, flatEps }: { probs: number[], centers: number[], flatEps: number }) {
  const max = Math.max(...probs)
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 0, height: 60, background: 'var(--bg-elevated)', borderRadius: 4, padding: 4 }}>
      {probs.map((pr, i) => {
        const c = centers[i]
        const color = c > flatEps ? '#00d4aa' : c < -flatEps ? '#f05252' : '#8492a6'
        const h = max > 0 ? (pr / max) * 100 : 0
        return (
          <div
            key={i}
            title={`bin ${i}: return ${(c * 100).toFixed(3)}% · prob ${(pr * 100).toFixed(2)}%`}
            style={{
              flex: 1,
              height: `${h}%`,
              background: color,
              minHeight: pr > 0 ? 1 : 0,
              opacity: 0.85,
            }}
          />
        )
      })}
    </div>
  )
}
