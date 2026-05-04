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
      <TradeWheel p={p} ret2bps={ret2bps} fmtPrice={fmtPrice} />

      <div className="card">
        <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>
          Prediction detail
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

function TradeWheel({
  p,
  ret2bps,
  fmtPrice,
}: {
  p: Prediction
  ret2bps: (r: number) => string
  fmtPrice: (price: number) => string
}) {
  // Direction signal in [-1, +1]: positive = up, negative = down.
  const signal = p.p_up - p.p_down
  const conf = p.confidence

  // Decide what to recommend.
  // Low confidence overrides every direction signal — never tell the user
  // to act when the model is essentially guessing.
  const verb =
    conf < 0.10 ? { label: 'HOLD',          color: 'var(--fg-dim)', sub: 'Model unsure — wait for conviction' } :
    signal >= 0.30 ? { label: 'STRONG BUY', color: '#00d4aa', sub: 'Up bins dominate the distribution' } :
    signal >= 0.10 ? { label: 'BUY',         color: '#00d4aa', sub: 'Mild up bias' } :
    signal <= -0.30 ? { label: 'STRONG SELL', color: '#f05252', sub: 'Down bins dominate the distribution' } :
    signal <= -0.10 ? { label: 'SELL',        color: '#f05252', sub: 'Mild down bias' } :
                      { label: 'HOLD',         color: 'var(--fg-dim)', sub: 'Direction not committed' }

  // Confidence bucket label + color.
  const confBucket =
    conf < 0.05 ? { label: 'very low',  color: '#f05252' } :
    conf < 0.15 ? { label: 'low',       color: '#f5a623' } :
    conf < 0.30 ? { label: 'moderate',  color: '#f5a623' } :
    conf < 0.55 ? { label: 'high',      color: '#00d4aa' } :
                  { label: 'very high', color: '#00d4aa' }

  // Needle math. We're drawing a 180° arc from -90° (Sell, far left) to +90° (Buy, far right).
  // Clamp signal to [-1, 1] then map linearly to [-90°, +90°].
  const clampedSignal = Math.max(-1, Math.min(1, signal))
  const needleDeg = clampedSignal * 90  // -90 .. +90
  const needleRad = (needleDeg - 90) * Math.PI / 180  // SVG: 0° points right, so subtract 90 to put 0 at top.

  // SVG geometry — half-circle dial.
  const W = 320, H = 200
  const cx = W / 2, cy = H - 18
  const rOuter = 130, rTickOuter = 130, rTickInner = 116
  const rNeedleTip = 110, rNeedleBase = 12

  // Build the half-circle arc path (sweep left to right across the top).
  const arcStart = { x: cx - rOuter, y: cy }
  const arcEnd = { x: cx + rOuter, y: cy }
  const arcPath = `M ${arcStart.x} ${arcStart.y} A ${rOuter} ${rOuter} 0 0 1 ${arcEnd.x} ${arcEnd.y}`

  // Tick marks every 15° from -90° to +90° (13 ticks).
  const ticks: { x1: number, y1: number, x2: number, y2: number, color: string, w: number }[] = []
  for (let deg = -90; deg <= 90; deg += 15) {
    const rad = (deg - 90) * Math.PI / 180
    const x1 = cx + rTickInner * Math.cos(rad)
    const y1 = cy + rTickInner * Math.sin(rad)
    const x2 = cx + rTickOuter * Math.cos(rad)
    const y2 = cy + rTickOuter * Math.sin(rad)
    // Color ticks: outer thirds = red/green, middle third = gray.
    const color =
      deg <= -30 ? '#f05252' :
      deg >=  30 ? '#00d4aa' :
                   '#3a4458'
    const w = deg % 30 === 0 ? 2.5 : 1.2  // major every 30°.
    ticks.push({ x1, y1, x2, y2, color, w })
  }

  // Needle endpoints.
  const needleTipX = cx + rNeedleTip * Math.cos(needleRad)
  const needleTipY = cy + rNeedleTip * Math.sin(needleRad)
  const needlePerpRad = needleRad + Math.PI / 2
  const baseLeftX = cx + rNeedleBase * Math.cos(needlePerpRad)
  const baseLeftY = cy + rNeedleBase * Math.sin(needlePerpRad)
  const baseRightX = cx - rNeedleBase * Math.cos(needlePerpRad)
  const baseRightY = cy - rNeedleBase * Math.sin(needlePerpRad)

  return (
    <div className="card" style={{ padding: '20px 24px' }}>
      <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 12 }}>
        Trade analysis — next 1m bar
      </div>

      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <svg width={W} height={H} style={{ overflow: 'visible' }}>
          {/* Background arc */}
          <path d={arcPath} stroke="#1c2230" strokeWidth={20} fill="none" strokeLinecap="round" />

          {/* Three colored zone arcs */}
          {(() => {
            // Sell zone: -90° to -30°. Hold: -30° to +30°. Buy: +30° to +90°.
            const arc = (deg1: number, deg2: number, color: string) => {
              const r1 = (deg1 - 90) * Math.PI / 180
              const r2 = (deg2 - 90) * Math.PI / 180
              const x1 = cx + rOuter * Math.cos(r1), y1 = cy + rOuter * Math.sin(r1)
              const x2 = cx + rOuter * Math.cos(r2), y2 = cy + rOuter * Math.sin(r2)
              return (
                <path
                  key={color}
                  d={`M ${x1} ${y1} A ${rOuter} ${rOuter} 0 0 1 ${x2} ${y2}`}
                  stroke={color}
                  strokeWidth={6}
                  fill="none"
                  strokeLinecap="round"
                  opacity={0.55}
                />
              )
            }
            return [
              arc(-90, -30, '#f05252'),
              arc(-30,  30, '#3a4458'),
              arc( 30,  90, '#00d4aa'),
            ]
          })()}

          {/* Tick marks */}
          {ticks.map((t, i) => (
            <line key={i} x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2} stroke={t.color} strokeWidth={t.w} strokeLinecap="round" />
          ))}

          {/* Needle */}
          <g>
            <polygon
              points={`${baseLeftX},${baseLeftY} ${needleTipX},${needleTipY} ${baseRightX},${baseRightY}`}
              fill={verb.color}
              opacity={conf < 0.10 ? 0.5 : 0.95}
            />
            <circle cx={cx} cy={cy} r={10} fill="#0b0e13" stroke={verb.color} strokeWidth={2} />
          </g>

          {/* Side labels */}
          <text x={cx - rOuter + 4} y={cy + 14} fill="#f05252" fontSize={12} fontWeight={600} fontFamily="var(--font-mono)">SELL</text>
          <text x={cx + rOuter - 28} y={cy + 14} fill="#00d4aa" fontSize={12} fontWeight={600} fontFamily="var(--font-mono)">BUY</text>
        </svg>
      </div>

      {/* Verb */}
      <div style={{ textAlign: 'center', marginTop: 4 }}>
        <div style={{
          fontSize: 32,
          fontWeight: 800,
          color: verb.color,
          letterSpacing: '0.04em',
          fontFamily: 'var(--font-mono)',
        }}>
          {verb.label}
        </div>
        <div style={{ fontSize: 12, color: 'var(--fg-dim)', marginTop: 2 }}>{verb.sub}</div>
      </div>

      {/* Confidence bar */}
      <div style={{ marginTop: 18 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
          <span>Confidence</span>
          <span style={{ fontFamily: 'var(--font-mono)', color: confBucket.color }}>
            {(conf * 100).toFixed(1)}% — {confBucket.label}
          </span>
        </div>
        <div style={{ position: 'relative', height: 10, background: 'var(--bg-elevated)', borderRadius: 4, overflow: 'hidden' }}>
          {/* Tick marks for low / moderate / high thresholds (15%, 30%, 55%). */}
          <div style={{ position: 'absolute', left: '15%', top: 0, bottom: 0, width: 1, background: '#1c2230' }} />
          <div style={{ position: 'absolute', left: '30%', top: 0, bottom: 0, width: 1, background: '#1c2230' }} />
          <div style={{ position: 'absolute', left: '55%', top: 0, bottom: 0, width: 1, background: '#1c2230' }} />
          <div style={{
            position: 'absolute', left: 0, top: 0, bottom: 0,
            width: `${Math.max(2, conf * 100)}%`,
            background: confBucket.color,
            transition: 'width 200ms',
          }} />
        </div>
      </div>

      {/* Quick stats row */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 12,
        marginTop: 18,
        paddingTop: 14,
        borderTop: '1px solid #1c2230',
      }}>
        <Stat label="P(Up)"   value={`${(p.p_up * 100).toFixed(1)}%`}  color="#00d4aa" />
        <Stat label="P(Down)" value={`${(p.p_down * 100).toFixed(1)}%`} color="#f05252" />
        <Stat label="Expected" value={ret2bps(p.expected_ret)} color={p.expected_ret > p.flat_eps ? '#00d4aa' : p.expected_ret < -p.flat_eps ? '#f05252' : 'var(--fg-dim)'} />
        <Stat label="Target" value={fmtPrice(p.expected_close)} />
      </div>
    </div>
  )
}

function Stat({ label, value, color }: { label: string, value: string, color?: string }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 10, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</div>
      <div style={{ fontSize: 14, fontFamily: 'var(--font-mono)', color: color ?? 'var(--fg)', marginTop: 2 }}>{value}</div>
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
