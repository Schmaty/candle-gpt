import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { fetchCandles, fetchPredictAtAnchor } from '../api'

interface Candle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
  i?: number
}

interface PredictedPoint {
  time: number
  close: number
  ret_bps: number
  cumulative_ret_bps: number
  cumulative_std_bps: number
  cumulative_z: number
  cumulative_close: number
  horizon: number
}

interface Prediction {
  predicted_path: PredictedPoint[]
  horizon_bars: number
  horizon_cumulative_ret: number
  horizon_cumulative_close: number
  horizon_cumulative_std: number
  horizon_cumulative_z: number
  last_close: number
  expected_close: number
  expected_ret: number
  confidence: number
  entropy_bits: number
  max_entropy_bits: number
  p_up: number
  p_down: number
  p_flat: number
}

type Interval = '1m' | '5m' | '15m' | '1h' | '4h' | '1d'

// "What kind of bar is this?" — pure shape function. Mirrors the reference
// component but uses scale-invariant (% of close) thresholds so it stays
// useful across asset prices (BTC at $80k vs an altcoin at $0.50).
function tokenize(c: Candle) {
  const body = c.close - c.open
  const range = Math.max(c.close * 1e-9, c.high - c.low)
  const bodyRatio = Math.abs(body) / range
  const upperWick = c.high - Math.max(c.open, c.close)
  const lowerWick = Math.min(c.open, c.close) - c.low
  const rangePct = range / Math.max(1e-9, c.close)
  const flatEps = c.close * 5e-5  // 5 bps
  return {
    direction: body > flatEps ? 'UP' : body < -flatEps ? 'DOWN' : 'FLAT',
    body: bodyRatio > 0.65 ? 'LARGE' : bodyRatio > 0.3 ? 'MED' : 'SMALL',
    upper: upperWick / range > 0.35 ? 'LONG_UPPER' : 'SHORT_UPPER',
    lower: lowerWick / range > 0.35 ? 'LONG_LOWER' : 'SHORT_LOWER',
    range: rangePct > 0.005 ? 'WIDE' : rangePct > 0.001 ? 'NORMAL' : 'TIGHT',
  }
}

// Compare predicted closes vs actual closes for past anchors (where the
// real future is known). Direction match is computed bar-vs-bar (sign of
// (close - open)) — for predicted bars the synthetic open is the previous
// predicted close, so this measures cumulative direction agreement.
function computeMetrics(
  realFuture: Candle[],
  predictions: Candle[],
): { count: number; closeMae: number; directionAccuracy: number; avgPctError: number } | null {
  const pairs: [Candle, Candle][] = []
  predictions.forEach((p, idx) => {
    const r = realFuture[idx]
    if (r) pairs.push([r, p])
  })
  if (!pairs.length) return null
  const closeMae = pairs.reduce((s, [r, p]) => s + Math.abs(r.close - p.close), 0) / pairs.length
  const directionHits = pairs.filter(
    ([r, p]) => Math.sign(r.close - r.open) === Math.sign(p.close - p.open),
  ).length
  const avgPctError = pairs.reduce((s, [r, p]) => s + Math.abs(r.close - p.close) / r.close, 0) / pairs.length
  return {
    count: pairs.length,
    closeMae,
    directionAccuracy: directionHits / pairs.length,
    avgPctError,
  }
}

// Turn the backend's close-only path into pseudo-OHLC candles whose open is
// the previous bar's close. high/low collapse to the open/close range — the
// model doesn't predict wicks, so we don't fake them.
function pathToCandles(anchor: Candle, path: PredictedPoint[]): Candle[] {
  const out: Candle[] = []
  let prevClose = anchor.close
  path.forEach((p) => {
    const open = prevClose
    const close = p.close
    out.push({
      time: p.time,
      open,
      close,
      high: Math.max(open, close),
      low: Math.min(open, close),
      volume: 0,
    })
    prevClose = close
  })
  return out
}

const STYLE = `
.tcp-root {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  background:
    radial-gradient(ellipse 80% 50% at 50% -10%, rgba(56, 189, 248, 0.08), transparent),
    radial-gradient(ellipse 60% 40% at 100% 100%, rgba(244, 114, 182, 0.04), transparent),
    #0a0a0b;
  color: #e8e6e1;
  letter-spacing: -0.01em;
  position: relative;
  isolation: isolate;
  border-radius: 14px;
  padding: 28px 28px 32px;
  overflow: hidden;
}
.tcp-root::before {
  content: '';
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  z-index: -1;
}
.tcp-display {
  font-family: 'Fraunces', 'Georgia', serif;
  font-feature-settings: 'ss01', 'ss02';
  letter-spacing: -0.025em;
}
.tcp-panel {
  background: linear-gradient(180deg, rgba(24, 24, 27, 0.7) 0%, rgba(15, 15, 17, 0.7) 100%);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.06);
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.04) inset,
    0 20px 60px -20px rgba(0, 0, 0, 0.6);
  border-radius: 16px;
}
.tcp-pred-candle {
  animation: tcp-draw 240ms cubic-bezier(0.2, 0.7, 0.3, 1) backwards;
}
@keyframes tcp-draw {
  from { opacity: 0; transform: translateY(6px) scaleY(0.6); transform-origin: center; }
  to { opacity: 1; transform: translateY(0) scaleY(1); }
}
.tcp-pred-line {
  stroke-dasharray: 800;
  stroke-dashoffset: 800;
  animation: tcp-stroke 600ms cubic-bezier(0.4, 0, 0.2, 1) forwards;
}
@keyframes tcp-stroke { to { stroke-dashoffset: 0; } }
.tcp-cursor-line { stroke-dasharray: 2 4; opacity: 0.5; }
.tcp-tick { font-size: 10px; fill: #6b6b6b; }
.tcp-status-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: #4ade80;
  box-shadow: 0 0 8px #4ade80, 0 0 0 3px rgba(74, 222, 128, 0.15);
  animation: tcp-pulse 2s ease-in-out infinite;
  display: inline-block;
}
@keyframes tcp-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
.tcp-divider {
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
  height: 1px;
}
.tcp-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.02);
  color: #d4d4d8;
  cursor: default;
}
.tcp-pill.btn { cursor: pointer; transition: border-color 150ms, color 150ms; }
.tcp-pill.btn:hover { border-color: rgba(125, 211, 252, 0.5); color: #7dd3fc; }
.tcp-pill.btn.active { border-color: #7dd3fc; color: #7dd3fc; background: rgba(125, 211, 252, 0.06); }
.tcp-meter-bg {
  background: rgba(255,255,255,0.04);
  border-radius: 999px;
  overflow: hidden;
  height: 4px;
}
.tcp-meter-fill {
  height: 100%;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
  transition: width 400ms cubic-bezier(0.2, 0.7, 0.3, 1);
}
.tcp-empty-hint { animation: tcp-float 3s ease-in-out infinite; }
@keyframes tcp-float {
  0%, 100% { transform: translateY(0); opacity: 0.7; }
  50% { transform: translateY(-3px); opacity: 1; }
}
.tcp-pin-btn {
  width: 100%;
  border-radius: 16px;
  padding: 12px 14px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  cursor: pointer;
  transition: transform 150ms;
  background: linear-gradient(180deg, #fafafa, #d4d4d8);
  color: #0a0a0b;
  box-shadow: 0 4px 14px -2px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.5);
  border: none;
  font-family: inherit;
}
.tcp-pin-btn:hover { transform: translateY(-1px); }
.tcp-pin-btn:active { transform: translateY(0); }
.tcp-fonts {
  /* Triggers Google Font loading without visible ::before content. */
  position: absolute; visibility: hidden; pointer-events: none;
  font-family: 'JetBrains Mono', 'Fraunces', sans-serif;
}
`

// Inject the Google Fonts <link> once per page-load.
let _fontsInjected = false
function ensureFonts() {
  if (_fontsInjected || typeof document === 'undefined') return
  _fontsInjected = true
  const link = document.createElement('link')
  link.rel = 'stylesheet'
  link.href = 'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&display=swap'
  document.head.appendChild(link)
}

const HORIZON = 30
const FETCH_LIMIT = 300

export function LivePage() {
  const [candles, setCandles] = useState<Candle[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [interval, setIntervalTf] = useState<Interval>('1m')
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)
  const [pinnedIndex, setPinnedIndex] = useState<number | null>(null)
  const [predCache, setPredCache] = useState<Map<string, Prediction>>(new Map())
  const [pendingAnchor, setPendingAnchor] = useState<number | null>(null)
  const inFlightRef = useRef<Set<string>>(new Set())

  const activeIndex = hoveredIndex ?? pinnedIndex
  const cacheKey = (anchorTime: number) => `${interval}|${anchorTime}|${HORIZON}`

  useEffect(() => { ensureFonts() }, [])

  // Refresh candles whenever interval changes.
  useEffect(() => {
    let cancelled = false
    const reload = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await fetchCandles(FETCH_LIMIT, interval)
        if (cancelled) return
        const cs: Candle[] = (data.candles ?? []).map((c: Candle, i: number) => ({ ...c, i }))
        setCandles(cs)
      } catch (e: any) {
        if (!cancelled) setError(e.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    reload()
    // Auto-refresh: re-fetch candles every 30s. Cached predictions (keyed
    // on anchor_time) survive the refresh so a pinned forecast stays put.
    const id = window.setInterval(reload, 30_000)
    return () => { cancelled = true; window.clearInterval(id) }
  }, [interval])

  // When the active anchor changes, fetch its prediction (unless cached).
  useEffect(() => {
    if (activeIndex == null || candles.length === 0) return
    const c = candles[activeIndex]
    if (!c) return
    const key = cacheKey(c.time)
    if (predCache.has(key) || inFlightRef.current.has(key)) return
    inFlightRef.current.add(key)
    setPendingAnchor(activeIndex)
    fetchPredictAtAnchor({ anchorTime: c.time, interval, horizon: HORIZON })
      .then((res) => {
        setPredCache((m) => {
          const n = new Map(m)
          n.set(key, res.prediction as Prediction)
          return n
        })
      })
      .catch((e) => {
        // Cache the failure as null so we don't retry every hover.
        setPredCache((m) => {
          const n = new Map(m)
          n.set(key, null as any)
          return n
        })
        console.warn('predict failed:', e.message)
      })
      .finally(() => {
        inFlightRef.current.delete(key)
        setPendingAnchor((p) => (p === activeIndex ? null : p))
      })
  }, [activeIndex, candles, interval, predCache])

  const activeCandle = activeIndex != null ? candles[activeIndex] : null
  const activePrediction: Prediction | null = useMemo(() => {
    if (!activeCandle) return null
    const key = cacheKey(activeCandle.time)
    return predCache.get(key) ?? null
  }, [activeCandle, predCache, interval])

  const predictedCandles: Candle[] = useMemo(() => {
    if (!activeCandle || !activePrediction?.predicted_path) return []
    return pathToCandles(activeCandle, activePrediction.predicted_path)
  }, [activeCandle, activePrediction])

  // Real future bars after the anchor (only present for past anchors). For
  // each predicted bar, look up the matching real bar by index.
  const realFuture: Candle[] = useMemo(() => {
    if (activeIndex == null) return []
    return candles.slice(activeIndex + 1, activeIndex + 1 + predictedCandles.length)
  }, [activeIndex, candles, predictedCandles.length])

  const metrics = useMemo(() => computeMetrics(realFuture, predictedCandles), [realFuture, predictedCandles])
  const activeToken = activeCandle ? tokenize(activeCandle) : null

  // Forecast headline: % change from anchor close to last predicted close.
  const forecastDelta = predictedCandles.length && activeCandle ? predictedCandles[predictedCandles.length - 1].close - activeCandle.close : 0
  const forecastPct = activeCandle && activeCandle.close ? (forecastDelta / activeCandle.close) * 100 : 0

  // Visible window: last ~140 bars, with room for the predicted overlay.
  const VISIBLE_TAIL = 140
  const tailStart = Math.max(0, candles.length - VISIBLE_TAIL)
  const visible = useMemo(() => candles.slice(tailStart), [candles, tailStart])

  // ---- chart geometry --------------------------------------------------
  const chartHeight = 460
  const candleWidth = 7
  const gap = 3
  const leftPad = 28
  const rightPad = 56
  const predOffset = predictedCandles.length
    ? Math.max(0, (visible[visible.length - 1]?.i ?? 0) - (activeIndex ?? 0))
    : 0
  // Reserve space at the right for predicted bars that extend past the
  // last visible real bar, so they actually fit on screen.
  const predOverflow = predictedCandles.length
    ? Math.max(0, predictedCandles.length - predOffset)
    : 0
  const chartWidth = (visible.length + predOverflow) * (candleWidth + gap) + leftPad + rightPad

  const minPrice = useMemo(() => {
    let m = Infinity
    visible.forEach((c) => { if (c.low < m) m = c.low })
    predictedCandles.forEach((c) => { if (c.low < m) m = c.low })
    return m === Infinity ? 0 : m
  }, [visible, predictedCandles])
  const maxPrice = useMemo(() => {
    let m = -Infinity
    visible.forEach((c) => { if (c.high > m) m = c.high })
    predictedCandles.forEach((c) => { if (c.high > m) m = c.high })
    return m === -Infinity ? 1 : m
  }, [visible, predictedCandles])

  const y = (price: number) => {
    const padding = 28
    return padding + ((maxPrice - price) / Math.max(1e-9, maxPrice - minPrice)) * (chartHeight - padding * 2)
  }
  const ticks = useMemo(() => {
    const out: number[] = []
    const steps = 6
    for (let i = 0; i <= steps; i++) {
      out.push(minPrice + ((maxPrice - minPrice) * i) / steps)
    }
    return out
  }, [minPrice, maxPrice])

  // X position for a candle by its global index (into `candles`, which is
  // shared across visible + predicted). We map the global index onto the
  // visible sub-window. Predicted bars use the same mapping (they extend
  // past visible[last]).
  const xForGlobalIndex = (gi: number): number | null => {
    const localIdx = gi - tailStart
    if (localIdx < 0) return null
    return leftPad + localIdx * (candleWidth + gap)
  }

  const lastCandle = candles[candles.length - 1]

  // Pin a random anchor that has enough context (so the predict call
  // actually returns 200 from the backend).
  const pinRandom = () => {
    if (candles.length < 30) return
    const minIdx = Math.max(0, Math.floor(candles.length * 0.2))
    const maxIdx = Math.max(minIdx + 1, candles.length - 2)
    const pick = minIdx + Math.floor(Math.random() * (maxIdx - minIdx))
    setHoveredIndex(null)
    setPinnedIndex(pick)
  }

  // ---- render ----------------------------------------------------------
  const fmt = (n: number, d = 2) => n.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })

  return (
    <div>
      <style>{STYLE}</style>
      <span className="tcp-fonts">.</span>
      <div className="tcp-root">
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', marginBottom: 28, flexWrap: 'wrap', gap: 16 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
              <span className="tcp-status-dot" />
              <span style={{ fontSize: 10, letterSpacing: '0.2em', color: '#71717a', textTransform: 'uppercase' }}>
                Live BTC/USDT · model run via /predict
              </span>
            </div>
            <h1 className="tcp-display" style={{ fontSize: 56, fontWeight: 600, lineHeight: 1, margin: 0 }}>
              Tokenized Candle <span style={{ fontStyle: 'italic', color: 'rgba(125, 211, 252, 0.9)' }}>Predictor</span>
            </h1>
            <p style={{ fontSize: 14, color: '#71717a', marginTop: 12, maxWidth: 560 }}>
              Hover any candle to project the next {HORIZON} bars from that point. Click to pin. Prediction comes from the trained CandleGPTv2 model anchored at the chosen bar.
            </p>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, alignItems: 'flex-end' }}>
            <div style={{ display: 'flex', gap: 8 }}>
              <span className="tcp-pill" style={{ color: '#7dd3fc' }}>{interval} · OHLCV</span>
              <span className="tcp-pill">N = {candles.length}</span>
              <span className="tcp-pill">Horizon {HORIZON}</span>
            </div>
            <div style={{ display: 'flex', gap: 4 }}>
              {(['1m', '5m', '15m', '1h', '4h', '1d'] as const).map((tf) => (
                <button
                  key={tf}
                  className={`tcp-pill btn ${interval === tf ? 'active' : ''}`}
                  onClick={() => { setIntervalTf(tf); setHoveredIndex(null); setPinnedIndex(null) }}
                  style={{ fontFamily: 'inherit' }}
                >
                  {tf}
                </button>
              ))}
            </div>
            {error && <span style={{ color: '#fb7185', fontSize: 11 }}>{error}</span>}
            {loading && <span style={{ color: '#71717a', fontSize: 11 }}>loading…</span>}
          </div>
        </div>

        {/* Body */}
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 320px', gap: 20 }}>
          {/* Chart panel */}
          <div className="tcp-panel" style={{ padding: 20 }}>
            <ChartHeader
              candles={candles}
              lastCandle={lastCandle}
            />
            <div style={{ overflowX: 'auto', overflowY: 'hidden', borderRadius: 14, position: 'relative' }}>
              <svg
                width={chartWidth}
                height={chartHeight}
                style={{ display: 'block', userSelect: 'none' }}
                onMouseLeave={() => setHoveredIndex(null)}
              >
                <defs>
                  <linearGradient id="tcp-pred-gradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#7dd3fc" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="#7dd3fc" stopOpacity="0" />
                  </linearGradient>
                  <linearGradient id="tcp-up" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#4ade80" />
                    <stop offset="100%" stopColor="#22c55e" />
                  </linearGradient>
                  <linearGradient id="tcp-down" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#fb7185" />
                    <stop offset="100%" stopColor="#e11d48" />
                  </linearGradient>
                </defs>

                {ticks.map((t, idx) => (
                  <g key={idx}>
                    <line x1={leftPad} x2={chartWidth - rightPad + 16} y1={y(t)} y2={y(t)}
                          stroke="rgba(255,255,255,0.04)" strokeWidth={1} />
                    <text className="tcp-tick" x={chartWidth - rightPad + 4} y={y(t) + 3}>
                      {t.toLocaleString(undefined, { maximumFractionDigits: 1 })}
                    </text>
                  </g>
                ))}

                {/* Crosshair at the active anchor */}
                {activeIndex != null && (() => {
                  const x = xForGlobalIndex(activeIndex)
                  if (x == null) return null
                  return (
                    <line x1={x + candleWidth / 2} x2={x + candleWidth / 2}
                          y1={16} y2={chartHeight - 16}
                          stroke="#fbbf24" strokeWidth={1} className="tcp-cursor-line" />
                  )
                })()}

                {/* Real candles + hit areas */}
                {visible.map((c) => {
                  const x = xForGlobalIndex(c.i ?? 0)
                  if (x == null) return null
                  const up = c.close >= c.open
                  const isActive = c.i === activeIndex
                  const isFaded = activeIndex != null && (c.i ?? 0) > activeIndex
                  return (
                    <g
                      key={c.i}
                      onMouseEnter={() => setHoveredIndex(c.i ?? null)}
                      onClick={() => setPinnedIndex(c.i ?? null)}
                      style={{ cursor: 'crosshair' }}
                    >
                      <rect x={x - gap / 2} y={0} width={candleWidth + gap} height={chartHeight} fill="transparent" />
                      <g opacity={isFaded ? 0.25 : 1} style={{ transition: 'opacity 200ms' }}>
                        <line x1={x + candleWidth / 2} x2={x + candleWidth / 2}
                              y1={y(c.high)} y2={y(c.low)}
                              stroke={up ? '#4ade80' : '#fb7185'} strokeWidth={1} />
                        <rect x={x}
                              y={Math.min(y(c.open), y(c.close))}
                              width={candleWidth}
                              height={Math.max(1.5, Math.abs(y(c.open) - y(c.close)))}
                              rx={0.5}
                              fill={up ? 'url(#tcp-up)' : 'url(#tcp-down)'} />
                      </g>
                      {isActive && (
                        <>
                          <circle cx={x + candleWidth / 2} cy={y(c.close)} r={5} fill="none" stroke="#fbbf24" strokeWidth={1.5} />
                          <circle cx={x + candleWidth / 2} cy={y(c.close)} r={2} fill="#fbbf24" />
                        </>
                      )}
                    </g>
                  )
                })}

                {/* Predicted candles */}
                {predictedCandles.length > 0 && activeIndex != null && predictedCandles.map((p, stepIdx) => {
                  const globalIdx = activeIndex + 1 + stepIdx
                  const x = xForGlobalIndex(globalIdx)
                  if (x == null) return null
                  const px = x + 1.5
                  return (
                    <g key={`pred-${stepIdx}`} className="tcp-pred-candle" style={{ animationDelay: `${stepIdx * 14}ms` }}>
                      <line x1={px + (candleWidth - 3) / 2} x2={px + (candleWidth - 3) / 2}
                            y1={y(p.high)} y2={y(p.low)}
                            stroke="#7dd3fc" strokeWidth={1.5} strokeDasharray="2 2" />
                      <rect x={px}
                            y={Math.min(y(p.open), y(p.close))}
                            width={candleWidth - 3}
                            height={Math.max(1.5, Math.abs(y(p.open) - y(p.close)))}
                            rx={0.5}
                            fill="rgba(125, 211, 252, 0.3)"
                            stroke="#7dd3fc"
                            strokeWidth={1} />
                    </g>
                  )
                })}

                {/* Trend line through predicted closes */}
                {predictedCandles.length > 0 && activeIndex != null && (() => {
                  const anchorX = xForGlobalIndex(activeIndex)
                  if (anchorX == null) return null
                  const anchorY = y(candles[activeIndex].close)
                  const pts: Array<[number, number]> = [[anchorX + candleWidth / 2, anchorY]]
                  predictedCandles.forEach((p, idx) => {
                    const gx = xForGlobalIndex(activeIndex + 1 + idx)
                    if (gx == null) return
                    pts.push([gx + 1.5 + (candleWidth - 3) / 2, y(p.close)])
                  })
                  const d = pts.map((pt, i) => `${i === 0 ? 'M' : 'L'} ${pt[0]} ${pt[1]}`).join(' ')
                  return <path d={d} fill="none" stroke="#7dd3fc" strokeWidth={1} opacity={0.6} className="tcp-pred-line" />
                })()}
              </svg>
            </div>

            {activeIndex == null ? (
              <div className="tcp-empty-hint" style={{ textAlign: 'center', fontSize: 11, color: '#71717a', marginTop: 12, letterSpacing: '0.15em', textTransform: 'uppercase' }}>
                ↑ Hover over a candle to project ↑
              </div>
            ) : (
              <div style={{ textAlign: 'center', fontSize: 11, color: '#52525b', marginTop: 12, letterSpacing: '0.15em', textTransform: 'uppercase' }}>
                {pendingAnchor === activeIndex
                  ? 'fetching prediction…'
                  : pinnedIndex === activeIndex
                    ? `Pinned anchor: bar #${activeIndex} · ${activeCandle ? new Date(activeCandle.time * 1000).toISOString().slice(11, 16) + ' UTC' : ''}`
                    : `Anchor: bar #${activeIndex} · click to pin`}
              </div>
            )}
          </div>

          {/* Side panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Forecast */}
            <div className="tcp-panel" style={{ padding: 20 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                <span style={{ fontSize: 10, letterSpacing: '0.2em', color: '#71717a', textTransform: 'uppercase' }}>Forecast</span>
                <span className="tcp-pill" style={{ fontSize: 9 }}>+{HORIZON} bars</span>
              </div>
              {activeCandle && activePrediction ? (
                <>
                  <div className="tcp-display" style={{ fontSize: 52, fontWeight: 600, lineHeight: 1, marginBottom: 4 }}>
                    <span style={{ color: forecastDelta >= 0 ? '#34d399' : '#fb7185' }}>
                      {forecastDelta >= 0 ? '+' : ''}{forecastPct.toFixed(2)}
                    </span>
                    <span style={{ color: '#52525b', fontSize: 22, marginLeft: 4 }}>%</span>
                  </div>
                  <div style={{ fontSize: 12, color: '#71717a', marginBottom: 16 }}>
                    {fmt(activeCandle.close)} → {fmt(predictedCandles[predictedCandles.length - 1]?.close ?? activeCandle.close)}
                    <span style={{ marginLeft: 8, color: '#52525b' }}>
                      conf {(activePrediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="tcp-divider" style={{ marginBottom: 16 }} />
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                    <Metric label="Direction"
                            value={metrics ? `${(metrics.directionAccuracy * 100).toFixed(0)}%` : '—'}
                            meter={metrics?.directionAccuracy} />
                    <Metric label="Close MAE" value={metrics ? `$${fmt(metrics.closeMae)}` : '—'} />
                    <Metric label="% Error" value={metrics ? `${(metrics.avgPctError * 100).toFixed(2)}%` : '—'} />
                    <Metric label="Coverage" value={metrics ? `${metrics.count}/${HORIZON}` : '—'} />
                  </div>
                </>
              ) : activeCandle && pendingAnchor === activeIndex ? (
                <div style={{ color: '#52525b', fontSize: 13, padding: '24px 0', textAlign: 'center' }}>fetching…</div>
              ) : activeCandle ? (
                <div style={{ color: '#52525b', fontSize: 13, padding: '24px 0', textAlign: 'center' }}>
                  prediction unavailable for this anchor
                </div>
              ) : (
                <div style={{ color: '#52525b', fontSize: 13, padding: '24px 0', textAlign: 'center' }}>
                  No anchor selected
                </div>
              )}
            </div>

            {/* Token breakdown */}
            <div className="tcp-panel" style={{ padding: 20 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <span style={{ fontSize: 10, letterSpacing: '0.2em', color: '#71717a', textTransform: 'uppercase' }}>Token</span>
                {activeIndex != null && <span style={{ fontSize: 10, color: '#52525b' }}>#{activeIndex}</span>}
              </div>
              {activeToken ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12 }}>
                  <TokenRow k="direction" v={activeToken.direction}
                            highlight={activeToken.direction === 'UP' ? 'emerald' : activeToken.direction === 'DOWN' ? 'rose' : null} />
                  <TokenRow k="body" v={activeToken.body} />
                  <TokenRow k="range" v={activeToken.range} />
                  <TokenRow k="upper" v={activeToken.upper.replace('_', ' ').toLowerCase()} />
                  <TokenRow k="lower" v={activeToken.lower.replace('_', ' ').toLowerCase()} />
                </div>
              ) : (
                <div style={{ color: '#52525b', fontSize: 12, padding: '12px 0', textAlign: 'center' }}>—</div>
              )}
            </div>

            <button className="tcp-pin-btn" onClick={pinRandom}>↳ Pin random anchor</button>
            {pinnedIndex != null && (
              <button
                onClick={() => { setPinnedIndex(null); setHoveredIndex(null) }}
                style={{
                  background: 'transparent', border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 12, padding: '8px 12px', color: '#a1a1aa', fontSize: 11,
                  letterSpacing: '0.1em', textTransform: 'uppercase', cursor: 'pointer',
                  fontFamily: 'inherit',
                }}
              >
                clear pin
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function ChartHeader({ candles, lastCandle }: { candles: Candle[]; lastCandle: Candle | undefined }) {
  const high = useMemo(() => candles.reduce((m, c) => Math.max(m, c.high), -Infinity), [candles])
  const low = useMemo(() => candles.reduce((m, c) => Math.min(m, c.low), Infinity), [candles])
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, padding: '0 4px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
        <Stat label="Last" value={lastCandle ? lastCandle.close.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—'} accent />
        <Stat label="High" value={Number.isFinite(high) ? high.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—'} />
        <Stat label="Low" value={Number.isFinite(low) ? low.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—'} />
        <Stat label="Bars" value={String(candles.length)} />
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, fontSize: 10, letterSpacing: '0.15em', color: '#71717a', textTransform: 'uppercase' }}>
        <LegendDot color="#34d399" label="Up" />
        <LegendDot color="#fb7185" label="Down" />
        <LegendDot color="#7dd3fc" label="Predicted" dashed />
      </div>
    </div>
  )
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
      <span style={{ fontSize: 9, letterSpacing: '0.15em', color: '#52525b', textTransform: 'uppercase' }}>{label}</span>
      <span className="tcp-display" style={{ fontSize: 16, fontWeight: 600, color: accent ? '#7dd3fc' : '#e4e4e7' }}>{value}</span>
    </div>
  )
}

function LegendDot({ color, label, dashed }: { color: string; label: string; dashed?: boolean }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <span style={{
        width: 10, height: 4, borderRadius: 1,
        background: dashed ? 'transparent' : color,
        border: dashed ? `1px dashed ${color}` : 'none',
      }} />
      {label}
    </span>
  )
}

function Metric({ label, value, meter }: { label: string; value: string; meter?: number }) {
  return (
    <div>
      <div style={{ fontSize: 9, letterSpacing: '0.15em', color: '#52525b', textTransform: 'uppercase', marginBottom: 4 }}>{label}</div>
      <div className="tcp-display" style={{ fontSize: 20, fontWeight: 600, color: '#f4f4f5', marginBottom: 6 }}>{value}</div>
      {meter != null && (
        <div className="tcp-meter-bg">
          <div className="tcp-meter-fill" style={{ width: `${Math.max(0, Math.min(1, meter)) * 100}%` }} />
        </div>
      )}
    </div>
  )
}

function TokenRow({ k, v, highlight }: { k: string; v: string; highlight?: 'emerald' | 'rose' | null }) {
  const colorMap: Record<string, string> = { emerald: '#4ade80', rose: '#fb7185' }
  const style: CSSProperties = { color: highlight ? colorMap[highlight] : '#e8e6e1', fontWeight: 600 }
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
      <span style={{ color: '#71717a' }}>{k}</span>
      <span style={style}>{v}</span>
    </div>
  )
}
