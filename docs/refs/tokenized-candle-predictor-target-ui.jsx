import React, { useMemo, useState, useRef, useEffect } from "react";

function generateCandles(count = 180) {
  let price = 100;
  const candles = [];

  for (let i = 0; i < count; i++) {
    const drift = Math.sin(i / 14) * 0.08;
    const noise = (Math.random() - 0.5) * 0.75;
    const open = price;
    const close = Math.max(1, open + drift + noise);
    const high = Math.max(open, close) + Math.random() * 0.55;
    const low = Math.min(open, close) - Math.random() * 0.55;
    const volume = 800 + Math.random() * 900 + Math.abs(close - open) * 1000;

    candles.push({ i, open, high, low, close, volume });
    price = close;
  }

  return candles;
}

function tokenize(candle) {
  const body = candle.close - candle.open;
  const range = Math.max(0.0001, candle.high - candle.low);
  const bodyRatio = Math.abs(body) / range;
  const upperWick = candle.high - Math.max(candle.open, candle.close);
  const lowerWick = Math.min(candle.open, candle.close) - candle.low;

  return {
    direction: body > 0.05 ? "UP" : body < -0.05 ? "DOWN" : "FLAT",
    body: bodyRatio > 0.65 ? "LARGE" : bodyRatio > 0.3 ? "MED" : "SMALL",
    upper: upperWick / range > 0.35 ? "LONG_UPPER" : "SHORT_UPPER",
    lower: lowerWick / range > 0.35 ? "LONG_LOWER" : "SHORT_LOWER",
    range: range > 1.0 ? "WIDE" : range > 0.45 ? "NORMAL" : "TIGHT",
  };
}

function detokenize(token, prevClose, step) {
  const directionMove = { UP: 0.42, DOWN: -0.42, FLAT: 0.02 }[token.direction];
  const bodyMult = { LARGE: 1.45, MED: 0.85, SMALL: 0.35 }[token.body];
  const rangeMult = { WIDE: 1.4, NORMAL: 0.8, TIGHT: 0.45 }[token.range];

  const open = prevClose;
  const wave = Math.sin(step / 3) * 0.09;
  const close = open + directionMove * bodyMult + wave;
  const upper = token.upper === "LONG_UPPER" ? 0.6 : 0.22;
  const lower = token.lower === "LONG_LOWER" ? 0.6 : 0.22;

  return {
    open,
    close,
    high: Math.max(open, close) + upper * rangeMult,
    low: Math.min(open, close) - lower * rangeMult,
    volume: 1000,
    predicted: true,
  };
}

function mutateToken(token, step) {
  const next = { ...token };
  if (step % 5 === 0) next.direction = next.direction === "UP" ? "DOWN" : "UP";
  if (step % 7 === 0) next.body = next.body === "LARGE" ? "SMALL" : "MED";
  if (step % 4 === 0) next.upper = next.upper === "LONG_UPPER" ? "SHORT_UPPER" : "LONG_UPPER";
  if (step % 6 === 0) next.lower = next.lower === "LONG_LOWER" ? "SHORT_LOWER" : "LONG_LOWER";
  return next;
}

function predictNext30(candles, anchorIndex) {
  const context = candles.slice(0, anchorIndex + 1);
  let prevCandle = context[context.length - 1];
  let token = tokenize(prevCandle);
  const predictions = [];

  for (let step = 1; step <= 30; step++) {
    token = mutateToken(token, step);
    const predicted = detokenize(token, prevCandle.close, step);
    predicted.i = anchorIndex + step;
    predicted.token = token;
    predictions.push(predicted);
    prevCandle = predicted;
  }

  return predictions;
}

function computeMetrics(real, predicted) {
  const pairs = predicted.map((p, idx) => [real[idx], p]).filter(([r]) => Boolean(r));
  if (!pairs.length) return null;

  const closeMae = pairs.reduce((s, [r, p]) => s + Math.abs(r.close - p.close), 0) / pairs.length;
  const directionHits = pairs.filter(
    ([r, p]) => Math.sign(r.close - r.open) === Math.sign(p.close - p.open)
  ).length;
  const directionAccuracy = directionHits / pairs.length;
  const avgPctError = pairs.reduce((s, [r, p]) => s + Math.abs(r.close - p.close) / r.close, 0) / pairs.length;

  return { count: pairs.length, closeMae, directionAccuracy, avgPctError };
}

export default function TokenizedCandlePredictorPreview() {
  const candles = useMemo(() => generateCandles(), []);
  const [hoveredIndex, setHoveredIndex] = useState(null);
  const [pinnedIndex, setPinnedIndex] = useState(null);
  const containerRef = useRef(null);

  const activeIndex = hoveredIndex ?? pinnedIndex;

  const predictions = useMemo(
    () => (activeIndex != null ? predictNext30(candles, activeIndex) : []),
    [candles, activeIndex]
  );

  const realFuture = activeIndex != null ? candles.slice(activeIndex + 1, activeIndex + 31) : [];
  const metrics = computeMetrics(realFuture, predictions);

  const visible = candles.slice(30, 160);
  const predictedByIndex = new Map(predictions.map((c) => [c.i, c]));

  const minPrice = Math.min(...visible.map((c) => c.low), ...predictions.map((c) => c.low));
  const maxPrice = Math.max(...visible.map((c) => c.high), ...predictions.map((c) => c.high));

  const chartHeight = 440;
  const candleWidth = 8;
  const gap = 3;
  const leftPad = 28;
  const rightPad = 28;
  const chartWidth = visible.length * (candleWidth + gap) + leftPad + rightPad;

  const y = (price) => {
    const padding = 32;
    return (
      padding +
      ((maxPrice - price) / Math.max(0.0001, maxPrice - minPrice)) * (chartHeight - padding * 2)
    );
  };

  // price axis ticks
  const ticks = useMemo(() => {
    const out = [];
    const steps = 6;
    for (let i = 0; i <= steps; i++) {
      const v = minPrice + ((maxPrice - minPrice) * i) / steps;
      out.push(v);
    }
    return out;
  }, [minPrice, maxPrice]);

  const activeCandle = activeIndex != null ? candles[activeIndex] : null;
  const activeToken = activeCandle ? tokenize(activeCandle) : null;

  // Forecast summary stats
  const forecastDelta = predictions.length
    ? predictions[predictions.length - 1].close - candles[activeIndex].close
    : 0;
  const forecastPct = activeCandle ? (forecastDelta / activeCandle.close) * 100 : 0;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&display=swap');

        .tcp-root {
          font-family: 'JetBrains Mono', ui-monospace, monospace;
          background:
            radial-gradient(ellipse 80% 50% at 50% -10%, rgba(56, 189, 248, 0.08), transparent),
            radial-gradient(ellipse 60% 40% at 100% 100%, rgba(244, 114, 182, 0.04), transparent),
            #0a0a0b;
          color: #e8e6e1;
          min-height: 100vh;
          letter-spacing: -0.01em;
        }
        .tcp-root::before {
          content: '';
          position: fixed; inset: 0;
          background-image:
            linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);
          background-size: 40px 40px;
          pointer-events: none;
          z-index: 0;
        }
        .tcp-display {
          font-family: 'Fraunces', serif;
          font-feature-settings: 'ss01', 'ss02';
          letter-spacing: -0.025em;
        }
        .tcp-panel {
          background: linear-gradient(180deg, rgba(24, 24, 27, 0.7) 0%, rgba(15, 15, 17, 0.7) 100%);
          backdrop-filter: blur(12px);
          border: 1px solid rgba(255, 255, 255, 0.06);
          box-shadow:
            0 1px 0 rgba(255, 255, 255, 0.04) inset,
            0 20px 60px -20px rgba(0, 0, 0, 0.6);
        }
        .tcp-chart-wrap {
          position: relative;
          overflow: hidden;
          border-radius: 14px;
        }
        .tcp-pred-candle {
          animation: tcp-draw 240ms cubic-bezier(0.2, 0.7, 0.3, 1) backwards;
        }
        @keyframes tcp-draw {
          from {
            opacity: 0;
            transform: translateY(6px) scaleY(0.6);
            transform-origin: center;
          }
          to {
            opacity: 1;
            transform: translateY(0) scaleY(1);
          }
        }
        .tcp-pred-line {
          stroke-dasharray: 100;
          stroke-dashoffset: 100;
          animation: tcp-stroke 320ms cubic-bezier(0.4, 0, 0.2, 1) forwards;
        }
        @keyframes tcp-stroke {
          to { stroke-dashoffset: 0; }
        }
        .tcp-cursor-line {
          stroke-dasharray: 2 4;
          opacity: 0.5;
        }
        .tcp-tick { font-size: 10px; fill: #6b6b6b; }
        .tcp-status-dot {
          width: 6px; height: 6px; border-radius: 50%;
          background: #4ade80;
          box-shadow: 0 0 8px #4ade80, 0 0 0 3px rgba(74, 222, 128, 0.15);
          animation: tcp-pulse 2s ease-in-out infinite;
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
        }
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
        .tcp-empty-hint {
          animation: tcp-float 3s ease-in-out infinite;
        }
        @keyframes tcp-float {
          0%, 100% { transform: translateY(0); opacity: 0.7; }
          50% { transform: translateY(-3px); opacity: 1; }
        }
        .tcp-overlay-bg {
          fill: url(#tcp-pred-gradient);
          opacity: 0.15;
        }
      `}</style>

      <div className="tcp-root" ref={containerRef}>
        <div className="relative max-w-[1400px] mx-auto px-8 py-10" style={{ zIndex: 1 }}>
          {/* Header */}
          <div className="flex items-end justify-between mb-8 flex-wrap gap-4">
            <div>
              <div className="flex items-center gap-3 mb-3">
                <span className="tcp-status-dot" />
                <span className="text-[10px] tracking-[0.2em] text-zinc-500 uppercase">Live preview · synthetic feed</span>
              </div>
              <h1 className="tcp-display text-5xl md:text-6xl font-semibold leading-none">
                Tokenized Candle <span className="italic text-sky-300/90">Predictor</span>
              </h1>
              <p className="text-sm text-zinc-500 mt-3 max-w-xl">
                Hover any candle to autoregressively project the next 30 minutes. The overlay redraws on every move.
              </p>
            </div>

            <div className="flex items-center gap-2 text-xs text-zinc-400">
              <span className="tcp-pill" style={{ color: '#7dd3fc' }}>1m · OHLCV</span>
              <span className="tcp-pill">N = {candles.length}</span>
              <span className="tcp-pill">Horizon 30</span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-5">
            {/* Chart */}
            <div className="tcp-panel rounded-2xl p-5">
              {/* Top bar */}
              <div className="flex items-center justify-between mb-4 px-1">
                <div className="flex items-center gap-6 text-xs">
                  <Stat label="Last" value={candles[candles.length - 1].close.toFixed(2)} accent />
                  <Stat label="High" value={Math.max(...candles.map(c => c.high)).toFixed(2)} />
                  <Stat label="Low" value={Math.min(...candles.map(c => c.low)).toFixed(2)} />
                  <Stat label="Bars" value={candles.length} />
                </div>

                <div className="flex items-center gap-3 text-[10px] tracking-[0.15em] text-zinc-500 uppercase">
                  <LegendDot color="#34d399" label="Up" />
                  <LegendDot color="#fb7185" label="Down" />
                  <LegendDot color="#7dd3fc" label="Predicted" dashed />
                </div>
              </div>

              <div className="tcp-chart-wrap" style={{ overflowX: 'auto' }}>
                <svg
                  width={chartWidth}
                  height={chartHeight}
                  className="select-none block"
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

                  {/* Grid lines */}
                  {ticks.map((t, idx) => (
                    <g key={idx}>
                      <line
                        x1={leftPad}
                        x2={chartWidth - rightPad + 16}
                        y1={y(t)}
                        y2={y(t)}
                        stroke="rgba(255,255,255,0.04)"
                        strokeWidth="1"
                      />
                      <text className="tcp-tick" x={chartWidth - rightPad + 4} y={y(t) + 3}>
                        {t.toFixed(1)}
                      </text>
                    </g>
                  ))}

                  {/* Cursor crosshair */}
                  {activeIndex != null && (() => {
                    const localIdx = visible.findIndex((c) => c.i === activeIndex);
                    if (localIdx < 0) return null;
                    const x = leftPad + localIdx * (candleWidth + gap) + candleWidth / 2;
                    return (
                      <line
                        x1={x} x2={x}
                        y1={16} y2={chartHeight - 16}
                        stroke="#fbbf24"
                        strokeWidth="1"
                        className="tcp-cursor-line"
                      />
                    );
                  })()}

                  {/* Candles */}
                  {visible.map((c, localIdx) => {
                    const x = leftPad + localIdx * (candleWidth + gap);
                    const actualIndex = c.i;
                    const up = c.close >= c.open;
                    const isActive = actualIndex === activeIndex;
                    const pred = predictedByIndex.get(actualIndex);
                    const isFaded = activeIndex != null && actualIndex > activeIndex && !pred;

                    return (
                      <g
                        key={actualIndex}
                        onMouseEnter={() => setHoveredIndex(actualIndex)}
                        onClick={() => setPinnedIndex(actualIndex)}
                        style={{ cursor: 'crosshair' }}
                      >
                        {/* Hit area */}
                        <rect
                          x={x - gap / 2}
                          y={0}
                          width={candleWidth + gap}
                          height={chartHeight}
                          fill="transparent"
                        />

                        {/* Real candle */}
                        <g opacity={isFaded ? 0.25 : 1} style={{ transition: 'opacity 200ms' }}>
                          <line
                            x1={x + candleWidth / 2}
                            x2={x + candleWidth / 2}
                            y1={y(c.high)}
                            y2={y(c.low)}
                            stroke={up ? '#4ade80' : '#fb7185'}
                            strokeWidth="1"
                          />
                          <rect
                            x={x}
                            y={Math.min(y(c.open), y(c.close))}
                            width={candleWidth}
                            height={Math.max(1.5, Math.abs(y(c.open) - y(c.close)))}
                            rx="0.5"
                            fill={up ? 'url(#tcp-up)' : 'url(#tcp-down)'}
                          />
                        </g>

                        {/* Anchor marker */}
                        {isActive && (
                          <>
                            <circle
                              cx={x + candleWidth / 2}
                              cy={y(c.close)}
                              r="5"
                              fill="none"
                              stroke="#fbbf24"
                              strokeWidth="1.5"
                            />
                            <circle
                              cx={x + candleWidth / 2}
                              cy={y(c.close)}
                              r="2"
                              fill="#fbbf24"
                            />
                          </>
                        )}

                        {/* Predicted overlay */}
                        {pred && (() => {
                          const stepIndex = pred.i - activeIndex;
                          const px = x + 1.5;
                          return (
                            <g
                              className="tcp-pred-candle"
                              style={{ animationDelay: `${stepIndex * 14}ms` }}
                            >
                              <line
                                x1={px + candleWidth / 2}
                                x2={px + candleWidth / 2}
                                y1={y(pred.high)}
                                y2={y(pred.low)}
                                stroke="#7dd3fc"
                                strokeWidth="1.5"
                                strokeDasharray="2 2"
                              />
                              <rect
                                x={px}
                                y={Math.min(y(pred.open), y(pred.close))}
                                width={candleWidth - 3}
                                height={Math.max(1.5, Math.abs(y(pred.open) - y(pred.close)))}
                                rx="0.5"
                                fill="rgba(125, 211, 252, 0.3)"
                                stroke="#7dd3fc"
                                strokeWidth="1"
                              />
                            </g>
                          );
                        })()}
                      </g>
                    );
                  })}

                  {/* Forecast trend line */}
                  {predictions.length > 0 && activeIndex != null && (() => {
                    const localAnchor = visible.findIndex((c) => c.i === activeIndex);
                    if (localAnchor < 0) return null;
                    const points = [
                      { x: leftPad + localAnchor * (candleWidth + gap) + candleWidth / 2, y: y(candles[activeIndex].close) },
                      ...predictions.map((p, idx) => {
                        const localIdx = visible.findIndex((c) => c.i === p.i);
                        if (localIdx < 0) return null;
                        return {
                          x: leftPad + localIdx * (candleWidth + gap) + (candleWidth - 3) / 2 + 1.5,
                          y: y(p.close),
                        };
                      }).filter(Boolean),
                    ];
                    const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
                    return (
                      <path
                        d={path}
                        fill="none"
                        stroke="#7dd3fc"
                        strokeWidth="1"
                        opacity="0.6"
                        className="tcp-pred-line"
                      />
                    );
                  })()}
                </svg>
              </div>

              {!activeIndex && (
                <div className="tcp-empty-hint text-center text-xs text-zinc-500 mt-3 tracking-[0.15em] uppercase">
                  ↑ Hover over a candle to project ↑
                </div>
              )}
              {activeIndex != null && (
                <div className="text-center text-xs text-zinc-600 mt-3 tracking-[0.15em] uppercase">
                  Anchor: bar #{activeIndex} · click to pin
                </div>
              )}
            </div>

            {/* Side panel */}
            <div className="space-y-4">
              {/* Forecast */}
              <div className="tcp-panel rounded-2xl p-5">
                <div className="flex items-center justify-between mb-4">
                  <span className="text-[10px] tracking-[0.2em] text-zinc-500 uppercase">Forecast</span>
                  <span className="tcp-pill" style={{ fontSize: 9 }}>+30 bars</span>
                </div>

                {activeCandle ? (
                  <>
                    <div className="tcp-display text-5xl font-semibold leading-none mb-1">
                      <span className={forecastDelta >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                        {forecastDelta >= 0 ? '+' : ''}{forecastPct.toFixed(2)}
                      </span>
                      <span className="text-zinc-600 text-2xl ml-1">%</span>
                    </div>
                    <div className="text-xs text-zinc-500 mb-4 font-normal">
                      {activeCandle.close.toFixed(2)} → {predictions[predictions.length - 1]?.close.toFixed(2)}
                    </div>

                    <div className="tcp-divider mb-4" />

                    <div className="grid grid-cols-2 gap-3">
                      <Metric
                        label="Direction"
                        value={metrics ? `${(metrics.directionAccuracy * 100).toFixed(0)}%` : '—'}
                        meter={metrics?.directionAccuracy}
                      />
                      <Metric
                        label="Close MAE"
                        value={metrics ? metrics.closeMae.toFixed(2) : '—'}
                      />
                      <Metric
                        label="% Error"
                        value={metrics ? `${(metrics.avgPctError * 100).toFixed(1)}%` : '—'}
                      />
                      <Metric
                        label="Coverage"
                        value={metrics ? `${metrics.count}/30` : '—'}
                      />
                    </div>
                  </>
                ) : (
                  <div className="text-zinc-600 text-sm py-6 text-center">
                    No anchor selected
                  </div>
                )}
              </div>

              {/* Token */}
              <div className="tcp-panel rounded-2xl p-5">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-[10px] tracking-[0.2em] text-zinc-500 uppercase">Token</span>
                  {activeIndex != null && (
                    <span className="text-[10px] text-zinc-600">#{activeIndex}</span>
                  )}
                </div>

                {activeToken ? (
                  <div className="space-y-1.5 text-xs">
                    <TokenRow k="direction" v={activeToken.direction} highlight={
                      activeToken.direction === 'UP' ? 'emerald' :
                      activeToken.direction === 'DOWN' ? 'rose' : null
                    } />
                    <TokenRow k="body" v={activeToken.body} />
                    <TokenRow k="range" v={activeToken.range} />
                    <TokenRow k="upper" v={activeToken.upper.replace('_', ' ').toLowerCase()} />
                    <TokenRow k="lower" v={activeToken.lower.replace('_', ' ').toLowerCase()} />
                  </div>
                ) : (
                  <div className="text-zinc-600 text-xs py-4 text-center">
                    —
                  </div>
                )}
              </div>

              {/* Random pin */}
              <button
                onClick={() => setPinnedIndex(Math.floor(35 + Math.random() * 95))}
                className="w-full rounded-2xl py-3 text-xs font-semibold tracking-[0.15em] uppercase transition-all"
                style={{
                  background: 'linear-gradient(180deg, #fafafa, #d4d4d8)',
                  color: '#0a0a0b',
                  boxShadow: '0 4px 14px -2px rgba(0,0,0,0.4), 0 1px 0 rgba(255,255,255,0.5) inset',
                }}
                onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-1px)'}
                onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
              >
                ↳ Pin random anchor
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function Stat({ label, value, accent }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className="text-[9px] tracking-[0.15em] text-zinc-600 uppercase">{label}</span>
      <span className={`tcp-display text-base font-semibold ${accent ? 'text-sky-300' : 'text-zinc-200'}`}>
        {value}
      </span>
    </div>
  );
}

function LegendDot({ color, label, dashed }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        style={{
          width: 10, height: 4, borderRadius: 1,
          background: dashed ? 'transparent' : color,
          border: dashed ? `1px dashed ${color}` : 'none',
        }}
      />
      {label}
    </span>
  );
}

function Metric({ label, value, meter }) {
  return (
    <div>
      <div className="text-[9px] tracking-[0.15em] text-zinc-600 uppercase mb-1">{label}</div>
      <div className="tcp-display text-xl font-semibold text-zinc-100 mb-1.5">{value}</div>
      {meter != null && (
        <div className="tcp-meter-bg">
          <div className="tcp-meter-fill" style={{ width: `${meter * 100}%` }} />
        </div>
      )}
    </div>
  );
}

function TokenRow({ k, v, highlight }) {
  const colorMap = {
    emerald: '#4ade80',
    rose: '#fb7185',
  };
  return (
    <div className="flex items-center justify-between py-1 border-b border-white/5 last:border-0">
      <span className="text-zinc-600">{k}</span>
      <span
        className="font-semibold"
        style={{ color: highlight ? colorMap[highlight] : '#e8e6e1' }}
      >
        {v}
      </span>
    </div>
  );
}
