import { useEffect, useRef, useState, useCallback } from 'react'
import { createChart, type IChartApi, LineSeries } from 'lightweight-charts'
import { getTrainingStatus, getTrainingEvents } from '../api'
import { ProgressBar } from '../components/ProgressBar'
import { SpecPanel } from '../components/SpecPanel'

const STATE_COLORS: Record<string, string> = {
  starting: '#4a90e2', training: 'var(--accent)', evaluating: '#f5a623',
  checkpointing: '#9b59b6', done: 'var(--green)', failed: 'var(--red)',
}

function fmt_s(s: number | null | undefined): string {
  if (s == null) return '—'
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60)
  if (h > 0) return `${h}h ${m}m`
  return `${m}m ${sec}s`
}

export function TrainingPage() {
  const [status, setStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const lossRef = useRef<HTMLDivElement>(null)
  const lossChart = useRef<IChartApi | null>(null)
  const trainSeries = useRef<any>(null)
  const valSeries = useRef<any>(null)
  const throughputRef = useRef<HTMLDivElement>(null)
  const throughputChart = useRef<IChartApi | null>(null)
  const throughputSeries = useRef<any>(null)
  const cursorRef = useRef<number | null>(null)
  const [eventCount, setEventCount] = useState(0)

  // Poll status every 2s
  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const s = await getTrainingStatus()
        if (!cancelled) setStatus(s)
      } catch { /* ignore */ }
      if (!cancelled) setLoading(false)
    }
    poll()
    const id = setInterval(poll, 2000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  // Tick every 500ms so we can interpolate elapsed/ETA between server writes
  // (status.json is only rewritten every log_interval_steps ~ 46s).
  const [, setNowTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setNowTick(t => t + 1), 500)
    return () => clearInterval(id)
  }, [])

  // Poll events every 5s, update chart
  useEffect(() => {
    if (!lossRef.current) return
    const chart = createChart(lossRef.current, {
      layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
      grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
      rightPriceScale: { borderColor: '#252d3d' },
      timeScale: { borderColor: '#252d3d', timeVisible: false },
      width: lossRef.current.clientWidth,
      height: 220,
    })
    const ts = chart.addSeries(LineSeries, { color: '#00d4aa', lineWidth: 2, title: 'train' })
    const vs = chart.addSeries(LineSeries, { color: '#f5a623', lineWidth: 1, title: 'val' })
    lossChart.current = chart
    trainSeries.current = ts
    valSeries.current = vs

    const ro = new ResizeObserver(() => {
      if (lossRef.current) chart.resize(lossRef.current.clientWidth, 220)
    })
    ro.observe(lossRef.current)

    return () => { ro.disconnect(); chart.remove() }
  }, [])

  // Throughput sparkline chart
  useEffect(() => {
    if (!throughputRef.current) return
    const chart = createChart(throughputRef.current, {
      layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
      grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
      rightPriceScale: { borderColor: '#252d3d' },
      timeScale: { borderColor: '#252d3d', timeVisible: false },
      width: throughputRef.current.clientWidth,
      height: 100,
    })
    const ts = chart.addSeries(LineSeries, { color: '#9b59b6', lineWidth: 1 })
    throughputChart.current = chart
    throughputSeries.current = ts
    const ro = new ResizeObserver(() => {
      if (throughputRef.current) chart.resize(throughputRef.current.clientWidth, 100)
    })
    ro.observe(throughputRef.current)
    return () => { ro.disconnect(); chart.remove() }
  }, [])

  const trainPoints = useRef<{time: number, value: number}[]>([])
  const valPoints = useRef<{time: number, value: number}[]>([])
  const throughputPoints = useRef<{time: number, value: number}[]>([])
  const lrPoints = useRef<{time: number, value: number}[]>([])
  const gradNormPoints = useRef<{time: number, value: number}[]>([])
  const bestValPoints = useRef<{time: number, value: number}[]>([])
  const bestValSoFar = useRef<number | null>(null)
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null)
  const [historyTick, setHistoryTick] = useState(0)

  const fetchEvents = useCallback(async () => {
    try {
      const res = await getTrainingEvents(cursorRef.current)
      if (!res.events?.length) return
      let changed = false
      for (const ev of res.events) {
        if (ev.kind === 'step' && ev.step != null) {
          const t = ev.step + 1000000
          if (ev.loss != null) {
            trainPoints.current.push({ time: t, value: ev.loss })
          }
          if (ev.throughput_tok_per_s != null) {
            throughputPoints.current.push({ time: t, value: ev.throughput_tok_per_s })
          }
          if (ev.lr != null) {
            lrPoints.current.push({ time: t, value: ev.lr })
          }
          if (ev.grad_norm != null) {
            gradNormPoints.current.push({ time: t, value: ev.grad_norm })
          }
          changed = true
        }
        if (ev.kind === 'val' && ev.step != null && ev.val_loss != null) {
          const t = ev.step + 1000000
          valPoints.current.push({ time: t, value: ev.val_loss })
          if (bestValSoFar.current == null || ev.val_loss < bestValSoFar.current) {
            bestValSoFar.current = ev.val_loss
          }
          bestValPoints.current.push({ time: t, value: bestValSoFar.current as number })
          changed = true
        }
      }
      if (changed) {
        trainSeries.current?.setData([...trainPoints.current].sort((a, b) => a.time - b.time))
        valSeries.current?.setData([...valPoints.current].sort((a, b) => a.time - b.time))
        lossChart.current?.timeScale().fitContent()
        if (throughputPoints.current.length > 0) {
          throughputSeries.current?.setData([...throughputPoints.current].sort((a, b) => a.time - b.time))
          throughputChart.current?.timeScale().fitContent()
        }
        setEventCount(c => c + res.events.length)
        setHistoryTick(t => t + 1)
      }
      if (res.cursor) cursorRef.current = res.cursor
    } catch { /* ignore */ }
  }, [])

  // ESC closes the modal
  useEffect(() => {
    if (selectedMetric == null) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setSelectedMetric(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selectedMetric])

  const METRIC_DEFS: Record<string, { label: string, points: () => {time:number,value:number}[], color: string, format: (v: number) => string }> = {
    train_loss: { label: 'Train loss',   points: () => trainPoints.current,      color: '#00d4aa', format: v => v.toFixed(4) },
    val_loss:   { label: 'Val loss',     points: () => valPoints.current,        color: '#f5a623', format: v => v.toFixed(4) },
    best_val:   { label: 'Best val',     points: () => bestValPoints.current,    color: '#ffcc66', format: v => v.toFixed(4) },
    lr:         { label: 'LR',           points: () => lrPoints.current,         color: '#4a90e2', format: v => v.toExponential(2) },
    grad_norm:  { label: 'Grad norm',    points: () => gradNormPoints.current,   color: '#e74c3c', format: v => v.toFixed(3) },
    throughput: { label: 'Throughput',   points: () => throughputPoints.current, color: '#9b59b6', format: v => `${Math.round(v).toLocaleString()} tok/s` },
  }

  useEffect(() => {
    fetchEvents()
    const id = setInterval(fetchEvents, 5000)
    return () => clearInterval(id)
  }, [fetchEvents])

  if (loading) return <div style={{ color: 'var(--fg-dim)', padding: 24 }}>Loading training status…</div>

  if (!status?.available) {
    return (
      <div className="card" style={{ maxWidth: 480, marginTop: 32 }}>
        <div style={{ fontSize: 13, color: 'var(--fg-dim)', lineHeight: 1.8 }}>
          No training run found. Start one:<br />
          <code style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)', fontSize: 12 }}>
            uv run python -m v2.train.run --raw-dir v2/data/raw
          </code>
        </div>
      </div>
    )
  }

  const stateColor = STATE_COLORS[status.state] ?? 'var(--fg-dim)'

  // Interpolate elapsed/ETA between server writes. status.json carries
  // last_update_utc; we add (now - that timestamp) to elapsed_s and subtract
  // it from eta_s so the seconds tick locally even though the server only
  // rewrites every log_interval_steps. Only do this while training is active.
  const liveState = status.state === 'training' || status.state === 'evaluating' || status.state === 'checkpointing'
  let displayElapsed = status.elapsed_s ?? 0
  let displayEta = status.eta_s ?? 0
  if (liveState && status.last_update_utc) {
    const lastUpdateMs = Date.parse(status.last_update_utc)
    if (!Number.isNaN(lastUpdateMs)) {
      const drift_s = Math.max(0, (Date.now() - lastUpdateMs) / 1000)
      displayElapsed = (status.elapsed_s ?? 0) + drift_s
      displayEta = Math.max(0, (status.eta_s ?? 0) - drift_s)
    }
  }
  const cap = status.wall_clock_cap_s ?? 1
  const stepProgress = (status.step ?? 0) / Math.max(1, status.max_steps ?? 1)
  const timeProgress = displayElapsed / cap
  const progress = Math.min(1, Math.max(stepProgress, timeProgress))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header strip */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--fg-dim)' }}>{status.run_id}</span>
        <span style={{ background: stateColor, color: '#0b0e13', borderRadius: 4, padding: '2px 8px', fontSize: 12, fontWeight: 600 }}>
          {status.state}
        </span>
        <span style={{ color: 'var(--fg-dim)', fontSize: 13 }}>elapsed {fmt_s(displayElapsed)}</span>
        <span style={{ color: 'var(--fg-dim)', fontSize: 13 }}>ETA {fmt_s(displayEta)}</span>
        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)', fontSize: 13 }}>
          {((progress) * 100).toFixed(1)}%
        </span>
      </div>

      {/* Progress bar */}
      <ProgressBar
        frac={progress}
        label={`step ${status.step?.toLocaleString() ?? 0} / ${status.max_steps?.toLocaleString() ?? '—'} · best_val_loss = ${status.best_val_loss != null ? status.best_val_loss.toFixed(4) : '—'} · last ckpt @ step ${status.last_checkpoint_step ?? '—'}`}
      />

      {/* Metrics + hardware */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="card">
          <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 10 }}>Live Metrics</div>
          <div className="metric-row" style={{ marginBottom: 0 }}>
            {([
              ['train_loss', 'Train loss', status.train_loss != null ? status.train_loss.toFixed(4) : '—'],
              ['val_loss',   'Val loss',   status.val_loss   != null ? status.val_loss.toFixed(4)   : '—'],
              ['best_val',   'Best val',   status.best_val_loss != null ? status.best_val_loss.toFixed(4) : '—'],
              ['lr',         'LR',         status.lr != null ? status.lr.toExponential(2) : '—'],
              ['grad_norm',  'Grad norm',  status.grad_norm != null ? status.grad_norm.toFixed(3) : '—'],
              ['throughput', 'Throughput', status.throughput_tok_per_s != null ? `${Math.round(status.throughput_tok_per_s).toLocaleString()} tok/s` : '—'],
            ] as Array<[string, string, string]>).map(([key, l, v]) => {
              const n = METRIC_DEFS[key]?.points().length ?? 0
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => setSelectedMetric(key)}
                  className="metric"
                  title={n > 0 ? `Click for ${n}-point history` : 'No history yet'}
                  style={{
                    minWidth: 110,
                    background: 'transparent',
                    border: '1px solid transparent',
                    borderRadius: 6,
                    padding: '6px 10px',
                    margin: 0,
                    textAlign: 'left',
                    cursor: n > 0 ? 'pointer' : 'default',
                    color: 'inherit',
                    fontFamily: 'inherit',
                    transition: 'border-color 120ms, background 120ms',
                  }}
                  onMouseEnter={e => {
                    if (n > 0) {
                      ;(e.currentTarget as HTMLElement).style.borderColor = METRIC_DEFS[key].color
                      ;(e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.02)'
                    }
                  }}
                  onMouseLeave={e => {
                    ;(e.currentTarget as HTMLElement).style.borderColor = 'transparent'
                    ;(e.currentTarget as HTMLElement).style.background = 'transparent'
                  }}
                >
                  <div className="label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: METRIC_DEFS[key].color, display: 'inline-block' }} />
                    {l}
                  </div>
                  <div className="value" style={{ fontSize: 15 }}>{v}</div>
                  <div style={{ fontSize: 10, color: 'var(--fg-dim)', marginTop: 2, fontFamily: 'var(--font-mono)' }}>
                    {n > 0 ? `${n} pts ›` : '—'}
                  </div>
                </button>
              )
            })}
          </div>
        </div>

        {status.hardware && (
          <SpecPanel title="Hardware" items={[
            { label: 'Device', value: `${status.hardware.device} — ${status.hardware.device_name}` },
            { label: 'Host', value: status.hardware.hostname },
            { label: 'CPUs', value: status.hardware.cpu_count },
            { label: 'RAM', value: `${status.hardware.ram_gb} GB` },
            { label: 'PyTorch', value: status.hardware.torch },
            { label: 'Python', value: status.hardware.python },
          ]} />
        )}
      </div>

      {/* Loss curves */}
      <div className="card">
        <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
          Loss Curves — {eventCount} events loaded · <span style={{ color: '#00d4aa' }}>● train</span> <span style={{ color: '#f5a623' }}>● val</span>
        </div>
        <div ref={lossRef} style={{ width: '100%' }} />
      </div>

      {/* Throughput sparkline */}
      <div className="card">
        <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
          <span style={{ color: '#9b59b6' }}>● Throughput</span> tok/s over time
        </div>
        <div ref={throughputRef} style={{ width: '100%' }} />
      </div>

      {/* Model spec */}
      {status.model && (
        <SpecPanel title="Model" items={[
          { label: 'Params', value: `${(status.model.n_params / 1e6).toFixed(1)} M` },
          { label: 'Layers', value: status.model.n_layers },
          { label: 'Heads', value: status.model.n_heads },
          { label: 'd_model', value: status.model.d_model },
          { label: 'n_bins', value: status.model.n_bins },
          { label: 'Window', value: `${status.model.window} bars` },
        ]} />
      )}

      {selectedMetric && METRIC_DEFS[selectedMetric] && (
        <MetricDetailModal
          def={METRIC_DEFS[selectedMetric]}
          historyTick={historyTick}
          onClose={() => setSelectedMetric(null)}
        />
      )}
    </div>
  )
}

function MetricDetailModal({
  def,
  historyTick,
  onClose,
}: {
  def: { label: string, points: () => {time:number,value:number}[], color: string, format: (v: number) => string }
  historyTick: number
  onClose: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<any>(null)

  useEffect(() => {
    if (!ref.current) return
    const chart = createChart(ref.current, {
      layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
      grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
      rightPriceScale: { borderColor: '#252d3d' },
      timeScale: { borderColor: '#252d3d', timeVisible: false },
      width: ref.current.clientWidth,
      height: 380,
      crosshair: { mode: 0 },
    })
    const s = chart.addSeries(LineSeries, { color: def.color, lineWidth: 2, title: def.label })
    chartRef.current = chart
    seriesRef.current = s
    const ro = new ResizeObserver(() => {
      if (ref.current) chart.resize(ref.current.clientWidth, 380)
    })
    ro.observe(ref.current)
    return () => { ro.disconnect(); chart.remove() }
  }, [def.label, def.color])

  useEffect(() => {
    if (!seriesRef.current) return
    const data = [...def.points()].sort((a, b) => a.time - b.time)
    seriesRef.current.setData(data)
    chartRef.current?.timeScale().fitContent()
  }, [def, historyTick])

  const points = def.points()
  const n = points.length
  const last = n > 0 ? points[n - 1].value : null
  const min = n > 0 ? Math.min(...points.map(p => p.value)) : null
  const max = n > 0 ? Math.max(...points.map(p => p.value)) : null
  const lastStep = n > 0 ? points[n - 1].time - 1000000 : null
  const firstStep = n > 0 ? points[0].time - 1000000 : null

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(4px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000, padding: 24,
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        className="card"
        style={{ width: '100%', maxWidth: 960, padding: 20, position: 'relative' }}
      >
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          style={{
            position: 'absolute', top: 12, right: 12,
            background: 'transparent', border: 'none', color: 'var(--fg-dim)',
            fontSize: 20, cursor: 'pointer', lineHeight: 1, padding: 4,
          }}
        >×</button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: '50%', background: def.color }} />
          <h2 style={{ margin: 0, fontSize: 18 }}>{def.label} — full history</h2>
        </div>
        <div style={{ fontSize: 12, color: 'var(--fg-dim)', marginBottom: 16 }}>
          {n > 0
            ? `${n} points · steps ${firstStep?.toLocaleString()} → ${lastStep?.toLocaleString()} · last ${def.format(last!)}`
            : 'No data points yet — wait for the next training event.'}
        </div>

        <div ref={ref} style={{ width: '100%' }} />

        {n > 0 && (
          <div className="metric-row" style={{ marginTop: 16, gap: 24 }}>
            <div className="metric"><div className="label">Last</div><div className="value">{def.format(last!)}</div></div>
            <div className="metric"><div className="label">Min</div><div className="value">{def.format(min!)}</div></div>
            <div className="metric"><div className="label">Max</div><div className="value">{def.format(max!)}</div></div>
            <div className="metric"><div className="label">Points</div><div className="value">{n}</div></div>
            <div className="metric"><div className="label">Last step</div><div className="value">{lastStep?.toLocaleString()}</div></div>
          </div>
        )}

        <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginTop: 16 }}>
          Press <kbd style={{ background: '#1c2230', padding: '1px 6px', borderRadius: 3, fontSize: 10 }}>Esc</kbd> or click outside to close.
        </div>
      </div>
    </div>
  )
}
