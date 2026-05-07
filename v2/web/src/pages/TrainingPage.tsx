import { useEffect, useRef, useState, useCallback } from 'react'
import { createChart, type IChartApi, LineSeries } from 'lightweight-charts'
import { getTrainingStatus, getTrainingEvents, getSystemStats } from '../api'

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

function fmt_pct(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—'
  return `${(v * 100).toFixed(1)}%`
}

function fmt_num(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return '—'
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString()
  return v.toFixed(digits)
}

function chartSize(el: HTMLElement, fallbackHeight: number) {
  const rect = el.getBoundingClientRect()
  const width = Math.max(320, Math.floor(rect.width || el.clientWidth || 0))
  const cssHeight = parseFloat(getComputedStyle(el).height || '')
  const height = Math.max(120, Math.floor(rect.height || el.clientHeight || cssHeight || fallbackHeight))
  return { width, height }
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

  // Poll events every 5s, update chart. Important: this depends on loading
  // because the chart containers do not exist during the initial Loading state.
  useEffect(() => {
    if (loading || !status?.available || !lossRef.current || lossChart.current) return
    const chart = createChart(lossRef.current, {
      layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
      grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
      rightPriceScale: { borderColor: '#252d3d' },
      timeScale: { borderColor: '#252d3d', timeVisible: false },
      ...chartSize(lossRef.current, 260),
    })
    const ts = chart.addSeries(LineSeries, { color: '#00d4aa', lineWidth: 2, title: 'train' })
    const vs = chart.addSeries(LineSeries, { color: '#f5a623', lineWidth: 2, title: 'val' })
    lossChart.current = chart
    trainSeries.current = ts
    valSeries.current = vs

    // If events loaded before the chart mounted, draw them now.
    ts.setData([...trainPoints.current].sort((a, b) => a.time - b.time) as any)
    vs.setData([...valPoints.current].sort((a, b) => a.time - b.time) as any)
    chart.timeScale().fitContent()

    const ro = new ResizeObserver(() => {
      if (lossRef.current) {
        const { width, height } = chartSize(lossRef.current, 260)
        chart.resize(width, height)
      }
    })
    ro.observe(lossRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
      lossChart.current = null
      trainSeries.current = null
      valSeries.current = null
    }
  }, [loading, status?.available])

  // Throughput sparkline chart. Also waits until the real dashboard DOM exists.
  useEffect(() => {
    if (loading || !status?.available || !throughputRef.current || throughputChart.current) return
    const chart = createChart(throughputRef.current, {
      layout: { background: { color: '#0b0e13' }, textColor: '#8492a6' },
      grid: { vertLines: { color: '#1c2230' }, horzLines: { color: '#1c2230' } },
      rightPriceScale: { borderColor: '#252d3d' },
      timeScale: { borderColor: '#252d3d', timeVisible: false },
      ...chartSize(throughputRef.current, 140),
    })
    const ts = chart.addSeries(LineSeries, { color: '#9b59b6', lineWidth: 2 })
    throughputChart.current = chart
    throughputSeries.current = ts
    ts.setData([...throughputPoints.current].sort((a, b) => a.time - b.time) as any)
    chart.timeScale().fitContent()
    const ro = new ResizeObserver(() => {
      if (throughputRef.current) {
        const { width, height } = chartSize(throughputRef.current, 140)
        chart.resize(width, height)
      }
    })
    ro.observe(throughputRef.current)
    return () => {
      ro.disconnect()
      chart.remove()
      throughputChart.current = null
      throughputSeries.current = null
    }
  }, [loading, status?.available])

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
        trainSeries.current?.setData([...trainPoints.current].sort((a, b) => a.time - b.time) as any)
        valSeries.current?.setData([...valPoints.current].sort((a, b) => a.time - b.time) as any)
        lossChart.current?.timeScale().fitContent()
        if (throughputPoints.current.length > 0) {
          throughputSeries.current?.setData([...throughputPoints.current].sort((a, b) => a.time - b.time) as any)
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
  const stepProgress = Math.min(1, (status.step ?? 0) / Math.max(1, status.max_steps ?? 1))
  const timeProgress = Math.min(1, displayElapsed / cap)
  // The main progress bar is step progress. Wall-clock cap progress is shown
  // separately so the bar doesn't say ~4% while the label says step ~2%.
  const progress = stepProgress

  const model = status.model ?? {}
  const hasBestCheckpoint = status.best_val_loss != null
  const healthTone =
    status.state === 'failed' ? 'danger' :
    status.state === 'done' ? 'success' :
    status.state === 'training' ? 'success' :
    status.state === 'evaluating' || status.state === 'checkpointing' ? 'warning' : 'neutral'

  return (
    <div className="training-dashboard">
      <section className="training-hero card">
        <div className="training-hero-glow" />
        <div className="training-hero-main">
          <div className="eyebrow">CandleGPT training console</div>
          <div className="training-title-row">
            <h1>32M parameter run</h1>
            <span className={`status-pill status-${healthTone}`} style={{ ['--pill-color' as any]: stateColor }}>
              <span className="status-dot" />{status.state}
            </span>
          </div>
          <div className="run-id">{status.run_id}</div>
          <div className="hero-progress-meta">
            <span>Step <strong>{status.step?.toLocaleString() ?? 0}</strong> / {status.max_steps?.toLocaleString() ?? '—'}</span>
            <span>{fmt_pct(progress)} complete</span>
            <span>{fmt_s(displayElapsed)} elapsed</span>
            <span>{fmt_s(displayEta)} ETA</span>
          </div>
          <div className="pro-progress" aria-label="Training progress">
            <div className="pro-progress-fill" style={{ width: `${Math.max(2, progress * 100)}%` }} />
          </div>
          <div className="hero-subprogress">
            <span>Step progress {fmt_pct(stepProgress)}</span>
            <span>Wall-clock cap {fmt_pct(timeProgress)}</span>
            <span>Last update {status.last_update_utc ? new Date(status.last_update_utc).toLocaleTimeString() : '—'}</span>
          </div>
        </div>
        <div className="training-hero-side">
          <div className="hero-side-card">
            <span>Best validation</span>
            <strong>{status.best_val_loss != null ? status.best_val_loss.toFixed(4) : '—'}</strong>
            <em>{hasBestCheckpoint ? `last eval @ ${status.last_eval_step ?? '—'}` : 'waiting for first eval'}</em>
          </div>
          <div className="hero-side-card muted">
            <span>Checkpoint</span>
            <strong>{status.last_checkpoint_step ?? (hasBestCheckpoint ? 'best_val saved' : '—')}</strong>
            <em>{status.last_checkpoint_step ? 'latest saved step' : 'best checkpoint available'}</em>
          </div>
        </div>
      </section>

      <section className="dashboard-filter-strip card" aria-label="Dashboard filters">
        <FilterChip label="Run" value={status.run_id} />
        <FilterChip label="State" value={status.state} toneColor={stateColor} />
        <FilterChip label="Device" value={status.hardware?.device_name ?? status.hardware?.device ?? '—'} />
        <FilterChip label="Interval" value={model.interval ?? '—'} />
        <FilterChip label="Window" value={model.window ? `${model.window} bars` : '—'} />
        <FilterChip label="Features" value={model.n_features ?? '—'} />
      </section>

      <section className="kpi-grid kpi-grid-extended">
        <MetricTile
          label="Train loss"
          value={status.train_loss != null ? status.train_loss.toFixed(4) : '—'}
          accent="#00d4aa"
          points={METRIC_DEFS.train_loss.points().length}
          onClick={() => setSelectedMetric('train_loss')}
        />
        <MetricTile
          label="Validation loss"
          value={status.val_loss != null ? status.val_loss.toFixed(4) : '—'}
          accent="#f5a623"
          points={METRIC_DEFS.val_loss.points().length}
          onClick={() => setSelectedMetric('val_loss')}
        />
        <MetricTile
          label="Learning rate"
          value={status.lr != null ? status.lr.toExponential(2) : '—'}
          accent="#4a90e2"
          points={METRIC_DEFS.lr.points().length}
          onClick={() => setSelectedMetric('lr')}
        />
        <MetricTile
          label="Throughput"
          value={status.throughput_tok_per_s != null ? `${Math.round(status.throughput_tok_per_s).toLocaleString()}` : '—'}
          suffix="tok/s"
          accent="#9b59b6"
          points={METRIC_DEFS.throughput.points().length}
          onClick={() => setSelectedMetric('throughput')}
        />
        <MetricTile
          label="Grad norm"
          value={status.grad_norm != null ? status.grad_norm.toFixed(3) : '—'}
          accent="#e74c3c"
          points={METRIC_DEFS.grad_norm.points().length}
          onClick={() => setSelectedMetric('grad_norm')}
        />
        <MetricTile
          label="Step progress"
          value={fmt_pct(stepProgress)}
          accent="#00b8ff"
          points={0}
          onClick={() => {}}
        />
        <MetricTile
          label="Wall clock"
          value={fmt_pct(timeProgress)}
          accent="#ffcc66"
          points={0}
          onClick={() => {}}
        />
        <MetricTile
          label="Checkpoint"
          value={`${status.last_checkpoint_step ?? (hasBestCheckpoint ? 'best' : '—')}`}
          accent="#b084ff"
          points={0}
          onClick={() => {}}
        />
        <MetricTile
          label="ETA"
          value={fmt_s(displayEta)}
          accent="#7dd3fc"
          points={0}
          onClick={() => {}}
        />
      </section>

      <section className="dashboard-grid-main">
        <div className="card chart-card loss-card">
          <div className="panel-header">
            <div>
              <div className="eyebrow">Optimization</div>
              <h2>Loss curves</h2>
            </div>
            <div className="legend-pills">
              <span><i style={{ background: '#00d4aa' }} /> train</span>
              <span><i style={{ background: '#f5a623' }} /> val</span>
              <span>{eventCount} events</span>
            </div>
          </div>
          <div ref={lossRef} className="chart-host loss-chart" />
        </div>

        <aside className="side-stack">
          <div className="card pro-panel">
            <div className="panel-header compact">
              <div>
                <div className="eyebrow">Run health</div>
                <h2>Live status</h2>
              </div>
            </div>
            <div className="status-list">
              <StatusLine label="State" value={status.state} color={stateColor} />
              <StatusLine label="Elapsed" value={fmt_s(displayElapsed)} />
              <StatusLine label="ETA" value={fmt_s(displayEta)} />
              <StatusLine label="Step progress" value={fmt_pct(stepProgress)} />
              <StatusLine label="Wall limit" value={fmt_pct(timeProgress)} />
              <StatusLine label="Last eval" value={status.last_eval_step?.toLocaleString?.() ?? '—'} />
            </div>
          </div>

          <div className="card pro-panel">
            <div className="panel-header compact">
              <div>
                <div className="eyebrow">Architecture</div>
                <h2>Model spec</h2>
              </div>
            </div>
            <div className="spec-grid-pro">
              <MiniSpec label="Params" value={model.n_params ? `${(model.n_params / 1e6).toFixed(1)}M` : '—'} />
              <MiniSpec label="Layers" value={model.n_layers ?? '—'} />
              <MiniSpec label="Heads" value={model.n_heads ?? '—'} />
              <MiniSpec label="d_model" value={model.d_model ?? '—'} />
              <MiniSpec label="Window" value={model.window ? `${model.window} bars` : '—'} />
              <MiniSpec label="Features" value={model.n_features ?? '—'} />
              <MiniSpec label="Interval" value={model.interval ?? '—'} />
              <MiniSpec label="Bins" value={model.n_bins ?? '—'} />
            </div>
          </div>
        </aside>
      </section>

      <section className="dashboard-grid-secondary">
        <div className="card chart-card">
          <div className="panel-header">
            <div>
              <div className="eyebrow">Performance</div>
              <h2>Throughput</h2>
            </div>
            <div className="panel-stat">{fmt_num(status.throughput_tok_per_s, 0)} tok/s</div>
          </div>
          <div ref={throughputRef} className="chart-host throughput-chart" />
        </div>

        {status.hardware && (
          <div className="card pro-panel hardware-panel">
            <div className="panel-header compact">
              <div>
                <div className="eyebrow">Hardware</div>
                <h2>Runtime</h2>
              </div>
            </div>
            <div className="status-list">
              <StatusLine label="Device" value={`${status.hardware.device} — ${status.hardware.device_name}`} />
              <StatusLine label="Host" value={status.hardware.hostname} />
              <StatusLine label="CPU" value={`${status.hardware.cpu_count} cores`} />
              <StatusLine label="RAM" value={`${status.hardware.ram_gb} GB`} />
              <StatusLine label="PyTorch" value={status.hardware.torch} />
              <StatusLine label="Python" value={status.hardware.python} />
            </div>
          </div>
        )}
      </section>

      <section className="card pro-panel dashboard-detail-panel">
        <div className="panel-header compact">
          <div>
            <div className="eyebrow">Dashboard QA</div>
            <h2>Training detail table</h2>
          </div>
          <div className="panel-stat">grain: run × training step</div>
        </div>
        <div className="dashboard-detail-grid">
          <MiniSpec label="Run ID" value={status.run_id} />
          <MiniSpec label="Current step" value={`${status.step?.toLocaleString() ?? 0} / ${status.max_steps?.toLocaleString() ?? '—'}`} />
          <MiniSpec label="Latest val" value={status.val_loss != null ? status.val_loss.toFixed(4) : '—'} />
          <MiniSpec label="Best val" value={status.best_val_loss != null ? status.best_val_loss.toFixed(4) : '—'} />
          <MiniSpec label="Last eval" value={status.last_eval_step?.toLocaleString?.() ?? '—'} />
          <MiniSpec label="Checkpoint" value={`${status.last_checkpoint_step ?? (hasBestCheckpoint ? 'best_val' : '—')}`} />
          <MiniSpec label="Elapsed" value={fmt_s(displayElapsed)} />
          <MiniSpec label="ETA" value={fmt_s(displayEta)} />
        </div>
      </section>

      <SystemStatsPanel />

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

function MetricTile({
  label,
  value,
  suffix,
  accent,
  points,
  onClick,
}: {
  label: string
  value: string
  suffix?: string
  accent: string
  points: number
  onClick: () => void
}) {
  return (
    <button type="button" className="kpi-card" onClick={onClick} style={{ ['--tile-accent' as any]: accent }}>
      <div className="kpi-topline"><span>{label}</span><i /></div>
      <div className="kpi-value">{value}{suffix && <small>{suffix}</small>}</div>
      <div className="kpi-foot">{points > 0 ? `${points} history points · click to inspect` : 'waiting for history'}</div>
    </button>
  )
}

function FilterChip({ label, value, toneColor }: { label: string; value: string | number; toneColor?: string }) {
  return (
    <div className="filter-chip" style={{ ['--chip-color' as any]: toneColor ?? 'var(--accent)' }}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function StatusLine({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="status-line">
      <span>{label}</span>
      <strong style={{ color: color ?? 'var(--fg)' }}>{value}</strong>
    </div>
  )
}

function MiniSpec({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="mini-spec">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}
function SystemStatsPanel() {
  const [stats, setStats] = useState<any>(null)
  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const s = await getSystemStats()
        if (!cancelled) setStats(s)
      } catch { /* ignore */ }
    }
    load()
    const id = setInterval(load, 1500)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  if (!stats) {
    return <div className="card system-panel loading-panel">Loading system stats…</div>
  }

  const cpu = stats.cpu_percent
  const ram = stats.ram_percent
  const gpu = stats.gpu_util_percent
  const thermal = stats.thermal ?? {}

  return (
    <section className="card system-panel">
      <div className="panel-header">
        <div>
          <div className="eyebrow">Host telemetry</div>
          <h2>Mac stats</h2>
        </div>
        <div className="panel-stat">1.5s refresh</div>
      </div>

      <div className="usage-grid">
        <UsageGauge label={`CPU (${stats.cpu_count ?? '?'} cores)`} pct={cpu} color="#4a90e2" suffix="%" />
        <UsageGauge label="RAM" pct={ram} color="#f5a623" suffix={ram != null ? `% · ${stats.ram_used_gb ?? '?'} / ${stats.ram_total_gb ?? '?'} GB` : ''} />
        <UsageGauge label="GPU (MPS)" pct={gpu} color="#00d4aa" suffix="%" />
      </div>

      {Array.isArray(stats.cpu_per_core) && stats.cpu_per_core.length > 0 && (
        <div className="core-panel">
          <div className="core-panel-label">Per-core CPU</div>
          <div className="core-bars">
            {stats.cpu_per_core.map((p: number, i: number) => (
              <div
                key={i}
                className="core-bar"
                title={`Core ${i}: ${p.toFixed(1)}%`}
                style={{
                  height: `${Math.max(2, p)}%`,
                  background: p > 80 ? '#f05252' : p > 50 ? '#f5a623' : '#4a90e2',
                }}
              />
            ))}
          </div>
        </div>
      )}

      <div className="thermal-panel">
        <div className="thermal-title">Thermal + power</div>
        {thermal.available ? (
          <>
            <div className="thermal-grid">
              <ThermalPressure level={thermal.pressure_level} />
              <Stat label="GPU power" value={thermal.gpu_power_mw != null ? `${(thermal.gpu_power_mw / 1000).toFixed(2)} W` : '—'} color="#00d4aa" />
              <Stat label="CPU power" value={thermal.cpu_power_mw != null ? `${(thermal.cpu_power_mw / 1000).toFixed(2)} W` : '—'} color="#4a90e2" />
            </div>
            {thermal.gpu_freq_mhz != null && (
              <div className="gpu-frequency">GPU active freq: {thermal.gpu_freq_mhz} MHz</div>
            )}
            <div className="thermal-note">
              Apple Silicon exposes thermal pressure and power draw as the reliable public proxies. Anything above Nominal means macOS may be throttling to manage heat.
            </div>
          </>
        ) : (
          <div className="thermal-note">{thermal.hint || 'Thermal readings unavailable.'}</div>
        )}
      </div>
    </section>
  )
}

function UsageGauge({ label, pct, color, suffix }: { label: string; pct: number | null; color: string; suffix: string }) {
  const v = pct ?? 0
  const ringColor = v > 90 ? '#f05252' : v > 70 ? '#f5a623' : color
  return (
    <div className="usage-gauge" style={{ ['--gauge-color' as any]: ringColor }}>
      <div className="usage-gauge-head">
        <span>{label}</span>
        <strong>{pct != null ? `${pct.toFixed(0)}` : '—'}<small>{pct != null ? suffix : ''}</small></strong>
      </div>
      <div className="usage-track">
        <div className="usage-fill" style={{ width: `${Math.max(0, Math.min(100, v))}%` }} />
      </div>
    </div>
  )
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="stat-block" style={{ ['--stat-color' as any]: color ?? 'var(--fg)' }}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function ThermalPressure({ level }: { level: string | null | undefined }) {
  const color =
    level === 'Critical' ? '#f05252' :
    level === 'Serious'  ? '#f58423' :
    level === 'Fair'     ? '#f5a623' :
    level === 'Nominal'  ? '#00d4aa' :
                           'var(--fg-dim)'
  return (
    <div className="thermal-pressure" style={{ ['--pressure-color' as any]: color }}>
      <span>Thermal pressure</span>
      <strong><i />{level ?? '—'}</strong>
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
