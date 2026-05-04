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

  const fetchEvents = useCallback(async () => {
    try {
      const res = await getTrainingEvents(cursorRef.current)
      if (!res.events?.length) return
      let changed = false
      for (const ev of res.events) {
        if (ev.kind === 'step' && ev.step != null && ev.loss != null) {
          trainPoints.current.push({ time: ev.step + 1000000, value: ev.loss })
          if (ev.throughput_tok_per_s != null) {
            throughputPoints.current.push({ time: ev.step + 1000000, value: ev.throughput_tok_per_s })
          }
          changed = true
        }
        if (ev.kind === 'val' && ev.step != null && ev.val_loss != null) {
          valPoints.current.push({ time: ev.step + 1000000, value: ev.val_loss })
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
      }
      if (res.cursor) cursorRef.current = res.cursor
    } catch { /* ignore */ }
  }, [])

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
  const progress = status.progress_frac ?? 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header strip */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--fg-dim)' }}>{status.run_id}</span>
        <span style={{ background: stateColor, color: '#0b0e13', borderRadius: 4, padding: '2px 8px', fontSize: 12, fontWeight: 600 }}>
          {status.state}
        </span>
        <span style={{ color: 'var(--fg-dim)', fontSize: 13 }}>elapsed {fmt_s(status.elapsed_s)}</span>
        <span style={{ color: 'var(--fg-dim)', fontSize: 13 }}>ETA {fmt_s(status.eta_s)}</span>
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
            {[
              ['Train loss', status.train_loss != null ? status.train_loss.toFixed(4) : '—'],
              ['Val loss',   status.val_loss   != null ? status.val_loss.toFixed(4)   : '—'],
              ['Best val',  status.best_val_loss != null ? status.best_val_loss.toFixed(4) : '—'],
              ['LR',        status.lr != null ? status.lr.toExponential(2) : '—'],
              ['Grad norm', status.grad_norm != null ? status.grad_norm.toFixed(3) : '—'],
              ['Throughput', status.throughput_tok_per_s != null ? `${Math.round(status.throughput_tok_per_s).toLocaleString()} tok/s` : '—'],
            ].map(([l, v]) => (
              <div key={l as string} className="metric" style={{ minWidth: 110 }}>
                <div className="label">{l}</div>
                <div className="value" style={{ fontSize: 15 }}>{v}</div>
              </div>
            ))}
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
    </div>
  )
}
