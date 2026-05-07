import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { getTrainingStatus, getTrainingEvents, getSystemStats } from '../api'
import { Panel, SLabel, Divider, MBox, Pill, PulseDot } from '../components/dash'

const TERMINAL_STATES = new Set(['done', 'failed', 'stopped', 'completed'])

// ---------- helpers --------------------------------------------------------

function fmt_s(s: number | null | undefined): string {
  if (s == null) return '—'
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60)
  if (h > 0) return `${h}h ${m}m`
  return `${m}m ${sec}s`
}

interface Pt { step: number; value: number }

// ---------- LossChart (inline SVG, ported from prototype) ----------------

function LossChart({
  train,
  val,
  bestStep,
  bestVal,
  W = 500,
  H = 180,
}: {
  train: Pt[]
  val: Pt[]
  bestStep: number | null
  bestVal: number | null
  W?: number
  H?: number
}) {
  const [tip, setTip] = useState<{ i: number; x: number } | null>(null)
  const all = [...train.map(p => p.value), ...val.map(p => p.value)]
  if (!all.length) {
    return (
      <div style={{ height: H, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#3f3f46', fontSize: 10 }}>
        Waiting for events…
      </div>
    )
  }
  const minV = Math.min(...all) * 0.97
  const maxV = Math.max(...all) * 1.03
  const rng = Math.max(maxV - minV, 1e-9)
  const stepMin = Math.min(...[...train, ...val].map(p => p.step))
  const stepMax = Math.max(...[...train, ...val].map(p => p.step))
  const stepRng = Math.max(stepMax - stepMin, 1)
  const px = (s: number) => 16 + ((s - stepMin) / stepRng) * (W - 32)
  const py = (v: number) => H - 20 - ((v - minV) / rng) * (H - 32)
  const path = (data: Pt[]) =>
    data
      .map((p, i) => `${i === 0 ? 'M' : 'L'}${px(p.step).toFixed(1)},${py(p.value).toFixed(1)}`)
      .join(' ')
  const ticks = [minV, minV + rng * 0.5, maxV]

  const pickIdx = (rx: number) => {
    if (!train.length) return null
    let best = 0
    let bestD = Infinity
    for (let i = 0; i < train.length; i++) {
      const d = Math.abs(px(train[i].step) - rx)
      if (d < bestD) { bestD = d; best = i }
    }
    return best
  }

  return (
    <div style={{ position: 'relative' }}>
      <svg
        width="100%"
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        style={{ overflow: 'visible' }}
        onMouseMove={e => {
          const r = e.currentTarget.getBoundingClientRect()
          const rx = ((e.clientX - r.left) / r.width) * W
          const i = pickIdx(rx)
          if (i != null) setTip({ i, x: rx })
        }}
        onMouseLeave={() => setTip(null)}
      >
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={16} x2={W - 8} y1={py(t)} y2={py(t)} stroke="rgba(255,255,255,.04)" strokeWidth="1" />
            <text
              x={W - 6}
              y={py(t) + 3}
              fontSize="8"
              fill="#3f3f46"
              fontFamily="'JetBrains Mono',monospace"
              textAnchor="end"
            >
              {t.toFixed(3)}
            </text>
          </g>
        ))}
        {bestStep != null && bestVal != null && (
          <line
            x1={px(bestStep)}
            x2={px(bestStep)}
            y1={6}
            y2={H - 20}
            stroke="rgba(251,191,36,.25)"
            strokeWidth="1"
            strokeDasharray="2 3"
          />
        )}
        {train.length > 1 && (
          <path d={path(train)} fill="none" stroke="#38bdf8" strokeWidth="1.5" strokeLinejoin="round" />
        )}
        {val.length > 1 && (
          <path d={path(val)} fill="none" stroke="#fb7185" strokeWidth="1.5" strokeLinejoin="round" strokeDasharray="4 2" />
        )}
        {bestStep != null && bestVal != null && (
          <circle cx={px(bestStep)} cy={py(bestVal)} r="3.5" fill="#fbbf24" />
        )}
        {tip && train[tip.i] && (
          <>
            <line x1={tip.x} x2={tip.x} y1={6} y2={H - 20} stroke="#fbbf24" strokeWidth="1" strokeDasharray="2 4" opacity=".4" />
            <circle cx={tip.x} cy={py(train[tip.i].value)} r="2.5" fill="#38bdf8" />
          </>
        )}
      </svg>
      {tip && train[tip.i] && (
        <div
          style={{
            position: 'absolute',
            top: 6,
            left: Math.min(tip.x + 8, W - 120),
            background: 'rgba(12,12,14,.97)',
            border: '1px solid rgba(255,255,255,.08)',
            borderRadius: 8,
            padding: '5px 9px',
            fontSize: 9,
            lineHeight: 1.7,
            fontFamily: "'JetBrains Mono',monospace",
            pointerEvents: 'none',
            whiteSpace: 'nowrap',
          }}
        >
          <div style={{ color: '#52525b' }}>step {train[tip.i].step.toLocaleString()}</div>
          <div style={{ color: '#38bdf8' }}>train {train[tip.i].value.toFixed(4)}</div>
        </div>
      )}
      <div style={{ display: 'flex', gap: 14, marginTop: 7, fontSize: 8, color: '#52525b', letterSpacing: '.1em' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 12, height: 2, background: '#38bdf8', display: 'inline-block', borderRadius: 1 }} />
          TRAIN
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 12, height: 2, background: '#fb7185', display: 'inline-block', borderRadius: 1 }} />
          VAL
        </span>
        {bestVal != null && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#fbbf24', display: 'inline-block' }} />
            BEST VAL {bestVal.toFixed(4)}
          </span>
        )}
      </div>
    </div>
  )
}

// ---------- TrainingPage --------------------------------------------------

export function TrainingPage() {
  const [status, setStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const cursorRef = useRef<number | null>(null)
  const [, setTick] = useState(0)

  const trainPoints = useRef<Pt[]>([])
  const valPoints = useRef<Pt[]>([])
  const tokpsHist = useRef<Pt[]>([])
  const lrHist = useRef<Pt[]>([])
  const gradNormHist = useRef<Pt[]>([])

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

  // Local interpolation tick (for elapsed/eta drift between server writes)
  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 500)
    return () => clearInterval(id)
  }, [])

  const fetchEvents = useCallback(async () => {
    try {
      const res = await getTrainingEvents(cursorRef.current)
      if (!res.events?.length) return
      for (const ev of res.events) {
        if (ev.kind === 'step' && ev.step != null) {
          if (ev.loss != null) trainPoints.current.push({ step: ev.step, value: ev.loss })
          if (ev.throughput_tok_per_s != null && ev.throughput_tok_per_s > 0) {
            tokpsHist.current.push({ step: ev.step, value: ev.throughput_tok_per_s })
            if (tokpsHist.current.length > 200) tokpsHist.current.shift()
          }
          if (ev.lr != null && ev.lr > 0) {
            lrHist.current.push({ step: ev.step, value: ev.lr })
            if (lrHist.current.length > 200) lrHist.current.shift()
          }
          if (ev.grad_norm != null) {
            gradNormHist.current.push({ step: ev.step, value: ev.grad_norm })
            if (gradNormHist.current.length > 200) gradNormHist.current.shift()
          }
        }
        if (ev.kind === 'val' && ev.step != null && ev.val_loss != null) {
          valPoints.current.push({ step: ev.step, value: ev.val_loss })
        }
      }
      if (res.cursor) cursorRef.current = res.cursor
      setTick(t => t + 1)
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    fetchEvents()
    const id = setInterval(fetchEvents, 5000)
    return () => clearInterval(id)
  }, [fetchEvents])

  const bestVal = useMemo(() => {
    if (!valPoints.current.length) return null
    return valPoints.current.reduce((b, p) => (p.value < b.value ? p : b), valPoints.current[0])
  }, [valPoints.current.length])

  if (loading) {
    return <div style={{ color: '#52525b', padding: 24, fontSize: 11 }}>Loading training status…</div>
  }

  if (!status?.available) {
    return (
      <Panel style={{ padding: 24, maxWidth: 520 }}>
        <SLabel>No training run</SLabel>
        <div style={{ fontSize: 11, color: '#71717a', lineHeight: 1.7 }}>
          Start one:{' '}
          <code style={{ color: '#7dd3fc' }}>uv run python -m v2.train.run --raw-dir v2/data/raw</code>
        </div>
      </Panel>
    )
  }

  // Live drift on elapsed/eta
  const liveState = ['training', 'evaluating', 'checkpointing'].includes(status.state)
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
  const maxSteps = status.max_steps ?? 1
  const prog = Math.min(1, (status.step ?? 0) / Math.max(1, maxSteps))
  const etaH = Math.floor(displayEta / 3600)
  const etaM = Math.floor((displayEta % 3600) / 60)

  const model = status.model ?? {}
  const hardware = status.hardware ?? {}

  const isTerminal = TERMINAL_STATES.has(status.state)
  const isFailed = status.state === 'failed'
  const isDone = status.state === 'done' || status.state === 'completed' || status.state === 'stopped'

  const trainLossLast = trainPoints.current[trainPoints.current.length - 1]?.value ?? status.train_loss
  const valLossLast = valPoints.current[valPoints.current.length - 1]?.value ?? status.val_loss

  // For terminal states, ignore the zeros that the backend writes to status.json.
  // Fall back to the last-known good value from the streamed events (if any).
  const tokpsHistLast = tokpsHist.current[tokpsHist.current.length - 1]?.value
  const lrHistLast = lrHist.current[lrHist.current.length - 1]?.value
  const gradNormHistLast = gradNormHist.current[gradNormHist.current.length - 1]?.value

  const liveTokps =
    status.throughput_tok_per_s != null && status.throughput_tok_per_s > 0
      ? status.throughput_tok_per_s
      : tokpsHistLast
  const liveLr =
    status.lr != null && status.lr > 0 ? status.lr : lrHistLast
  const liveGradNorm =
    status.grad_norm != null ? status.grad_norm : gradNormHistLast

  // Final gauge values shown in MBoxes.
  // - During live runs: show current value or "—".
  // - After terminal: show last-known-good with a "last known" sub-label,
  //   or "n/a" if we never observed any non-zero values.
  const tokpsDisplay = isTerminal ? tokpsHistLast : liveTokps
  const lrDisplay = isTerminal ? lrHistLast : liveLr
  const gradNormDisplay = isTerminal ? gradNormHistLast : liveGradNorm

  return (
    <div className="cgpt-train-layout">
      {/* Run bar — full width */}
      <Panel style={{ padding: '12px 18px', gridColumn: '1/-1' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <PulseDot color={isFailed ? '#fb7185' : isDone ? '#a3a3a3' : status.state === 'training' ? '#4ade80' : '#fbbf24'} />
            <span
              style={{
                fontSize: 9,
                letterSpacing: '.15em',
                textTransform: 'uppercase',
                color: '#71717a',
              }}
            >
              {status.state}
            </span>
          </div>
          {isTerminal && (
            <Pill color={isFailed ? '#fb7185' : '#4ade80'}>
              {isFailed ? 'Failed' : 'Completed'}
            </Pill>
          )}
          <span style={{ height: 12, width: 1, background: 'rgba(255,255,255,.06)' }} />
          {(isTerminal
            ? [
                ['run', status.run_id ?? '—'],
                ['total', fmt_s(displayElapsed)],
              ]
            : [
                ['run', status.run_id ?? '—'],
                ['elapsed', fmt_s(displayElapsed)],
                ['eta', `${etaH}h ${etaM}m`],
              ]
          ).map(([k, v]) => (
            <span key={k} style={{ fontSize: 9, color: '#3f3f46', fontFamily: "'JetBrains Mono',monospace" }}>
              {k} <span style={{ color: '#71717a' }}>{v}</span>
            </span>
          ))}
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 9, color: '#3f3f46' }}>
              {(status.step ?? 0).toLocaleString()} / {maxSteps.toLocaleString()}
            </span>
            <div style={{ width: 160, height: 3, background: 'rgba(255,255,255,.06)', borderRadius: 999, overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${prog * 100}%`,
                  background: 'linear-gradient(90deg,#38bdf8,#818cf8)',
                  borderRadius: 999,
                  transition: 'width 800ms',
                }}
              />
            </div>
            <span style={{ fontSize: 9, color: '#71717a' }}>{(prog * 100).toFixed(1)}%</span>
          </div>
        </div>
      </Panel>

      {/* Loss */}
      <Panel style={{ padding: 16 }}>
        <SLabel>Loss</SLabel>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12, marginBottom: 14 }}>
          <MBox label="Train" value={trainLossLast != null ? trainLossLast.toFixed(4) : '—'} accent="#38bdf8" />
          <MBox label="Val" value={valLossLast != null ? valLossLast.toFixed(4) : '—'} accent="#fb7185" />
          <MBox
            label="Best Val"
            value={status.best_val_loss != null ? status.best_val_loss.toFixed(4) : (bestVal?.value.toFixed(4) ?? '—')}
            accent="#fbbf24"
            sub={bestVal ? `step ${bestVal.step.toLocaleString()}` : undefined}
          />
          <MBox
            label="Grad Norm"
            value={gradNormDisplay != null ? gradNormDisplay.toFixed(3) : (isTerminal ? 'n/a' : '—')}
            sub={isTerminal && gradNormDisplay != null ? 'last known' : undefined}
          />
        </div>
        <LossChart
          train={trainPoints.current}
          val={valPoints.current}
          bestStep={bestVal?.step ?? null}
          bestVal={bestVal?.value ?? null}
          W={480}
          H={170}
        />
      </Panel>

      {/* Throughput + spec */}
      <Panel style={{ padding: 16 }}>
        <SLabel>Throughput &amp; Schedule</SLabel>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 12, marginBottom: 14 }}>
          <MBox
            label="Tok / s"
            value={tokpsDisplay != null ? `${(tokpsDisplay / 1000).toFixed(1)}k` : (isTerminal ? 'n/a' : '—')}
            sub={isTerminal && tokpsDisplay != null ? 'last known' : undefined}
          />
          <MBox
            label="Learn Rate"
            value={lrDisplay != null ? lrDisplay.toExponential(1) : (isTerminal ? 'n/a' : '—')}
            sub={isTerminal && lrDisplay != null ? 'last known' : undefined}
          />
          <MBox label={isTerminal ? 'Final Step' : 'Step'} value={(status.step ?? 0).toLocaleString()} />
          <MBox label="Progress" value={`${(prog * 100).toFixed(1)}%`} meter={prog} />
        </div>
        <Divider />
        <SLabel>Model Spec</SLabel>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8 }}>
          {[
            ['Params', model.n_params ? `${(model.n_params / 1e6).toFixed(1)}M` : '—'],
            ['Layers', model.n_layers ?? '—'],
            ['Heads', model.n_heads ?? '—'],
            ['d_model', model.d_model ?? '—'],
            ['n_bins', model.n_bins ?? '—'],
            ['Block', model.window ?? '—'],
            ['Interval', model.interval ?? '—'],
            ['Features', model.n_features ?? '—'],
          ].map(([k, v]) => (
            <div key={k}>
              <div
                style={{
                  fontSize: 7,
                  letterSpacing: '.15em',
                  textTransform: 'uppercase',
                  color: '#2a2a2e',
                  marginBottom: 2,
                }}
              >
                {k}
              </div>
              <div
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  color: '#71717a',
                  fontFamily: "'JetBrains Mono',monospace",
                }}
              >
                {v as any}
              </div>
            </div>
          ))}
        </div>
      </Panel>

      {/* Hardware */}
      <HardwarePanel hardware={hardware} />
    </div>
  )
}

function HardwarePanel({ hardware }: { hardware: any }) {
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

  const gpuUtil = stats?.gpu_util_percent
  const ramPct = stats?.ram_percent
  const cpuCores: number[] = Array.isArray(stats?.cpu_per_core) ? stats.cpu_per_core : []
  const thermal = stats?.thermal ?? {}
  const thermalLevel: string | undefined = thermal.pressure_level
  const thermalColor =
    thermalLevel === 'Critical' ? '#fb7185' :
    thermalLevel === 'Serious'  ? '#f58423' :
    thermalLevel === 'Fair'     ? '#fbbf24' :
                                  '#4ade80'

  return (
    <Panel style={{ padding: 16, gridColumn: '1/-1' }}>
      <SLabel>Hardware · System</SLabel>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 16 }}>
        {/* GPU */}
        <div>
          <div
            style={{
              fontSize: 8,
              color: '#2a2a2e',
              letterSpacing: '.12em',
              textTransform: 'uppercase',
              marginBottom: 8,
            }}
          >
            GPU · {hardware.device_name ?? hardware.device ?? '—'}
          </div>
          <MBox
            label="Utilisation"
            value={gpuUtil != null ? `${gpuUtil.toFixed(0)}%` : '—'}
            meter={gpuUtil != null ? gpuUtil / 100 : undefined}
          />
          <div style={{ marginTop: 10 }}>
            <MBox
              label="Power"
              value={thermal.gpu_power_mw != null ? `${(thermal.gpu_power_mw / 1000).toFixed(1)}W` : '—'}
            />
          </div>
          <div style={{ marginTop: 10 }}>
            <MBox label="Freq" value={thermal.gpu_freq_mhz != null ? `${thermal.gpu_freq_mhz}MHz` : '—'} />
          </div>
        </div>
        {/* Memory */}
        <div>
          <div
            style={{
              fontSize: 8,
              color: '#2a2a2e',
              letterSpacing: '.12em',
              textTransform: 'uppercase',
              marginBottom: 8,
            }}
          >
            Memory
          </div>
          <MBox
            label="RAM Used"
            value={ramPct != null ? `${ramPct.toFixed(0)}%` : '—'}
            meter={ramPct != null ? ramPct / 100 : undefined}
          />
          <div style={{ marginTop: 10 }}>
            <div
              style={{
                fontSize: 8,
                color: '#3f3f46',
                letterSpacing: '.1em',
                textTransform: 'uppercase',
                marginBottom: 3,
              }}
            >
              Used / Total
            </div>
            <div style={{ fontFamily: "'Fraunces',serif", fontSize: 16, color: '#71717a', fontWeight: 500 }}>
              {stats?.ram_used_gb ?? '—'} / {stats?.ram_total_gb ?? hardware.ram_gb ?? '—'} GB
            </div>
          </div>
        </div>
        {/* Thermal */}
        <div>
          <div
            style={{
              fontSize: 8,
              color: '#2a2a2e',
              letterSpacing: '.12em',
              textTransform: 'uppercase',
              marginBottom: 8,
            }}
          >
            Thermal
          </div>
          <div style={{ marginBottom: 10 }}>
            <div
              style={{
                fontSize: 8,
                color: '#3f3f46',
                letterSpacing: '.1em',
                textTransform: 'uppercase',
                marginBottom: 5,
              }}
            >
              Pressure
            </div>
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                padding: '4px 10px',
                borderRadius: 8,
                fontSize: 10,
                fontWeight: 700,
                letterSpacing: '.08em',
                textTransform: 'uppercase',
                background: 'rgba(74,222,128,.07)',
                border: `1px solid ${thermalColor}33`,
                color: thermalColor,
              }}
            >
              <span style={{ width: 5, height: 5, borderRadius: '50%', background: thermalColor }} />
              {thermalLevel ?? 'Unknown'}
            </div>
          </div>
          <MBox
            label="CPU Power"
            value={thermal.cpu_power_mw != null ? `${(thermal.cpu_power_mw / 1000).toFixed(1)}W` : '—'}
          />
        </div>
        {/* Cores */}
        <div>
          <div
            style={{
              fontSize: 8,
              color: '#2a2a2e',
              letterSpacing: '.12em',
              textTransform: 'uppercase',
              marginBottom: 8,
            }}
          >
            CPU Cores ({cpuCores.length || hardware.cpu_count || '—'})
          </div>
          {cpuCores.slice(0, 12).map((v, i) => (
            <div key={i} style={{ marginBottom: 5 }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: 8,
                  color: '#3f3f46',
                  marginBottom: 2,
                }}
              >
                <span>Core {i}</span>
                <span style={{ color: '#52525b' }}>{v.toFixed(0)}%</span>
              </div>
              <div style={{ height: 3, background: 'rgba(255,255,255,.04)', borderRadius: 999, overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${Math.max(0, Math.min(100, v))}%`,
                    background: v > 80 ? '#fb7185' : v > 60 ? '#fbbf24' : '#38bdf8',
                    borderRadius: 999,
                    transition: 'width 600ms',
                  }}
                />
              </div>
            </div>
          ))}
          {cpuCores.length === 0 && (
            <div style={{ color: '#3f3f46', fontSize: 9 }}>—</div>
          )}
        </div>
      </div>
    </Panel>
  )
}
