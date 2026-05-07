import { useState, useEffect } from 'react'
import { TrainingPage } from './pages/TrainingPage'
import { LivePage } from './pages/LivePage'
import { CalibrationPage } from './pages/CalibrationPage'
import { RegimePage } from './pages/RegimePage'
import { BacktestPage, type BacktestSeed } from './pages/BacktestPage'
import { fetchStatus, getTrainingStatus, reloadModel } from './api'
import { PulseDot } from './components/dash'

const TABS: { id: string; label: string; dot?: boolean }[] = [
  { id: 'live',        label: 'Live', dot: true },
  { id: 'training',    label: 'Training' },
  { id: 'backtest',    label: 'Backtest' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'regimes',     label: 'Regime' },
]

function shortRun(id?: string | null) {
  if (!id) return '—'
  return id.length > 24 ? `${id.slice(0, 10)}…${id.slice(-8)}` : id
}

export default function App() {
  const [activeTab, setActiveTab] = useState('live')
  const [status, setStatus] = useState<any>(null)
  const [trainingStatus, setTrainingStatus] = useState<any>(null)
  const [backtestSeed, setBacktestSeed] = useState<BacktestSeed | null>(null)
  const [clock, setClock] = useState(new Date())

  const refreshHeader = async () => {
    const [model, training] = await Promise.allSettled([fetchStatus(), getTrainingStatus()])
    if (model.status === 'fulfilled') setStatus(model.value)
    if (training.status === 'fulfilled') setTrainingStatus(training.value)
  }

  useEffect(() => {
    refreshHeader().catch(console.error)
    const id = setInterval(() => refreshHeader().catch(console.error), 5000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    const id = setInterval(() => setClock(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  const sendToBacktest = (seed: BacktestSeed) => {
    setBacktestSeed(seed)
    setActiveTab('backtest')
  }

  const ts = clock.toISOString().replace('T', ' ').slice(0, 19) + ' UTC'
  const modelLoaded = !!status?.model_loaded
  const ckptStep = status?.ckpt_step
  const headerStatus = modelLoaded
    ? `Model loaded · ${shortRun(status.run_id)} · ckpt ${
        typeof ckptStep === 'number' ? (ckptStep / 1000).toFixed(1) + 'k' : '—'
      }`
    : 'Inference: no model loaded'
  const trainMismatch =
    status?.run_id && trainingStatus?.run_id && status.run_id !== trainingStatus.run_id

  const nParams = status?.n_params
  const nParamsStr = nParams != null
    ? nParams >= 1e6
      ? `${(nParams / 1e6).toFixed(0)}M params`
      : `${nParams.toLocaleString()} params`
    : '— params'

  return (
    <>
      <div className="cgpt-tabbar">
        <div className="cgpt-logo">
          Candle<span style={{ color: '#7dd3fc', fontStyle: 'italic' }}>GPT</span>
          <span style={{ fontSize: 9, color: '#2a2a2e', letterSpacing: '.12em', marginLeft: 3 }}>v2</span>
        </div>
        {TABS.map(t => (
          <button
            key={t.id}
            className={`cgpt-tab${activeTab === t.id ? ' active' : ''}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.dot && activeTab === t.id && (
              <span
                className="pulse-dot"
                style={{ width: 5, height: 5, borderRadius: '50%', background: '#4ade80', boxShadow: '0 0 6px #4ade80' }}
              />
            )}
            {t.label}
          </button>
        ))}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <PulseDot size={5} color={modelLoaded ? '#4ade80' : '#fb7185'} />
            <span
              style={{
                fontSize: 8,
                color: '#3f3f46',
                letterSpacing: '.12em',
                textTransform: 'uppercase',
              }}
            >
              {headerStatus}
            </span>
          </div>
          {trainMismatch && (
            <span style={{ fontSize: 8, color: '#fbbf24', letterSpacing: '.12em', textTransform: 'uppercase' }}>
              · training ahead of inference
            </span>
          )}
          <button
            className="cgpt-ghost-btn"
            onClick={async () => {
              try {
                const res = await reloadModel()
                await refreshHeader()
                alert(`Reloaded inference: run ${res.run_id} step ${res.ckpt_step?.toLocaleString?.() ?? res.ckpt_step}`)
              } catch (e: any) {
                alert(`Reload failed: ${e.message}`)
              }
            }}
            title="Bind inference to the selected run's best_val checkpoint"
          >
            ↻ Reload
          </button>
          <span style={{ height: 12, width: 1, background: 'rgba(255,255,255,.05)' }} />
          <span style={{ fontSize: 8, color: '#3f3f46', fontFamily: "'JetBrains Mono',monospace" }}>{ts}</span>
        </div>
      </div>

      <div className="cgpt-content">
        <div className="cgpt-content-inner">
          {activeTab === 'training'    && <TrainingPage />}
          {activeTab === 'live'        && <LivePage />}
          {activeTab === 'calibration' && <CalibrationPage onUseInBacktest={sendToBacktest} />}
          {activeTab === 'backtest'    && <BacktestPage seed={backtestSeed} />}
          {activeTab === 'regimes'     && <RegimePage />}
        </div>
      </div>

      <div className="cgpt-statusbar">
        <div className="sitem">
          <PulseDot size={4} />
          <span>BTC · USDT</span>
        </div>
        <div className="sitem">{nParamsStr}</div>
        <div className="sitem">device {status?.device ?? '—'}</div>
        <div className="sitem" style={{ marginLeft: 'auto' }}>auto-refresh 5s</div>
        <div className="sitem">
          {trainingStatus?.available
            ? `Training step ${(trainingStatus.step ?? 0).toLocaleString()}`
            : 'no active training'}
        </div>
      </div>
    </>
  )
}
