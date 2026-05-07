import { useState, useEffect } from 'react'
import { TabBar } from './components/TabBar'
import { TrainingPage } from './pages/TrainingPage'
import { LivePage } from './pages/LivePage'
import { HistoryPage } from './pages/HistoryPage'
import { CalibrationPage } from './pages/CalibrationPage'
import { RegimePage } from './pages/RegimePage'
import { EquityPage } from './pages/EquityPage'
import { BacktestPage, type BacktestSeed } from './pages/BacktestPage'
import { fetchStatus, getTrainingStatus, reloadModel } from './api'

const TABS = [
  { id: 'training',    label: 'Training' },
  { id: 'live',        label: 'Live' },
  { id: 'history',     label: 'History' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'backtest',    label: 'Backtest' },
  { id: 'regimes',     label: 'Regimes' },
  { id: 'equity',      label: 'Equity' },
]

function shortRun(id?: string | null) {
  if (!id) return '—'
  return id.length > 24 ? `${id.slice(0, 10)}…${id.slice(-8)}` : id
}

function fmtParams(n?: number | null) {
  if (n == null) return '—'
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n.toLocaleString()
}

export default function App() {
  const [activeTab, setActiveTab] = useState('training')
  const [status, setStatus] = useState<any>(null)
  const [trainingStatus, setTrainingStatus] = useState<any>(null)
  const [backtestSeed, setBacktestSeed] = useState<BacktestSeed | null>(null)

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

  const sendToBacktest = (seed: BacktestSeed) => {
    setBacktestSeed(seed)
    setActiveTab('backtest')
  }

  const inferenceText = status?.model_loaded
    ? `Inference ${shortRun(status.run_id)} @ step ${status.ckpt_step?.toLocaleString?.() ?? '—'} · ${fmtParams(status.n_params)} · ${status.device}`
    : 'Inference: no model loaded'
  const trainingText = trainingStatus?.available
    ? `Training ${shortRun(trainingStatus.run_id)} · ${trainingStatus.state ?? '—'} · step ${(trainingStatus.step ?? 0).toLocaleString()} · loss ${trainingStatus.train_loss != null ? trainingStatus.train_loss.toFixed(4) : '—'}`
    : 'Training: no active run'
  const mismatch = status?.run_id && trainingStatus?.run_id && status.run_id !== trainingStatus.run_id

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', gap: '12px', borderBottom: '1px solid var(--border)', background: 'var(--bg-surface)' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent)', fontSize: 16 }}>candle-gpt v2</span>
        <div style={{ color: 'var(--fg-dim)', fontSize: 12, display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
          <span>{inferenceText}</span>
          <span>
            {trainingText}
            {mismatch && <span style={{ color: '#f5a623' }}> · training is ahead of loaded inference checkpoint</span>}
          </span>
        </div>
        <button
          onClick={async () => {
            try {
              const res = await reloadModel()
              await refreshHeader()
              alert(`Reloaded inference: run ${res.run_id} step ${res.ckpt_step?.toLocaleString?.() ?? res.ckpt_step}`)
            } catch (e: any) {
              alert(`Reload failed: ${e.message}`)
            }
          }}
          style={{ marginLeft: 'auto', height: 28, fontSize: 12 }}
          title="Bind inference/backtest to the selected run's best_val checkpoint when one exists"
        >
          ↻ Reload inference
        </button>
      </header>
      <TabBar tabs={TABS} active={activeTab} onSelect={setActiveTab} />
      <main style={{ flex: 1, overflow: 'auto', padding: '16px' }}>
        {activeTab === 'training'    && <TrainingPage />}
        {activeTab === 'live'        && <LivePage />}
        {activeTab === 'history'     && <HistoryPage />}
        {activeTab === 'calibration' && <CalibrationPage onUseInBacktest={sendToBacktest} />}
        {activeTab === 'backtest'    && <BacktestPage seed={backtestSeed} />}
        {activeTab === 'regimes'     && <RegimePage />}
        {activeTab === 'equity'      && <EquityPage />}
      </main>
    </div>
  )
}
