import { useState, useEffect } from 'react'
import { TabBar } from './components/TabBar'
import { TrainingPage } from './pages/TrainingPage'
import { LivePage } from './pages/LivePage'
import { HistoryPage } from './pages/HistoryPage'
import { CalibrationPage } from './pages/CalibrationPage'
import { RegimePage } from './pages/RegimePage'
import { EquityPage } from './pages/EquityPage'
import { BacktestPage, type BacktestSeed } from './pages/BacktestPage'
import { fetchStatus } from './api'

const TABS = [
  { id: 'training',    label: 'Training' },
  { id: 'live',        label: 'Live' },
  { id: 'history',     label: 'History' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'backtest',    label: 'Backtest' },
  { id: 'regimes',     label: 'Regimes' },
  { id: 'equity',      label: 'Equity' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('training')
  const [status, setStatus] = useState<any>(null)
  const [backtestSeed, setBacktestSeed] = useState<BacktestSeed | null>(null)

  useEffect(() => {
    fetchStatus().then(setStatus).catch(console.error)
  }, [])

  const sendToBacktest = (seed: BacktestSeed) => {
    setBacktestSeed(seed)
    setActiveTab('backtest')
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <header style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', gap: '12px', borderBottom: '1px solid var(--border)', background: 'var(--bg-surface)' }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent)', fontSize: 16 }}>candle-gpt v2</span>
        {status && (
          <span style={{ color: 'var(--fg-dim)', fontSize: 12 }}>
            {status.model_loaded
              ? `model loaded · step ${status.ckpt_step?.toLocaleString()} · ${status.n_params?.toLocaleString()} params · ${status.device}`
              : 'no model loaded'}
          </span>
        )}
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
