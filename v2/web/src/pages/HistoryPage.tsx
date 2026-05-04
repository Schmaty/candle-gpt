import { useEffect, useState } from 'react'
import { fetchHistory } from '../api'

interface HistoryItem {
  idx: number
  pred_ret: number
  true_ret: number
  correct: boolean
  confidence: number
  regime: number
}

export function HistoryPage() {
  const [windows, setWindows] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchHistory(500)
      .then(d => setWindows(d.windows ?? []))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const ret2pct = (r: number) => `${r >= 0 ? '+' : ''}${(r * 100).toFixed(3)}%`

  if (loading) return <div style={{ color: 'var(--fg-dim)', padding: 24 }}>Loading history…</div>
  if (error) return <div style={{ color: 'var(--red)', padding: 24 }}>{error}</div>

  if (windows.length === 0) {
    return (
      <div className="card" style={{ maxWidth: 480, marginTop: 32 }}>
        <div style={{ color: 'var(--fg-dim)', fontSize: 13 }}>
          No history available. Run evaluation after training to populate this view.
        </div>
      </div>
    )
  }

  const accuracy = windows.filter(w => w.correct).length / windows.length

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <div className="metric">
          <div className="label">Samples</div>
          <div className="value">{windows.length.toLocaleString()}</div>
        </div>
        <div className="metric">
          <div className="label">Accuracy</div>
          <div className="value" style={{ color: accuracy > 0.5 ? 'var(--green)' : 'var(--red)' }}>
            {(accuracy * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ maxHeight: 520, overflowY: 'auto' }}>
          <table>
            <thead style={{ position: 'sticky', top: 0, background: 'var(--bg-surface)', zIndex: 1 }}>
              <tr>
                <th>#</th>
                <th>Predicted</th>
                <th>Actual</th>
                <th>Match</th>
                <th>Regime</th>
              </tr>
            </thead>
            <tbody>
              {windows.map(w => (
                <tr key={w.idx} style={{ background: w.correct ? 'rgba(0,212,170,0.03)' : 'rgba(240,82,82,0.03)' }}>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--fg-dim)', fontSize: 12 }}>{w.idx}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: w.pred_ret >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {ret2pct(w.pred_ret)}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: w.true_ret >= 0 ? 'var(--green)' : 'var(--red)' }}>
                    {ret2pct(w.true_ret)}
                  </td>
                  <td style={{ color: w.correct ? 'var(--green)' : 'var(--red)', fontSize: 12 }}>
                    {w.correct ? '✓' : '✗'}
                  </td>
                  <td style={{ color: 'var(--fg-dim)', fontSize: 12 }}>{w.regime}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
