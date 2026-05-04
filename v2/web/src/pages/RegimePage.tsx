import { useEffect, useState } from 'react'
import { fetchRegimes } from '../api'

interface Regime {
  id: number
  name: string
  accuracy: number
  n: number
}

const REGIME_COLORS: Record<number, string> = {
  '-1': '#8492a6',
  0: '#4a90e2',
  1: '#00d4aa',
  2: '#f5a623',
} as any

export function RegimePage() {
  const [regimes, setRegimes] = useState<Regime[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchRegimes()
      .then(d => setRegimes(d.regimes ?? []))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ color: 'var(--fg-dim)', padding: 24 }}>Loading regime data…</div>
  if (error) return <div style={{ color: 'var(--red)', padding: 24 }}>{error}</div>

  if (regimes.length === 0) {
    return (
      <div className="card" style={{ maxWidth: 480, marginTop: 32 }}>
        <div style={{ color: 'var(--fg-dim)', fontSize: 13 }}>
          No regime data available. Run evaluation after training.
        </div>
      </div>
    )
  }

  const totalN = regimes.reduce((s, r) => s + r.n, 0)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Metric cards */}
      <div className="metric-row">
        {regimes.map(r => (
          <div key={r.id} className="metric">
            <div className="label">{r.name}</div>
            <div className="value" style={{ color: REGIME_COLORS[r.id] ?? 'var(--accent)' }}>
              {(r.accuracy * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>

      {/* Table */}
      <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
        <table>
          <thead>
            <tr>
              <th>Regime</th>
              <th>ID</th>
              <th>Accuracy</th>
              <th>Bars</th>
              <th>Share</th>
              <th>Accuracy bar</th>
            </tr>
          </thead>
          <tbody>
            {regimes.map(r => {
              const color = REGIME_COLORS[r.id] ?? 'var(--accent)'
              const share = totalN > 0 ? r.n / totalN : 0
              return (
                <tr key={r.id}>
                  <td style={{ color, fontWeight: 500 }}>{r.name}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--fg-dim)', fontSize: 12 }}>{r.id}</td>
                  <td style={{ fontFamily: 'var(--font-mono)', color }}>
                    {(r.accuracy * 100).toFixed(2)}%
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--fg-dim)', fontSize: 12 }}>
                    {r.n.toLocaleString()}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--fg-dim)', fontSize: 12 }}>
                    {(share * 100).toFixed(1)}%
                  </td>
                  <td style={{ minWidth: 160 }}>
                    <div style={{ height: 8, background: 'var(--bg-elevated)', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${(r.accuracy * 100).toFixed(1)}%`, background: color, borderRadius: 4 }} />
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
