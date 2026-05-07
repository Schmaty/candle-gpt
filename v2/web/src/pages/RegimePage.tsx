import { useEffect, useState } from 'react'
import { fetchRegimes } from '../api'
import { Panel, SLabel } from '../components/dash'

interface Regime {
  id: number
  name: string
  accuracy: number
  n: number
}

// ID-stable colors so each regime keeps the same look across reloads.
const REGIME_PALETTE: Record<string, { color: string; bg: string }> = {
  '-1': { color: '#8492a6', bg: 'rgba(132,146,166,.06)' },
  '0':  { color: '#4ade80', bg: 'rgba(74,222,128,.06)' },
  '1':  { color: '#fb7185', bg: 'rgba(251,113,133,.06)' },
  '2':  { color: '#fbbf24', bg: 'rgba(251,191,36,.06)' },
  '3':  { color: '#c084fc', bg: 'rgba(192,132,252,.06)' },
  '4':  { color: '#7dd3fc', bg: 'rgba(125,211,252,.06)' },
}
function paletteFor(id: number) {
  return REGIME_PALETTE[String(id)] ?? { color: '#7dd3fc', bg: 'rgba(125,211,252,.06)' }
}

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

  if (loading) return <div style={{ color: '#52525b', padding: 24, fontSize: 11 }}>Loading regime data…</div>
  if (error) return <div style={{ color: '#fb7185', padding: 24, fontSize: 11 }}>{error}</div>

  if (!regimes.length) {
    return (
      <Panel style={{ padding: 24, maxWidth: 480 }}>
        <SLabel>No regime data</SLabel>
        <div style={{ color: '#3f3f46', fontSize: 10 }}>Run evaluation after training.</div>
      </Panel>
    )
  }

  const totalN = regimes.reduce((s, r) => s + r.n, 0)
  const bestAcc = Math.max(...regimes.map(r => r.accuracy))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {/* KPI strip */}
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.max(2, regimes.length)},1fr)`, gap: 14 }}>
        {regimes.map(r => {
          const p = paletteFor(r.id)
          const share = totalN > 0 ? r.n / totalN : 0
          return (
            <Panel key={r.id} style={{ padding: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 7 }}>
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: p.color }} />
                <span style={{ fontSize: 8, letterSpacing: '.15em', color: p.color, fontWeight: 700, textTransform: 'uppercase' }}>
                  {r.name}
                </span>
              </div>
              <div
                style={{
                  fontFamily: "'Fraunces',serif",
                  fontSize: 28,
                  fontWeight: 600,
                  color: p.color,
                  letterSpacing: '-.025em',
                }}
              >
                {(share * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: 8, color: '#3f3f46', marginTop: 2 }}>{r.n.toLocaleString()} bars</div>
            </Panel>
          )
        })}
      </div>

      {/* Edge by regime */}
      <Panel style={{ padding: 16 }}>
        <SLabel>Model Edge by Regime</SLabel>
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.max(2, regimes.length)},1fr)`, gap: 14 }}>
          {regimes.map(r => {
            const p = paletteFor(r.id)
            const acc = r.accuracy
            return (
              <div key={r.id}>
                <div
                  style={{
                    fontSize: 8,
                    color: p.color,
                    letterSpacing: '.12em',
                    textTransform: 'uppercase',
                    marginBottom: 5,
                  }}
                >
                  {r.name}
                </div>
                <div
                  style={{
                    fontFamily: "'Fraunces',serif",
                    fontSize: 22,
                    fontWeight: 600,
                    color: acc > 0.55 ? '#4ade80' : '#e8e6e1',
                    marginBottom: 4,
                  }}
                >
                  {(acc * 100).toFixed(1)}%
                </div>
                <div style={{ height: 3, background: 'rgba(255,255,255,.04)', borderRadius: 999 }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${Math.max(0, Math.min(100, acc * 100))}%`,
                      background: p.color,
                      borderRadius: 999,
                      opacity: 0.7,
                    }}
                  />
                </div>
                <div style={{ fontSize: 8, color: '#3f3f46', marginTop: 3 }}>
                  {r.n.toLocaleString()} samples
                  {acc === bestAcc && <span style={{ color: '#fbbf24', marginLeft: 6 }}>· best</span>}
                </div>
              </div>
            )
          })}
        </div>
      </Panel>

      {/* Detail table */}
      <Panel style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: 16, paddingBottom: 0 }}>
          <SLabel>Regime Detail</SLabel>
        </div>
        <table>
          <thead>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,.06)' }}>
              {['Regime', 'ID', 'Accuracy', 'Bars', 'Share', 'Bar'].map(h => (
                <th
                  key={h}
                  style={{
                    fontSize: 8,
                    letterSpacing: '.12em',
                    textTransform: 'uppercase',
                    padding: '8px 12px',
                    color: '#3f3f46',
                    textAlign: 'left',
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {regimes.map(r => {
              const p = paletteFor(r.id)
              const share = totalN > 0 ? r.n / totalN : 0
              return (
                <tr key={r.id}>
                  <td style={{ padding: '6px 12px', fontSize: 10, color: p.color, fontWeight: 600 }}>{r.name}</td>
                  <td style={{ padding: '6px 12px', fontSize: 9, fontFamily: "'JetBrains Mono',monospace", color: '#52525b' }}>
                    {r.id}
                  </td>
                  <td style={{ padding: '6px 12px', fontSize: 9, fontFamily: "'JetBrains Mono',monospace", color: p.color }}>
                    {(r.accuracy * 100).toFixed(2)}%
                  </td>
                  <td style={{ padding: '6px 12px', fontSize: 9, fontFamily: "'JetBrains Mono',monospace", color: '#71717a' }}>
                    {r.n.toLocaleString()}
                  </td>
                  <td style={{ padding: '6px 12px', fontSize: 9, fontFamily: "'JetBrains Mono',monospace", color: '#71717a' }}>
                    {(share * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '6px 12px', minWidth: 160 }}>
                    <div style={{ height: 6, background: 'rgba(255,255,255,.04)', borderRadius: 999, overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${(r.accuracy * 100).toFixed(1)}%`,
                          background: p.color,
                        }}
                      />
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </Panel>
    </div>
  )
}
