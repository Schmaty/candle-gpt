import { useEffect, useMemo, useState } from 'react'
import { fetchCalibration, fetchHistory, runSweep } from '../api'
import { Panel, SLabel, Divider, MBox, Pill } from '../components/dash'
import type { BacktestSeed } from './BacktestPage'

interface Bucket {
  lo: number
  hi: number
  conf: number
  acc: number
  frac: number
}

interface SweepRow {
  temperature: number
  horizon: number
  dir_acc: number | null
  n_valid: number
  ece: number
  mean_conf: number
  top1_acc: number
}

// Reliability chart — bucket bars vs identity line.
function RelChart({ buckets, W = 300, H = 220 }: { buckets: Bucket[]; W?: number; H?: number }) {
  const pad = { t: 16, r: 16, b: 32, l: 32 }
  const IW = W - pad.l - pad.r
  const IH = H - pad.t - pad.b
  const tks = [0, 0.25, 0.5, 0.75, 1]
  return (
    <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`}>
      {tks.map((t, i) => (
        <g key={i}>
          <line
            x1={pad.l}
            x2={pad.l + IW}
            y1={pad.t + IH - t * IH}
            y2={pad.t + IH - t * IH}
            stroke="rgba(255,255,255,.04)"
            strokeWidth="1"
          />
          <line
            x1={pad.l + t * IW}
            x2={pad.l + t * IW}
            y1={pad.t}
            y2={pad.t + IH}
            stroke="rgba(255,255,255,.04)"
            strokeWidth="1"
          />
          <text
            x={pad.l - 4}
            y={pad.t + IH - t * IH + 3}
            fontSize="8"
            fill="#3f3f46"
            fontFamily="'JetBrains Mono',monospace"
            textAnchor="end"
          >
            {t.toFixed(1)}
          </text>
          <text
            x={pad.l + t * IW}
            y={pad.t + IH + 14}
            fontSize="8"
            fill="#3f3f46"
            fontFamily="'JetBrains Mono',monospace"
            textAnchor="middle"
          >
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      <line
        x1={pad.l}
        x2={pad.l + IW}
        y1={pad.t + IH}
        y2={pad.t}
        stroke="rgba(255,255,255,.1)"
        strokeWidth="1"
        strokeDasharray="4 3"
      />
      {buckets.map((b, i) => {
        const bw = Math.max(2, IW / buckets.length - 3)
        const bx = pad.l + b.conf * IW - bw / 2
        const bh = b.acc * IH
        return (
          <g key={i}>
            <rect
              x={bx}
              y={pad.t + IH - bh}
              width={bw}
              height={bh}
              fill="rgba(56,189,248,.2)"
              stroke="#38bdf8"
              strokeWidth=".75"
              rx="1"
            />
            <circle cx={pad.l + b.conf * IW} cy={pad.t + IH - b.acc * IH} r="2.5" fill="#38bdf8" />
          </g>
        )
      })}
    </svg>
  )
}

export function CalibrationPage({ onUseInBacktest }: { onUseInBacktest?: (seed: BacktestSeed) => void }) {
  const [data, setData] = useState<{ buckets: Bucket[]; ece: number } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Sweep
  const [tempsStr, setTempsStr] = useState('0.5, 0.8, 1.0, 1.5, 2.0')
  const [horizonsStr, setHorizonsStr] = useState('1, 3, 5, 10, 20, 30')
  const [nSamples, setNSamples] = useState(150)
  const [sweepResults, setSweepResults] = useState<SweepRow[] | null>(null)
  const [sweepBest, setSweepBest] = useState<SweepRow | null>(null)
  const [running, setRunning] = useState(false)
  const [sweepError, setSweepError] = useState<string | null>(null)

  const [sortCol, setSortCol] = useState<keyof SweepRow>('dir_acc')
  const [sortDir, setSortDir] = useState<-1 | 1>(-1)

  useEffect(() => {
    fetchCalibration()
      .then(d => setData(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const doSweep = async () => {
    setRunning(true)
    setSweepError(null)
    try {
      const T_list = tempsStr.split(',').map(s => parseFloat(s.trim())).filter(x => Number.isFinite(x) && x > 0)
      const H_list = horizonsStr.split(',').map(s => parseInt(s.trim(), 10)).filter(x => Number.isFinite(x) && x > 0)
      if (!T_list.length || !H_list.length) {
        throw new Error('Need at least one positive temperature and horizon.')
      }
      const res = await runSweep(T_list, H_list, nSamples)
      setSweepResults(res.results as SweepRow[])
      setSweepBest((res.best as SweepRow) ?? null)
    } catch (e: any) {
      setSweepError(e.message)
    } finally {
      setRunning(false)
    }
  }

  const tog = (col: keyof SweepRow) => {
    if (sortCol === col) setSortDir(d => (d === -1 ? 1 : -1))
    else { setSortCol(col); setSortDir(-1) }
  }

  if (loading) return <div style={{ color: '#52525b', padding: 24, fontSize: 11 }}>Loading calibration…</div>
  if (error) return <div style={{ color: '#fb7185', padding: 24, fontSize: 11 }}>{error}</div>

  const sortedRows = sweepResults
    ? [...sweepResults].sort((a, b) => {
        const av = a[sortCol]
        const bv = b[sortCol]
        if (av == null) return 1
        if (bv == null) return -1
        return sortDir * ((av as number) - (bv as number))
      })
    : null

  const th: React.CSSProperties = {
    fontSize: 8,
    letterSpacing: '.12em',
    textTransform: 'uppercase',
    padding: '7px 10px',
    cursor: 'pointer',
    userSelect: 'none',
    whiteSpace: 'nowrap',
    textAlign: 'left',
  }
  const td: React.CSSProperties = {
    padding: '6px 10px',
    fontSize: 9,
    fontFamily: "'JetBrains Mono',monospace",
    borderBottom: '1px solid rgba(255,255,255,.025)',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
    <div className="cgpt-cal-layout">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <Panel style={{ padding: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
            <SLabel>Reliability</SLabel>
            {data && <Pill>ECE {data.ece.toFixed(4)}</Pill>}
          </div>
          {data && data.buckets.length > 0 ? (
            <>
              <RelChart buckets={data.buckets} W={264} H={200} />
              <Divider />
              <div style={{ fontSize: 8, color: '#2a2a2e', letterSpacing: '.08em', lineHeight: 1.8 }}>
                Predicted probability vs realized frequency. Bars above diagonal = underconfident.
              </div>
            </>
          ) : (
            <div style={{ fontSize: 9, color: '#3f3f46' }}>No calibration data — run evaluation.</div>
          )}
        </Panel>
        <Panel style={{ padding: 16 }}>
          <SLabel>Run Sweep</SLabel>
          {([
            ['Temperatures', tempsStr, setTempsStr],
            ['Horizons', horizonsStr, setHorizonsStr],
          ] as Array<[string, string, (s: string) => void]>).map(([lbl, val, set]) => (
            <div key={lbl} style={{ marginBottom: 10 }}>
              <div
                style={{
                  fontSize: 8,
                  color: '#52525b',
                  letterSpacing: '.12em',
                  textTransform: 'uppercase',
                  marginBottom: 4,
                }}
              >
                {lbl}
              </div>
              <input type="text" value={val} onChange={e => set(e.target.value)} />
            </div>
          ))}
          <div style={{ marginBottom: 10 }}>
            <div
              style={{
                fontSize: 8,
                color: '#52525b',
                letterSpacing: '.12em',
                textTransform: 'uppercase',
                marginBottom: 4,
              }}
            >
              Samples per pair
            </div>
            <input
              type="number"
              value={nSamples}
              min={20}
              max={1000}
              onChange={e => setNSamples(parseInt(e.target.value, 10) || 150)}
            />
          </div>
          <button
            onClick={doSweep}
            disabled={running}
            style={{
              width: '100%',
              padding: '9px 0',
              borderRadius: 10,
              marginTop: 4,
              fontSize: 9,
              fontWeight: 700,
              letterSpacing: '.15em',
              textTransform: 'uppercase',
              background: running ? 'rgba(56,189,248,.04)' : 'rgba(56,189,248,.08)',
              color: running ? '#52525b' : '#7dd3fc',
              border: '1px solid rgba(56,189,248,.15)',
              cursor: running ? 'not-allowed' : 'pointer',
            }}
          >
            {running ? '— sweeping —' : '→ run sweep'}
          </button>
          {sweepError && (
            <div style={{ color: '#fb7185', fontSize: 9, marginTop: 8 }}>{sweepError}</div>
          )}
        </Panel>
      </div>

      <Panel style={{ padding: 16, overflow: 'hidden' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <SLabel>Sweep Results</SLabel>
          {sweepBest && (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <Pill color="#fbbf24">
                best: T={sweepBest.temperature} H={sweepBest.horizon}
                {sweepBest.dir_acc != null ? ` · ${(sweepBest.dir_acc * 100).toFixed(1)}%` : ''}
              </Pill>
              {onUseInBacktest && (
                <button
                  onClick={() => onUseInBacktest({ temperature: sweepBest.temperature, horizon: sweepBest.horizon })}
                  style={{
                    padding: '4px 10px',
                    borderRadius: 7,
                    fontSize: 9,
                    fontWeight: 700,
                    letterSpacing: '.1em',
                    textTransform: 'uppercase',
                    background: 'transparent',
                    color: '#7dd3fc',
                    border: '1px solid rgba(125,211,252,.25)',
                    cursor: 'pointer',
                  }}
                >
                  → use in backtest
                </button>
              )}
            </div>
          )}
        </div>
        {!sortedRows && (
          <div className="float-hint" style={{ fontSize: 9, color: '#2a2a2e', textAlign: 'center', padding: 24 }}>
            No sweep run yet — configure and click run.
          </div>
        )}
        {sortedRows && (
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,.06)' }}>
                  {([
                    ['temperature', 'T'],
                    ['horizon', 'H'],
                    ['n_valid', 'N'],
                    ['dir_acc', 'DIR ACC'],
                    ['mean_conf', 'CONF'],
                    ['top1_acc', 'TOP-1'],
                    ['ece', 'ECE'],
                  ] as Array<[keyof SweepRow, string]>).map(([col, lbl]) => (
                    <th
                      key={col as string}
                      style={{ ...th, color: sortCol === col ? '#7dd3fc' : '#3f3f46' }}
                      onClick={() => tog(col)}
                    >
                      {lbl}
                      {sortCol === col ? (sortDir < 0 ? ' ↓' : ' ↑') : ''}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedRows.map((row, i) => {
                  const isBest =
                    sweepBest && row.temperature === sweepBest.temperature && row.horizon === sweepBest.horizon
                  return (
                    <tr
                      key={i}
                      onClick={() =>
                        onUseInBacktest && onUseInBacktest({ temperature: row.temperature, horizon: row.horizon })
                      }
                      style={{
                        background: isBest ? 'rgba(251,191,36,.04)' : 'transparent',
                        borderLeft: isBest ? '2px solid rgba(251,191,36,.35)' : '2px solid transparent',
                        cursor: onUseInBacktest ? 'pointer' : 'default',
                      }}
                    >
                      <td style={{ ...td, color: '#71717a' }}>{row.temperature}</td>
                      <td style={{ ...td, color: '#71717a' }}>{row.horizon}</td>
                      <td style={{ ...td, color: '#52525b' }}>{row.n_valid}</td>
                      <td
                        style={{
                          ...td,
                          color: row.dir_acc != null && row.dir_acc > 0.58 ? '#4ade80' : '#e8e6e1',
                          fontWeight: 600,
                        }}
                      >
                        {row.dir_acc != null ? `${(row.dir_acc * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td style={{ ...td, color: '#71717a' }}>{(row.mean_conf * 100).toFixed(1)}%</td>
                      <td style={{ ...td, color: '#71717a' }}>{(row.top1_acc * 100).toFixed(1)}%</td>
                      <td style={{ ...td, color: row.ece > 0.08 ? '#fb7185' : '#52525b' }}>{row.ece.toFixed(4)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
    <PredictionHistorySection />
    </div>
  )
}

// ---------- Prediction history (per-window log, ported from old History tab) -----

interface HistoryItem {
  idx: number
  pred_ret: number
  true_ret: number
  correct: boolean
  confidence: number
  regime: number
}

type HSortCol = keyof HistoryItem

function PredictionHistorySection() {
  const [windows, setWindows] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState(false)

  const [sortCol, setSortCol] = useState<HSortCol>('idx')
  const [sortDir, setSortDir] = useState<-1 | 1>(-1)
  const [page, setPage] = useState(0)
  const PAGE = 40

  useEffect(() => {
    fetchHistory(500)
      .then(d => setWindows(d.windows ?? []))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const sorted = useMemo(
    () =>
      [...windows].sort((a, b) => {
        const av = a[sortCol]
        const bv = b[sortCol]
        const an = typeof av === 'boolean' ? Number(av) : (av as number)
        const bn = typeof bv === 'boolean' ? Number(bv) : (bv as number)
        return sortDir * (an - bn)
      }),
    [windows, sortCol, sortDir],
  )
  const totalP = Math.max(1, Math.ceil(sorted.length / PAGE))
  const pageRows = sorted.slice(page * PAGE, (page + 1) * PAGE)
  const tog = (col: HSortCol) => {
    if (sortCol === col) setSortDir(d => (d === -1 ? 1 : -1))
    else { setSortCol(col); setSortDir(-1); setPage(0) }
  }

  if (loading) {
    return (
      <Panel style={{ padding: 16 }}>
        <SLabel>Prediction History · loading…</SLabel>
      </Panel>
    )
  }
  if (error) {
    return (
      <Panel style={{ padding: 16 }}>
        <SLabel>Prediction History</SLabel>
        <div style={{ color: '#fb7185', fontSize: 9 }}>{error}</div>
      </Panel>
    )
  }
  if (!windows.length) {
    return (
      <Panel style={{ padding: 16 }}>
        <SLabel>Prediction History</SLabel>
        <div style={{ color: '#3f3f46', fontSize: 10 }}>
          Run evaluation after training to populate this view.
        </div>
      </Panel>
    )
  }

  const hitRate = windows.filter(w => w.correct).length / windows.length
  const meanConf = windows.reduce((s, w) => s + w.confidence, 0) / windows.length
  const meanPred = windows.reduce((s, w) => s + Math.abs(w.pred_ret), 0) / windows.length

  const th: React.CSSProperties = {
    fontSize: 8, letterSpacing: '.12em', textTransform: 'uppercase',
    padding: '8px 12px', cursor: 'pointer', userSelect: 'none', whiteSpace: 'nowrap', textAlign: 'left',
  }
  const td: React.CSSProperties = {
    padding: '6px 12px', fontSize: 9,
    fontFamily: "'JetBrains Mono',monospace",
    borderBottom: '1px solid rgba(255,255,255,.022)',
  }

  return (
    <Panel style={{ padding: 16, overflow: 'hidden' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, gap: 12, flexWrap: 'wrap' }}>
        <SLabel style={{ marginBottom: 0 }}>Prediction History · per window</SLabel>
        <button
          onClick={() => setOpen(o => !o)}
          className="cgpt-ghost-btn"
        >
          {open ? '▾ hide log' : '▸ show log'}
        </button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,minmax(0,1fr))', gap: 14, marginBottom: open ? 14 : 0 }}>
        <MBox label="Rows" value={windows.length.toLocaleString()} />
        <MBox
          label="Hit Rate"
          value={(hitRate * 100).toFixed(1) + '%'}
          accent={hitRate > 0.5 ? '#4ade80' : '#fb7185'}
          meter={hitRate}
        />
        <MBox label="Mean Conf" value={(meanConf * 100).toFixed(1) + '%'} meter={meanConf} />
        <MBox label="Mean |Pred|" value={(meanPred * 10000).toFixed(1) + ' bps'} />
      </div>
      {open && (
        <>
          <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: 460, borderTop: '1px solid rgba(255,255,255,.04)' }}>
            <table>
              <thead style={{ position: 'sticky', top: 0, background: 'rgba(12,12,14,.98)', backdropFilter: 'blur(8px)', zIndex: 2 }}>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,.06)' }}>
                  {([
                    ['idx', '#'],
                    ['pred_ret', 'Pred bps'],
                    ['true_ret', 'Real bps'],
                    ['correct', 'Hit'],
                    ['confidence', 'Conf'],
                    ['regime', 'Regime'],
                  ] as Array<[HSortCol, string]>).map(([col, lbl]) => (
                    <th
                      key={col as string}
                      style={{ ...th, color: sortCol === col ? '#7dd3fc' : '#3f3f46' }}
                      onClick={() => tog(col)}
                    >
                      {lbl}
                      {sortCol === col ? (sortDir < 0 ? ' ↓' : ' ↑') : ''}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pageRows.map(row => (
                  <tr
                    key={row.idx}
                    onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,.02)')}
                    onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                    style={{ transition: 'background 80ms' }}
                  >
                    <td style={{ ...td, color: '#2a2a2e' }}>{row.idx}</td>
                    <td style={{ ...td, color: row.pred_ret > 0 ? '#4ade80' : '#fb7185' }}>
                      {(row.pred_ret * 10000).toFixed(1)}
                    </td>
                    <td style={{ ...td, color: row.true_ret > 0 ? '#4ade80' : '#fb7185' }}>
                      {(row.true_ret * 10000).toFixed(1)}
                    </td>
                    <td style={td}>
                      <span
                        style={{
                          display: 'inline-block', width: 5, height: 5,
                          borderRadius: '50%',
                          background: row.correct ? '#4ade80' : '#fb7185',
                        }}
                      />
                    </td>
                    <td style={{ ...td, color: '#52525b' }}>{(row.confidence * 100).toFixed(1)}%</td>
                    <td style={{ ...td, color: '#52525b' }}>{row.regime}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div
            style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '9px 4px 0', marginTop: 4, borderTop: '1px solid rgba(255,255,255,.04)',
            }}
          >
            <span style={{ fontSize: 8, color: '#2a2a2e', letterSpacing: '.1em' }}>
              {page * PAGE + 1}–{Math.min((page + 1) * PAGE, sorted.length)} of {sorted.length}
            </span>
            <div style={{ display: 'flex', gap: 6 }}>
              {([
                ['← prev', page > 0, () => setPage(p => p - 1)],
                ['next →', page < totalP - 1, () => setPage(p => p + 1)],
              ] as Array<[string, boolean, () => void]>).map(([lbl, en, fn]) => (
                <button
                  key={lbl}
                  onClick={fn}
                  disabled={!en}
                  style={{
                    padding: '3px 9px', borderRadius: 6,
                    fontSize: 8, fontWeight: 600, letterSpacing: '.1em',
                    background: 'transparent',
                    border: '1px solid rgba(255,255,255,.06)',
                    color: en ? '#52525b' : '#2a2a2e',
                    cursor: en ? 'pointer' : 'not-allowed',
                  }}
                >
                  {lbl}
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </Panel>
  )
}
