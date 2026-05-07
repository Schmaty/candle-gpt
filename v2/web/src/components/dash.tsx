// Shared design primitives ported from /tmp/candle-dash-new/components/shared.jsx
// and CandleGPT.html. All components are presentation-only.

import type { CSSProperties, ReactNode } from 'react'

export const Panel = ({ children, style }: { children: ReactNode; style?: CSSProperties }) => (
  <div
    style={{
      background: 'linear-gradient(180deg,rgba(24,24,27,.78) 0%,rgba(15,15,17,.78) 100%)',
      backdropFilter: 'blur(12px)',
      border: '1px solid rgba(255,255,255,.06)',
      boxShadow: '0 1px 0 rgba(255,255,255,.04) inset,0 20px 60px -20px rgba(0,0,0,.6)',
      borderRadius: 16,
      ...style,
    }}
  >
    {children}
  </div>
)

export const SLabel = ({ children, style }: { children: ReactNode; style?: CSSProperties }) => (
  <div
    style={{
      fontSize: 9,
      fontWeight: 700,
      letterSpacing: '.2em',
      textTransform: 'uppercase',
      color: '#52525b',
      marginBottom: 10,
      ...style,
    }}
  >
    {children}
  </div>
)

export const Divider = ({ style }: { style?: CSSProperties }) => (
  <div
    style={{
      height: 1,
      background: 'linear-gradient(90deg,transparent,rgba(255,255,255,.07),transparent)',
      margin: '12px 0',
      ...style,
    }}
  />
)

export const Pill = ({
  children,
  color = '#52525b',
  style,
}: { children: ReactNode; color?: string; style?: CSSProperties }) => (
  <span
    style={{
      display: 'inline-flex',
      alignItems: 'center',
      gap: 5,
      padding: '3px 9px',
      borderRadius: 999,
      fontSize: 9,
      fontWeight: 700,
      letterSpacing: '.08em',
      textTransform: 'uppercase',
      border: '1px solid rgba(255,255,255,.07)',
      background: 'rgba(255,255,255,.02)',
      color,
      whiteSpace: 'nowrap',
      ...style,
    }}
  >
    {children}
  </span>
)

export const PulseDot = ({ color = '#4ade80', size = 5 }: { color?: string; size?: number }) => (
  <span
    className="pulse-dot"
    style={{
      display: 'inline-block',
      width: size,
      height: size,
      borderRadius: '50%',
      background: color,
      boxShadow: `0 0 7px ${color}`,
      flexShrink: 0,
    }}
  />
)

export function MBox({
  label,
  value,
  accent,
  meter,
  sub,
}: {
  label: string
  value: ReactNode
  accent?: string
  meter?: number | null
  sub?: ReactNode
}) {
  return (
    <div>
      <div
        style={{
          fontSize: 8,
          letterSpacing: '.18em',
          textTransform: 'uppercase',
          color: '#3f3f46',
          marginBottom: 3,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: "'Fraunces',serif",
          fontSize: 20,
          fontWeight: 600,
          letterSpacing: '-.025em',
          color: accent || '#e8e6e1',
          lineHeight: 1,
        }}
      >
        {value}
      </div>
      {sub && <div style={{ fontSize: 9, color: '#52525b', marginTop: 2 }}>{sub}</div>}
      {meter != null && (
        <div
          style={{
            height: 3,
            background: 'rgba(255,255,255,.04)',
            borderRadius: 999,
            overflow: 'hidden',
            marginTop: 5,
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${Math.max(0, Math.min(100, meter * 100))}%`,
              background: 'linear-gradient(90deg,#38bdf8,#818cf8)',
              borderRadius: 999,
              transition: 'width 400ms',
            }}
          />
        </div>
      )}
    </div>
  )
}

export function TokRow({ k, v }: { k: string; v: string }) {
  const c = v === 'UP' ? '#4ade80' : v === 'DOWN' ? '#fb7185' : '#e8e6e1'
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '5px 0',
        borderBottom: '1px solid rgba(255,255,255,.035)',
      }}
    >
      <span style={{ fontSize: 9, color: '#52525b', letterSpacing: '.06em' }}>{k}</span>
      <span style={{ fontSize: 9, fontWeight: 700, color: c, letterSpacing: '.08em' }}>{v}</span>
    </div>
  )
}
