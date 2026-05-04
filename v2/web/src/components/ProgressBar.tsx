interface ProgressBarProps {
  frac: number   // 0..1
  label?: string
  color?: string
}

export function ProgressBar({ frac, label, color = 'var(--accent)' }: ProgressBarProps) {
  const pct = Math.min(100, Math.max(0, frac * 100))
  return (
    <div>
      <div style={{ height: 8, background: 'var(--bg-elevated)', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 4, transition: 'width 0.5s ease' }} />
      </div>
      {label && <div style={{ fontSize: 11, color: 'var(--fg-dim)', marginTop: 4 }}>{label}</div>}
    </div>
  )
}
