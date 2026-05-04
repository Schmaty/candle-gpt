interface SpecPanelProps {
  title: string
  items: { label: string; value: string | number | null | undefined }[]
}

export function SpecPanel({ title, items }: SpecPanelProps) {
  return (
    <div className="card">
      <div style={{ fontSize: 11, color: 'var(--fg-dim)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 10 }}>{title}</div>
      <table>
        <tbody>
          {items.map(({ label, value }) => (
            <tr key={label}>
              <td style={{ color: 'var(--fg-dim)', width: '50%', paddingLeft: 0 }}>{label}</td>
              <td style={{ fontFamily: 'var(--font-mono)', paddingLeft: 0, color: 'var(--fg)' }}>{value ?? '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
