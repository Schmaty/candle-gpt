interface Tab { id: string; label: string }
interface TabBarProps { tabs: Tab[]; active: string; onSelect: (id: string) => void }

export function TabBar({ tabs, active, onSelect }: TabBarProps) {
  return (
    <nav style={{ display: 'flex', gap: '2px', padding: '8px 16px', borderBottom: '1px solid var(--border)', background: 'var(--bg-surface)' }}>
      {tabs.map(tab => (
        <button key={tab.id} className={active === tab.id ? 'active' : ''} onClick={() => onSelect(tab.id)} style={{ fontSize: '13px', padding: '6px 16px' }}>
          {tab.label}
        </button>
      ))}
    </nav>
  )
}
