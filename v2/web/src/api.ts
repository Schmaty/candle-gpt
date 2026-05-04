const BASE = '/api/v2'

export async function fetchStatus() {
  const r = await fetch(`${BASE}/status`)
  if (!r.ok) throw new Error(`status: ${r.status}`)
  return r.json()
}
export async function fetchCandles(limit = 300, interval = '1m') {
  const r = await fetch(`${BASE}/candles?limit=${limit}&interval=${encodeURIComponent(interval)}`)
  if (!r.ok) throw new Error(`candles: ${r.status}`)
  return r.json()
}
export async function fetchHistory(limit = 500) {
  const r = await fetch(`${BASE}/history?limit=${limit}`)
  if (!r.ok) throw new Error(`history: ${r.status}`)
  return r.json()
}
export async function fetchCalibration() {
  const r = await fetch(`${BASE}/calibration`)
  if (!r.ok) throw new Error(`calibration: ${r.status}`)
  return r.json()
}
export async function fetchRegimes() {
  const r = await fetch(`${BASE}/regimes`)
  if (!r.ok) throw new Error(`regimes: ${r.status}`)
  return r.json()
}
export async function fetchEquity() {
  const r = await fetch(`${BASE}/equity`)
  if (!r.ok) throw new Error(`equity: ${r.status}`)
  return r.json()
}
export async function getTrainingStatus() {
  const r = await fetch(`${BASE}/training/status`)
  if (!r.ok) throw new Error(`training/status: ${r.status}`)
  return r.json()
}
export async function getTrainingEvents(afterTs: number | null = null) {
  const url = afterTs !== null ? `${BASE}/training/events?after=${afterTs}` : `${BASE}/training/events`
  const r = await fetch(url)
  if (!r.ok) throw new Error(`training/events: ${r.status}`)
  return r.json()
}
export async function runSweep(temperatures: number[], horizons: number[], nSamples: number) {
  const t = temperatures.join(',')
  const h = horizons.join(',')
  const r = await fetch(`${BASE}/calibration/sweep?temperatures=${t}&horizons=${h}&n_samples=${nSamples}`)
  if (!r.ok) throw new Error(`sweep: ${r.status} ${await r.text()}`)
  return r.json()
}
export async function getSystemStats() {
  const r = await fetch(`${BASE}/system/stats`)
  if (!r.ok) throw new Error(`system/stats: ${r.status}`)
  return r.json()
}
export async function getEvalHistory(runId?: string) {
  const url = runId ? `${BASE}/eval_history?run_id=${encodeURIComponent(runId)}` : `${BASE}/eval_history`
  const r = await fetch(url)
  if (!r.ok) throw new Error(`eval_history: ${r.status}`)
  return r.json()
}
export async function runBacktest(opts: {
  temperature: number; horizon: number; z_threshold: number;
  start_frac?: number; end_frac?: number; fee_bps?: number;
}) {
  const params = new URLSearchParams({
    temperature: String(opts.temperature),
    horizon: String(opts.horizon),
    z_threshold: String(opts.z_threshold),
    start_frac: String(opts.start_frac ?? 0),
    end_frac: String(opts.end_frac ?? 1),
    fee_bps: String(opts.fee_bps ?? 1.0),
  })
  const r = await fetch(`${BASE}/backtest?${params}`)
  if (!r.ok) throw new Error(`backtest: ${r.status} ${await r.text()}`)
  return r.json()
}
