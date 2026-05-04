const BASE = '/api/v2'

export async function fetchStatus() {
  const r = await fetch(`${BASE}/status`)
  if (!r.ok) throw new Error(`status: ${r.status}`)
  return r.json()
}
export async function fetchCandles(limit = 300) {
  const r = await fetch(`${BASE}/candles?limit=${limit}`)
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
