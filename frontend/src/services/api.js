const BASE = import.meta.env.VITE_API_BASE_URL ?? ''

/**
 * POST /analyze
 * @param {string} query
 * @param {number} k  number of cases to retrieve (default 5)
 * @returns {Promise<AnalyzeResponse>}
 */
export async function analyze(query, k = 5) {
  const res = await fetch(`${BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, k }),
  })

  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch (_) {}
    throw new Error(`Analysis failed (${res.status}): ${detail}`)
  }

  return res.json()
}
