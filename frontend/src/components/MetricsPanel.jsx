function fmt(ms) {
  if (ms == null) return '—'
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}

export default function MetricsPanel({ retrievedCount, topScore, isWeak, threshold, latencyMs }) {
  const scoreStr = topScore != null ? `${(topScore * 100).toFixed(0)}%` : '—'
  const strengthMod = isWeak ? 'signal-chip--weak' : 'signal-chip--strong'
  const strengthLabel = isWeak ? 'Weak retrieval' : 'Strong retrieval'

  return (
    <div className="section">
      <div className="section-label">Retrieval Diagnostics</div>

      <div className="signal-row">
        <div className="signal-chip">
          <span className="signal-chip-label">Cases</span>
          <span className="signal-chip-value">{retrievedCount}</span>
        </div>
        <div className="signal-chip">
          <span className="signal-chip-label">Top Score</span>
          <span className="signal-chip-value">{scoreStr}</span>
        </div>
        <div className="signal-chip">
          <span className="signal-chip-label">Threshold</span>
          <span className="signal-chip-value">{(threshold * 100).toFixed(0)}%</span>
        </div>
        <div className={`signal-chip ${strengthMod}`}>
          <span className="signal-chip-value">{strengthLabel}</span>
        </div>
      </div>

      {isWeak && (
        <div className="weak-alert">
          <span className="weak-alert-icon">⚠</span>
          <div>
            <div className="weak-alert-title">Weak retrieval</div>
            <div className="weak-alert-body">
              No close matches found in the knowledge base. The RAG answer may rely more on
              general LLM knowledge than retrieved cases.
            </div>
          </div>
        </div>
      )}

      {latencyMs && (
        <>
          <div className="latency-section-label">Latency</div>
          <div className="latency-grid">
            {[
              ['Retrieval', latencyMs.retrieval],
              ['ML', latencyMs.ml],
              ['LLM Priority', latencyMs.llm_zero_shot_priority],
              ['RAG Answer', latencyMs.rag],
              ['Non-RAG', latencyMs.non_rag],
              ['Total', latencyMs.total],
            ].map(([label, val]) => (
              <div
                key={label}
                className={`latency-chip${label === 'Total' ? ' latency-chip--total' : ''}`}
              >
                <div className="latency-chip-label">{label}</div>
                <div className="latency-chip-value">{fmt(val)}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
