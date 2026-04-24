function fmt(ms) {
  if (ms == null) return '—'
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}

export default function MetricsPanel({ retrievedCount, topScore, isWeak, threshold, latencyMs }) {
  return (
    <div className="panel">
      <div className="panel-title">Retrieval Diagnostics</div>
      <div className="metrics-row">
        <div className="metric-item">
          <div className="metric-label">Cases Retrieved</div>
          <div className="metric-value">{retrievedCount}</div>
        </div>
        <div className="metric-item">
          <div className="metric-label">Top Score</div>
          <div className="metric-value">
            {topScore != null ? `${(topScore * 100).toFixed(0)}%` : '—'}
          </div>
        </div>
        <div className="metric-item">
          <div className="metric-label">Threshold</div>
          <div className="metric-value">{(threshold * 100).toFixed(0)}%</div>
        </div>
      </div>
      {isWeak && (
        <div className="weak-warning">
          Weak retrieval — no close matches found. The answer may rely more on general LLM knowledge than retrieved cases.
        </div>
      )}

      {latencyMs && (
        <>
          <div className="panel-title" style={{ marginTop: '1.25rem' }}>Latency Breakdown</div>
          <div className="latency-grid">
            {[
              ['Retrieval', latencyMs.retrieval],
              ['ML Priority', latencyMs.ml],
              ['LLM Priority', latencyMs.llm_zero_shot_priority],
              ['RAG Answer', latencyMs.rag],
              ['Non-RAG', latencyMs.non_rag],
              ['Total', latencyMs.total],
            ].map(([label, val]) => (
              <div key={label} className={`latency-item${label === 'Total' ? ' latency-total' : ''}`}>
                <div className="metric-label">{label}</div>
                <div className="metric-value">{fmt(val)}</div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
