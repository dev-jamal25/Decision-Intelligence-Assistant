export default function MetricsPanel({ retrievedCount, topScore, isWeak, threshold }) {
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
    </div>
  )
}
