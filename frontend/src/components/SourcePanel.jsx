function scoreChipClass(score) {
  if (score >= 0.6) return 'score-chip--high'
  if (score >= 0.35) return 'score-chip--mid'
  return 'score-chip--low'
}

export default function SourcePanel({ cases }) {
  return (
    <div className="section">
      <div className="section-label">Retrieved Cases ({cases.length})</div>
      {cases.length === 0 ? (
        <p className="no-cases">No cases retrieved.</p>
      ) : (
        <ul className="cases-list">
          {cases.map((c) => (
            <li key={c.case_id} className="case-item">
              <div className="case-header">
                <code className="case-id">{c.case_id}</code>
                <span className={`score-chip ${scoreChipClass(c.score)}`}>
                  {(c.score * 100).toFixed(0)}%
                </span>
              </div>
              <p className="case-text">{c.text}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
