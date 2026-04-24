export default function SourcePanel({ cases }) {
  return (
    <div className="panel">
      <div className="panel-title">Retrieved Cases ({cases.length})</div>
      {cases.length === 0 ? (
        <p className="no-cases">No cases retrieved.</p>
      ) : (
        <ul className="sources-list">
          {cases.map((c) => (
            <li key={c.case_id} className="source-item">
              <div className="source-header">
                <code className="source-id">{c.case_id}</code>
                <span className="score-badge">{(c.score * 100).toFixed(0)}%</span>
              </div>
              <p className="source-text">{c.text}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
