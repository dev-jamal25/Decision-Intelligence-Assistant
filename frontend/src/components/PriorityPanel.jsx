function PriorityCard({ label, prediction, confidence, sub, rationale }) {
  const isUrgent = prediction?.toLowerCase() === 'urgent'
  const mod = isUrgent ? 'priority-card--urgent' : 'priority-card--normal'
  return (
    <div className={`priority-card ${mod}`}>
      <div className="priority-card-label">{label}</div>
      <div className="priority-row">
        <span className={`badge ${isUrgent ? 'badge-urgent' : 'badge-normal'}`}>
          {prediction ?? '—'}
        </span>
        {confidence != null && (
          <span className="confidence-text">
            {(confidence * 100).toFixed(0)}% conf.
          </span>
        )}
      </div>
      {rationale && (
        <p className="priority-rationale">"{rationale}"</p>
      )}
      {sub && <p className="model-tag">{sub}</p>}
    </div>
  )
}

export default function PriorityPanel({
  mlPrediction,
  mlConfidence,
  mlModel,
  llmPrediction,
  llmConfidence,
  llmRationale,
  answerModel,
}) {
  const agree = mlPrediction?.toLowerCase() === llmPrediction?.toLowerCase()

  return (
    <div className="section">
      <div className="section-label">Priority Comparison</div>
      <div className="priority-grid">
        <PriorityCard
          label="ML Model"
          prediction={mlPrediction}
          confidence={mlConfidence}
          sub={`Model: ${mlModel}`}
        />
        <PriorityCard
          label="LLM Zero-Shot"
          prediction={llmPrediction}
          confidence={llmConfidence}
          rationale={llmRationale}
          sub={`Model: ${answerModel}`}
        />
      </div>
      <div className="agreement-row">
        <span className={`agreement-badge ${agree ? 'agreement-badge--agree' : 'agreement-badge--disagree'}`}>
          {agree ? '✓ Both models agree' : '⚡ Models disagree on priority'}
        </span>
      </div>
    </div>
  )
}
