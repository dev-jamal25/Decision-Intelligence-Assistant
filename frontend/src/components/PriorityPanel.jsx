function PriorityCard({ label, prediction, confidence, sub }) {
  const isUrgent = prediction?.toLowerCase() === 'urgent'
  return (
    <div className="priority-card">
      <div className="priority-card-label">{label}</div>
      <div className="priority-row">
        <span className={`badge ${isUrgent ? 'badge-urgent' : 'badge-normal'}`}>
          {prediction}
        </span>
        {confidence != null && (
          <span className="confidence-text">
            {(confidence * 100).toFixed(0)}% confidence
          </span>
        )}
      </div>
      {sub && <p className="model-text">{sub}</p>}
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
    <div className="panel">
      <div className="panel-title">Priority Comparison</div>
      <div className="priority-comparison-grid">
        <PriorityCard
          label="ML Model"
          prediction={mlPrediction}
          confidence={mlConfidence}
          sub={`Model: ${mlModel}`}
        />
        <PriorityCard
          label="LLM Zero-shot"
          prediction={llmPrediction}
          confidence={llmConfidence}
          sub={`Model: ${answerModel}`}
        />
      </div>
      {llmRationale && (
        <p className="llm-rationale">LLM rationale: {llmRationale}</p>
      )}
      <p className={`agreement-note ${agree ? 'agreement-yes' : 'agreement-no'}`}>
        {agree ? 'Both models agree' : 'Models disagree on priority'}
      </p>
    </div>
  )
}
