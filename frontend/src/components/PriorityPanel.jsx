export default function PriorityPanel({ prediction, confidence, model }) {
  const isUrgent = prediction?.toLowerCase() === 'urgent'

  return (
    <div className="panel">
      <div className="panel-title">Priority Prediction</div>
      <div className="priority-row">
        <span className={`badge ${isUrgent ? 'badge-urgent' : 'badge-normal'}`}>
          {prediction}
        </span>
        <span className="confidence-text">
          {(confidence * 100).toFixed(0)}% confidence
        </span>
      </div>
      <p className="model-text">Model: {model}</p>
    </div>
  )
}
