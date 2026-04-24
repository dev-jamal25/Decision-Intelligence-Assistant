export default function AnswerPanel({ answer, label }) {
  return (
    <div>
      <div className="panel-title">{label}</div>
      <pre className="answer-text">{answer}</pre>
    </div>
  )
}
