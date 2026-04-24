export default function AnswerPanel({ answer, label }) {
  return (
    <div>
      <div className="section-label">{label}</div>
      <div className="answer-text">{answer}</div>
    </div>
  )
}
