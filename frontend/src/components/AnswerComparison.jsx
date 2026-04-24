export default function AnswerComparison({ ragAnswer, nonRagAnswer, model, provider, retrievalIsWeak }) {
  return (
    <div className="section">
      <div className="section-label">Answer Comparison</div>
      <div className="answer-comparison-grid">
        <div className="answer-col answer-col--rag">
          <div className="answer-col-header">
            <span>RAG</span>
            <span className="answer-col-tag answer-col-tag--rag">With context</span>
            {retrievalIsWeak && (
              <span className="answer-col-tag answer-col-tag--weak">Weak retrieval</span>
            )}
          </div>
          <div className="answer-text">{ragAnswer}</div>
          <div className="answer-model-row">{provider} · {model}</div>
        </div>

        <div className="vs-divider">
          <div className="vs-divider-line" />
          <span className="vs-badge">vs</span>
          <div className="vs-divider-line" />
        </div>

        <div className="answer-col answer-col--nonrag">
          <div className="answer-col-header">
            <span>Non-RAG</span>
            <span className="answer-col-tag">Zero-shot</span>
          </div>
          <div className="answer-text">{nonRagAnswer}</div>
          <div className="answer-model-row">{provider} · {model}</div>
        </div>
      </div>
    </div>
  )
}
