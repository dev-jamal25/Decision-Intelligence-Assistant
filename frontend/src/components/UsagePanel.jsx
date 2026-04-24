function CallCard({ label, usage }) {
  if (!usage) return null
  const cost = usage.estimated_cost_usd
  return (
    <div className="call-usage-card">
      <div className="call-usage-label">{label}</div>
      <div className="call-usage-meta">
        <span className="usage-tag">{usage.provider}</span>
        <span className="usage-tag">{usage.model}</span>
      </div>
      <div className="call-usage-tokens">
        {usage.prompt_tokens != null && (
          <span>In: <strong>{usage.prompt_tokens.toLocaleString()}</strong></span>
        )}
        {usage.completion_tokens != null && (
          <span>Out: <strong>{usage.completion_tokens.toLocaleString()}</strong></span>
        )}
        {usage.total_tokens != null && (
          <span>Total: <strong>{usage.total_tokens.toLocaleString()}</strong></span>
        )}
      </div>
      <div className="call-usage-cost">
        {cost != null ? `$${cost.toFixed(6)}` : 'Cost: —'}
      </div>
    </div>
  )
}

export default function UsagePanel({ ragAnswerUsage, nonRagAnswerUsage, llmPriorityUsage, usageSummary, fallbackUsed }) {
  if (!ragAnswerUsage && !nonRagAnswerUsage && !llmPriorityUsage) return null

  const summary = usageSummary
  const totalCost = summary?.estimated_cost_usd

  return (
    <div className="panel">
      <div className="panel-title">LLM Usage &amp; Cost</div>

      <div className="call-usage-grid">
        <CallCard label="RAG Answer" usage={ragAnswerUsage} />
        <CallCard label="Non-RAG Answer" usage={nonRagAnswerUsage} />
        <CallCard label="Priority (zero-shot)" usage={llmPriorityUsage} />
      </div>

      {summary && (
        <div className="usage-summary-row">
          {summary.total_tokens != null && (
            <span className="usage-tag">Total tokens: {summary.total_tokens.toLocaleString()}</span>
          )}
          <span className="usage-tag">
            Est. total: {totalCost != null ? `$${totalCost.toFixed(6)}` : '—'}
          </span>
          {fallbackUsed && <span className="usage-tag usage-tag-warn">Fallback used</span>}
        </div>
      )}
    </div>
  )
}
