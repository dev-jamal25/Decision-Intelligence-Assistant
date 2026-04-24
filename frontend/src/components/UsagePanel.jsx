function CallCard({ label, usage }) {
  if (!usage) return null
  const cost = usage.estimated_cost_usd
  return (
    <div className="usage-card">
      <div className="usage-card-label">{label}</div>
      <div className="usage-tags">
        <span className="usage-tag">{usage.provider}</span>
        <span className="usage-tag">{usage.model}</span>
      </div>
      <div className="usage-tokens">
        {usage.prompt_tokens != null && (
          <span>In: <strong>{usage.prompt_tokens.toLocaleString()}</strong></span>
        )}
        {usage.completion_tokens != null && (
          <span>Out: <strong>{usage.completion_tokens.toLocaleString()}</strong></span>
        )}
      </div>
      <div className="usage-cost">
        {cost != null ? `$${cost.toFixed(6)}` : '—'}
      </div>
    </div>
  )
}

export default function UsagePanel({
  ragAnswerUsage,
  nonRagAnswerUsage,
  llmPriorityUsage,
  usageSummary,
  fallbackUsed,
}) {
  if (!ragAnswerUsage && !nonRagAnswerUsage && !llmPriorityUsage) return null
  const totalCost = usageSummary?.estimated_cost_usd

  return (
    <div className="section">
      <div className="section-label">LLM Usage &amp; Cost</div>
      <div className="usage-grid">
        <CallCard label="RAG Answer" usage={ragAnswerUsage} />
        <CallCard label="Non-RAG Answer" usage={nonRagAnswerUsage} />
        <CallCard label="Priority (LLM)" usage={llmPriorityUsage} />
      </div>
      {usageSummary && (
        <div className="usage-summary">
          {usageSummary.total_tokens != null && (
            <span className="usage-summary-tag">
              {usageSummary.total_tokens.toLocaleString()} tokens total
            </span>
          )}
          <span className="usage-summary-tag">
            Est. cost: {totalCost != null ? `$${totalCost.toFixed(6)}` : '—'}
          </span>
          {fallbackUsed && (
            <span className="usage-summary-tag usage-summary-tag--warn">Fallback used</span>
          )}
        </div>
      )}
    </div>
  )
}
