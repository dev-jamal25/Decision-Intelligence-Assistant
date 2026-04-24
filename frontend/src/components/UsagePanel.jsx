export default function UsagePanel({ usageInfo, costInfo, provider, fallbackUsed }) {
  if (!usageInfo && !costInfo) return null

  const totalTokens = usageInfo?.total_tokens
  const promptTokens = usageInfo?.prompt_tokens
  const completionTokens = usageInfo?.completion_tokens
  const model = usageInfo?.model
  const estimatedUsd = costInfo?.estimated_usd
  const costNote = costInfo?.note

  return (
    <div className="panel">
      <div className="panel-title">Usage &amp; Cost</div>
      <div className="metrics-row">
        {totalTokens != null && (
          <div className="metric-item">
            <div className="metric-label">Total Tokens</div>
            <div className="metric-value">{totalTokens.toLocaleString()}</div>
          </div>
        )}
        {promptTokens != null && (
          <div className="metric-item">
            <div className="metric-label">Prompt</div>
            <div className="metric-value">{promptTokens.toLocaleString()}</div>
          </div>
        )}
        {completionTokens != null && (
          <div className="metric-item">
            <div className="metric-label">Completion</div>
            <div className="metric-value">{completionTokens.toLocaleString()}</div>
          </div>
        )}
        <div className="metric-item">
          <div className="metric-label">Est. Cost</div>
          <div className="metric-value">
            {estimatedUsd != null ? `$${estimatedUsd.toFixed(4)}` : '—'}
          </div>
        </div>
      </div>

      <div className="usage-meta">
        {provider && <span className="usage-tag">Provider: {provider}</span>}
        {model && <span className="usage-tag">Model: {model}</span>}
        {fallbackUsed && <span className="usage-tag usage-tag-warn">Fallback used</span>}
        {costNote && <span className="usage-tag">{costNote}</span>}
      </div>
    </div>
  )
}
