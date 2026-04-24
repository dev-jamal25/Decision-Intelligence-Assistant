export default function QueryInput({ onSubmit, loading, value, onChange }) {
  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = value.trim()
    if (trimmed && !loading) onSubmit(trimmed)
  }

  function handleKeyDown(e) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault()
      const trimmed = value.trim()
      if (trimmed && !loading) onSubmit(trimmed)
    }
  }

  return (
    <div className="query-block">
      <form onSubmit={handleSubmit}>
        <textarea
          className="query-textarea"
          rows={4}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe the support issue or paste a customer query…"
          disabled={loading}
        />
        <div className="query-footer">
          <span className="query-hint">⌘ Enter to analyze</span>
          <button
            className="query-btn"
            type="submit"
            disabled={loading || !value.trim()}
          >
            {loading ? (
              <>
                <div className="spinner" />
                Analyzing…
              </>
            ) : (
              'Analyze Query'
            )}
          </button>
        </div>
      </form>
    </div>
  )
}
