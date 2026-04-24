import { useState } from 'react'

export default function QueryInput({ onSubmit, loading }) {
  const [value, setValue] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = value.trim()
    if (trimmed) onSubmit(trimmed)
  }

  return (
    <form className="query-form" onSubmit={handleSubmit}>
      <textarea
        className="query-textarea"
        rows={3}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Describe the support issue or paste a customer query…"
        disabled={loading}
      />
      <button
        className="query-btn"
        type="submit"
        disabled={loading || !value.trim()}
      >
        {loading ? 'Analyzing…' : 'Analyze'}
      </button>
    </form>
  )
}
