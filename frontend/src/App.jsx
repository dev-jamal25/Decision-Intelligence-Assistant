import { useState } from 'react'
import './App.css'
import { analyze } from './services/api'
import QueryInput from './components/QueryInput'
import PriorityPanel from './components/PriorityPanel'
import OutputSwitcher from './components/OutputSwitcher'
import MetricsPanel from './components/MetricsPanel'
import SourcePanel from './components/SourcePanel'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit(query) {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await analyze(query)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Decision Intelligence Assistant</h1>
        <p>Analyze customer support queries with RAG, ML priority prediction, and zero-shot LLM.</p>
      </header>

      <QueryInput onSubmit={handleSubmit} loading={loading} />

      {error && <div className="error-banner">{error}</div>}
      {loading && <div className="loading-banner">Analyzing query&hellip;</div>}

      {result && (
        <>
          <div className="panel">
            <div className="panel-title">Query</div>
            <p className="query-echo">&ldquo;{result.query}&rdquo;</p>
          </div>

          <PriorityPanel
            prediction={result.priority_prediction}
            confidence={result.priority_confidence}
            model={result.priority_model}
          />

          <OutputSwitcher
            ragAnswer={result.rag_answer}
            nonRagAnswer={result.non_rag_answer}
            model={result.answer_model}
          />

          <MetricsPanel
            retrievedCount={result.retrieved_count}
            topScore={result.top_score}
            isWeak={result.retrieval_is_weak}
            threshold={result.retrieval_threshold}
          />

          <SourcePanel cases={result.retrieved_cases} />
        </>
      )}
    </div>
  )
}
