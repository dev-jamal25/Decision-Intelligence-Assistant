import { useState } from 'react'
import './App.css'
import { analyze } from './services/api'
import QueryInput from './components/QueryInput'
import PriorityPanel from './components/PriorityPanel'
import OutputSwitcher from './components/OutputSwitcher'
import MetricsPanel from './components/MetricsPanel'
import SourcePanel from './components/SourcePanel'
import UsagePanel from './components/UsagePanel'

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
            mlPrediction={result.ml_priority_prediction}
            mlConfidence={result.ml_priority_confidence}
            mlModel={result.priority_model}
            llmPrediction={result.llm_zero_shot_priority_prediction}
            llmConfidence={result.llm_zero_shot_priority_confidence}
            llmRationale={result.llm_zero_shot_priority_rationale}
            answerModel={result.answer_model}
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
            latencyMs={result.latency_ms}
          />

          <UsagePanel
            usageInfo={result.usage_info}
            costInfo={result.cost_info}
            provider={result.answer_provider}
            fallbackUsed={result.fallback_used}
          />

          <SourcePanel cases={result.retrieved_cases} />
        </>
      )}
    </div>
  )
}
