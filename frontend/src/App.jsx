import { useState } from 'react'
import './App.css'
import { analyze } from './services/api'
import QueryInput from './components/QueryInput'
import PriorityPanel from './components/PriorityPanel'
import AnswerComparison from './components/AnswerComparison'
import MetricsPanel from './components/MetricsPanel'
import SourcePanel from './components/SourcePanel'
import UsagePanel from './components/UsagePanel'

const EXAMPLES = [
  'I was charged twice for my subscription this month and my account still shows as expired.',
  'How do I change the email address associated with my account?',
]

export default function App() {
  const [inputValue, setInputValue] = useState('')
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
      <header className="app-header">
        <div className="app-header-inner">
          <span className="app-title">Decision Intelligence Assistant</span>
          <span className="app-subtitle">
            RAG · ML Priority · LLM Zero-Shot — side-by-side comparison
          </span>
        </div>
      </header>

      <QueryInput
        onSubmit={handleSubmit}
        loading={loading}
        value={inputValue}
        onChange={setInputValue}
      />

      {error && (
        <div className="error-state">
          <span>⚠</span>
          <span>{error}</span>
        </div>
      )}

      {loading && (
        <div className="loading-state">
          <div className="spinner spinner--page" />
          <span>Analyzing query…</span>
        </div>
      )}

      {!result && !loading && !error && (
        <div className="empty-state">
          <p className="empty-state-title">Try an example query</p>
          <div className="empty-examples">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                className="example-btn"
                onClick={() => setInputValue(ex)}
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      )}

      {result && (
        <>
          <div className="section">
            <div className="section-label">Query</div>
            <p className="query-echo">"{result.query}"</p>
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

          <AnswerComparison
            ragAnswer={result.rag_answer}
            nonRagAnswer={result.non_rag_answer}
            model={result.answer_model}
            provider={result.answer_provider}
            retrievalIsWeak={result.retrieval_is_weak}
          />

          <MetricsPanel
            retrievedCount={result.retrieved_count}
            topScore={result.top_score}
            isWeak={result.retrieval_is_weak}
            threshold={result.retrieval_threshold}
            latencyMs={result.latency_ms}
          />

          <UsagePanel
            ragAnswerUsage={result.rag_answer_usage}
            nonRagAnswerUsage={result.non_rag_answer_usage}
            llmPriorityUsage={result.llm_zero_shot_priority_usage}
            usageSummary={result.usage_summary}
            fallbackUsed={result.fallback_used}
          />

          <SourcePanel cases={result.retrieved_cases} />
        </>
      )}
    </div>
  )
}
