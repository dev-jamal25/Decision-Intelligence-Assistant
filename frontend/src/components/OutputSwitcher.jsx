import { useState } from 'react'
import AnswerPanel from './AnswerPanel'

export default function OutputSwitcher({ ragAnswer, nonRagAnswer, model }) {
  const [activeTab, setActiveTab] = useState('rag')

  return (
    <div className="panel">
      <div className="tabs">
        <button
          className={`tab-btn ${activeTab === 'rag' ? 'active' : ''}`}
          onClick={() => setActiveTab('rag')}
        >
          RAG Answer
        </button>
        <button
          className={`tab-btn ${activeTab === 'non-rag' ? 'active' : ''}`}
          onClick={() => setActiveTab('non-rag')}
        >
          Non-Rag Answer
        </button>
      </div>
      <AnswerPanel
        answer={activeTab === 'rag' ? ragAnswer : nonRagAnswer}
        label={activeTab === 'rag' ? 'RAG-grounded response' : 'Zero-shot response (no context)'}
      />
      <p className="model-text" style={{ marginTop: '0.75rem' }}>Model: {model}</p>
    </div>
  )
}
