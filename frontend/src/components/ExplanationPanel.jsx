import { useState } from 'react'

export default function ExplanationPanel({ explanation, onExplain, hasPrediction }) {
  const [showEvidence, setShowEvidence] = useState(false)
  const [loading, setLoading] = useState(false)
  const [expanded, setExpanded] = useState({})

  const handleGenerate = async () => {
    setLoading(true)
    try {
      await onExplain(showEvidence)
    } finally {
      setLoading(false)
    }
  }

  const toggleEvidence = async () => {
    const next = !showEvidence
    setShowEvidence(next)
    if (hasPrediction) {
      setLoading(true)
      try {
        await onExplain(next)
      } finally {
        setLoading(false)
      }
    }
  }

  const toggleSnippet = (concept) => {
    setExpanded((prev) => ({ ...prev, [concept]: !prev[concept] }))
  }

  return (
    <div className="bg-white rounded-xl shadow p-5">
      <h2 className="text-lg font-semibold mb-3">Explanation</h2>

      <div className="flex items-center gap-4 mb-4">
        <button
          onClick={handleGenerate}
          disabled={!hasPrediction || loading}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-300 text-white font-medium px-4 py-2 rounded-lg transition text-sm"
        >
          {loading ? 'Generating…' : 'Generate Explanation'}
        </button>

        <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer select-none">
          <div
            onClick={toggleEvidence}
            className={`relative w-10 h-5 rounded-full transition ${showEvidence ? 'bg-indigo-600' : 'bg-gray-300'}`}
          >
            <div
              className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform ${
                showEvidence ? 'translate-x-5' : ''
              }`}
            />
          </div>
          Show evidence
        </label>
      </div>

      {!explanation && (
        <p className="text-gray-400 italic text-sm">
          {hasPrediction
            ? 'Click "Generate Explanation" to see a natural language explanation.'
            : 'Run a prediction first.'}
        </p>
      )}

      {explanation && (
        <div className="space-y-4">
          <p className="text-sm leading-relaxed text-gray-800">{explanation.explanation_text}</p>

          {explanation.evidence_snippets?.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-gray-700">Evidence</h3>
              {explanation.evidence_snippets.map(({ concept, activation, text }) => (
                <div key={concept} className="border rounded-lg">
                  <button
                    onClick={() => toggleSnippet(concept)}
                    className="w-full flex justify-between items-center px-3 py-2 text-sm font-medium text-left hover:bg-gray-50"
                  >
                    <span className="capitalize">{concept.replace(/_/g, ' ')} ({(activation * 100).toFixed(0)}%)</span>
                    <span className="text-gray-400">{expanded[concept] ? '▲' : '▼'}</span>
                  </button>
                  {expanded[concept] && (
                    <p className="px-3 pb-3 text-sm text-gray-600 leading-relaxed">{text}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
