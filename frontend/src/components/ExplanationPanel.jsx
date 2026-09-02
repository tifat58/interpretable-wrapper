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
    <div className="glass-card rounded-2xl shadow-md p-6">
      <h2 className="text-lg font-bold text-gray-800 mb-4">Explanation</h2>

      <div className="flex items-center gap-4 mb-4">
        <button
          onClick={handleGenerate}
          disabled={!hasPrediction || loading}
          className="bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 disabled:from-gray-300 disabled:to-gray-300 text-white font-semibold px-5 py-2.5 rounded-xl transition-all text-sm shadow-sm hover:shadow-md"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Generating…
            </span>
          ) : 'Generate Explanation'}
        </button>

        <label className="flex items-center gap-2.5 text-sm text-gray-600 cursor-pointer select-none">
          <div
            onClick={toggleEvidence}
            className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${showEvidence ? 'bg-indigo-600' : 'bg-gray-300'}`}
          >
            <div
              className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow-sm transition-transform duration-200 ${
                showEvidence ? 'translate-x-5' : ''
              }`}
            />
          </div>
          <span className="font-medium">Show evidence</span>
        </label>
      </div>

      {!explanation && (
        <div className="flex flex-col items-center justify-center py-6 text-center">
          <p className="text-gray-400 italic text-sm">
            {hasPrediction
              ? 'Click "Generate Explanation" to see a natural language explanation.'
              : 'Run a prediction first.'}
          </p>
        </div>
      )}

      {explanation && (
        <div className="space-y-4">
          <div className="bg-gray-50/80 rounded-xl p-4 border border-gray-100">
            {explanation.highlighted_segments ? (
              <p className="text-sm leading-relaxed text-gray-800">
                {explanation.highlighted_segments.map((seg, i) => {
                  if (seg.type === 'decision') {
                    return <span key={i} className="font-bold text-indigo-700 bg-indigo-50 px-1 rounded">{seg.text}</span>
                  }
                  if (seg.type === 'concept') {
                    return <span key={i} className="font-semibold text-purple-700 bg-purple-50 px-1 rounded">{seg.text}</span>
                  }
                  if (seg.type === 'percentage') {
                    return <span key={i} className="font-bold text-emerald-700 bg-emerald-50 px-1 rounded">{seg.text}</span>
                  }
                  return <span key={i}>{seg.text}</span>
                })}
              </p>
            ) : (
              <p className="text-sm leading-relaxed text-gray-800">{explanation.explanation_text}</p>
            )}
          </div>

          {explanation.evidence_snippets?.length > 0 && (
            <div className="space-y-2">
              <h3 className="text-sm font-bold text-gray-700 flex items-center gap-1.5">
                <svg className="w-4 h-4 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Evidence
              </h3>
              {explanation.evidence_snippets.map(({ concept, activation, text }) => (
                <div key={concept} className="border border-gray-200 rounded-xl overflow-hidden">
                  <button
                    onClick={() => toggleSnippet(concept)}
                    className="w-full flex justify-between items-center px-4 py-2.5 text-sm font-semibold text-left hover:bg-gray-50/80 transition"
                  >
                    <span className="flex items-center gap-2">
                      <span className="w-1 h-6 rounded-full bg-indigo-400 flex-shrink-0" />
                      <span className="capitalize">{concept.replace(/_/g, ' ')}</span>
                      <span className="text-xs text-gray-400 font-normal">({(activation * 100).toFixed(0)}%)</span>
                    </span>
                    <span className="text-gray-400 text-xs">{expanded[concept] ? '▲' : '▼'}</span>
                  </button>
                  {expanded[concept] && (
                    <div className="px-4 pb-3 border-t border-gray-100">
                      <p className="text-sm text-gray-600 leading-relaxed pt-2.5 pl-3">{text}</p>
                    </div>
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
