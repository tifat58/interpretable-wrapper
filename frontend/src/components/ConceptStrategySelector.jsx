import { useState, useEffect, useCallback } from 'react'

export default function ConceptStrategySelector({ domain, onStrategyChange, currentStrategy }) {
  const [strategies, setStrategies] = useState([])
  const [customInput, setCustomInput] = useState('')
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    if (!domain) return
    fetch(`/concept_strategies?domain=${encodeURIComponent(domain)}`)
      .then((r) => r.json())
      .then((d) => {
        setStrategies(d.strategies || [])
        // Reset custom input when domain changes
        setCustomInput('')
      })
      .catch(() => setStrategies([]))
  }, [domain])

  const handleSelect = useCallback((strategyId) => {
    if (strategyId === 'custom') {
      onStrategyChange({ strategy: 'custom', custom_concepts: customInput ? customInput.split(',').map((s) => s.trim()).filter(Boolean) : null })
    } else {
      onStrategyChange({ strategy: strategyId, custom_concepts: null })
    }
  }, [onStrategyChange, customInput])

  const handleCustomSubmit = useCallback(() => {
    const concepts = customInput.split(',').map((s) => s.trim()).filter(Boolean)
    if (concepts.length > 0) {
      onStrategyChange({ strategy: 'custom', custom_concepts: concepts })
    }
  }, [customInput, onStrategyChange])

  if (strategies.length === 0) return null

  const strategyIcons = {
    predefined: '📋',
    clip: '🔗',
    pca: '📐',
    kmeans: '🎯',
    custom: '✏️',
    token_aggregation: '🔤',
  }

  return (
    <div className="glass-card rounded-2xl shadow-md p-5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between"
      >
        <h3 className="text-sm font-bold text-gray-700 flex items-center gap-2">
          <svg className="w-4 h-4 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
          </svg>
          Concept Extraction
          <span className="ml-2 text-[10px] bg-indigo-100 text-indigo-600 px-1.5 py-0.5 rounded-full font-bold uppercase">
            {currentStrategy}
          </span>
        </h3>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="mt-3 space-y-1.5">
          {strategies.map((s) => (
            <label
              key={s.id}
              className={`flex items-start gap-2.5 px-3 py-2.5 rounded-xl cursor-pointer text-sm transition-all duration-150 ${
                currentStrategy === s.id
                  ? 'bg-indigo-50/80 text-indigo-700 ring-1 ring-indigo-200'
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              <input
                type="radio"
                name="concept_strategy"
                value={s.id}
                checked={currentStrategy === s.id}
                onChange={() => handleSelect(s.id)}
                className="accent-indigo-600 mt-0.5"
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-base">{strategyIcons[s.id] || '📊'}</span>
                  <span className="font-semibold">{s.name}</span>
                </div>
                <p className="text-[11px] text-gray-400 mt-0.5 leading-tight">{s.description}</p>
              </div>
            </label>
          ))}

          {/* Custom concept input */}
          {currentStrategy === 'custom' && (
            <div className="mt-2 pt-2 border-t border-gray-100 space-y-2">
              <p className="text-xs text-gray-500 font-semibold">Enter concepts (comma-separated)</p>
              <textarea
                className="w-full text-xs border border-gray-200 rounded-lg px-3 py-2 focus:ring-2 focus:ring-indigo-300 focus:outline-none resize-none"
                rows={2}
                placeholder="e.g. furry, has stripes, is large, has wings"
                value={customInput}
                onChange={(e) => setCustomInput(e.target.value)}
              />
              <button
                onClick={handleCustomSubmit}
                className="w-full text-xs bg-gradient-to-r from-indigo-600 to-indigo-700 text-white rounded-lg py-2 font-semibold hover:from-indigo-700 hover:to-indigo-800 transition-all"
              >
                Apply Custom Concepts
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
