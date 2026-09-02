import { useState, useCallback } from 'react'

export default function SurrogatePanel({ surrogateInfo, onFitSurrogate, hasPrediction }) {
  const [surrogateType, setSurrogateType] = useState('logistic')
  const [nPerturbations, setNPerturbations] = useState(200)
  const [fitting, setFitting] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const handleFit = useCallback(async () => {
    setFitting(true)
    try {
      await onFitSurrogate(surrogateType, nPerturbations)
    } finally {
      setFitting(false)
    }
  }, [onFitSurrogate, surrogateType, nPerturbations])

  const typeOptions = [
    { id: 'logistic', label: 'Logistic Regression', icon: '📈' },
    { id: 'ridge', label: 'Ridge Regression', icon: '📉' },
    { id: 'tree', label: 'Decision Tree', icon: '🌳' },
  ]

  // Color for fidelity score
  const fidelityColor = surrogateInfo?.fidelity_score >= 0.8
    ? 'text-green-600'
    : surrogateInfo?.fidelity_score >= 0.5
    ? 'text-yellow-600'
    : 'text-red-500'

  return (
    <div className="glass-card rounded-2xl shadow-md p-5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between"
      >
        <h3 className="text-sm font-bold text-gray-700 flex items-center gap-2">
          <svg className="w-4 h-4 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Local Surrogate
          {surrogateInfo && (
            <span className={`ml-2 text-[10px] px-1.5 py-0.5 rounded-full font-bold uppercase ${
              surrogateInfo.fidelity_score >= 0.8 ? 'bg-green-100 text-green-600' : 'bg-yellow-100 text-yellow-600'
            }`}>
              fidelity {(surrogateInfo.fidelity_score * 100).toFixed(0)}%
            </span>
          )}
        </h3>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="mt-3 space-y-3">
          {/* Surrogate type selector */}
          <div>
            <p className="text-xs text-gray-500 font-semibold mb-1.5">Surrogate Model</p>
            <div className="flex gap-1.5">
              {typeOptions.map((opt) => (
                <button
                  key={opt.id}
                  onClick={() => setSurrogateType(opt.id)}
                  className={`flex-1 text-xs px-2 py-1.5 rounded-lg font-medium transition-all ${
                    surrogateType === opt.id
                      ? 'bg-purple-100 text-purple-700 ring-1 ring-purple-200'
                      : 'bg-gray-50 text-gray-500 hover:bg-gray-100'
                  }`}
                >
                  <span className="mr-1">{opt.icon}</span>
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* Perturbation slider */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <p className="text-xs text-gray-500 font-semibold">Perturbations</p>
              <span className="text-xs text-gray-400 font-mono">{nPerturbations}</span>
            </div>
            <input
              type="range"
              min={50}
              max={500}
              step={50}
              value={nPerturbations}
              onChange={(e) => setNPerturbations(parseInt(e.target.value))}
              className="w-full h-1.5 rounded-lg appearance-none bg-gray-200 accent-purple-600"
            />
            <div className="flex justify-between text-[9px] text-gray-300 mt-0.5">
              <span>50</span><span>500</span>
            </div>
          </div>

          {/* Fit button */}
          <button
            onClick={handleFit}
            disabled={!hasPrediction || fitting}
            className="w-full text-xs bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-lg py-2 font-semibold hover:from-purple-700 hover:to-purple-800 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {fitting ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin w-3 h-3" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Fitting…
              </span>
            ) : 'Fit Local Surrogate'}
          </button>
          {!hasPrediction && (
            <p className="text-[10px] text-gray-400 text-center">Run a prediction first</p>
          )}

          {/* Results */}
          {surrogateInfo && !surrogateInfo.error && (
            <div className="border-t border-gray-100 pt-3 space-y-3">
              {/* Fidelity */}
              <div className="flex items-center gap-3">
                <div className="flex-1">
                  <p className="text-xs text-gray-500 font-semibold mb-1">Fidelity Score</p>
                  <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        surrogateInfo.fidelity_score >= 0.8 ? 'bg-green-500' :
                        surrogateInfo.fidelity_score >= 0.5 ? 'bg-yellow-400' : 'bg-red-400'
                      }`}
                      style={{ width: `${surrogateInfo.fidelity_score * 100}%` }}
                    />
                  </div>
                </div>
                <span className={`text-lg font-bold ${fidelityColor}`}>
                  {(surrogateInfo.fidelity_score * 100).toFixed(0)}%
                </span>
              </div>

              {/* Importance weights */}
              <div>
                <p className="text-xs text-gray-500 font-semibold mb-2">Concept Importance</p>
                <div className="space-y-1 max-h-60 overflow-y-auto pr-1">
                  {surrogateInfo.importance_weights &&
                    Object.entries(surrogateInfo.importance_weights)
                      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                      .slice(0, 15)
                      .map(([name, weight]) => {
                        const maxW = Math.max(
                          ...Object.values(surrogateInfo.importance_weights).map(Math.abs),
                          0.01,
                        )
                        const pct = Math.abs(weight) / maxW * 100
                        return (
                          <div key={name} className="flex items-center gap-2">
                            <span className="w-24 text-[10px] text-gray-600 truncate text-right" title={name}>
                              {name.replace(/_/g, ' ')}
                            </span>
                            <div className="flex-1 h-3 bg-gray-100 rounded-full relative overflow-hidden">
                              <div
                                className={`absolute top-0 h-full rounded-full transition-all duration-300 ${
                                  weight >= 0 ? 'bg-indigo-400' : 'bg-rose-400'
                                }`}
                                style={{
                                  width: `${pct}%`,
                                  [weight >= 0 ? 'left' : 'right']: '0',
                                }}
                              />
                            </div>
                            <span className={`w-10 text-[10px] font-mono text-right ${
                              weight >= 0 ? 'text-indigo-600' : 'text-rose-500'
                            }`}>
                              {weight >= 0 ? '+' : ''}{weight.toFixed(3)}
                            </span>
                          </div>
                        )
                      })}
                </div>
              </div>

              {/* Meta */}
              <div className="flex gap-2 text-[10px] text-gray-400">
                <span className="bg-gray-50 px-2 py-0.5 rounded-full">
                  {surrogateInfo.surrogate_type}
                </span>
                <span className="bg-gray-50 px-2 py-0.5 rounded-full">
                  {surrogateInfo.n_perturbations} perturbs
                </span>
                <span className="bg-gray-50 px-2 py-0.5 rounded-full">
                  {surrogateInfo.n_concepts} concepts
                </span>
              </div>
            </div>
          )}

          {surrogateInfo?.error && (
            <p className="text-xs text-red-500 bg-red-50 rounded-lg px-3 py-2">{surrogateInfo.error}</p>
          )}
        </div>
      )}
    </div>
  )
}
