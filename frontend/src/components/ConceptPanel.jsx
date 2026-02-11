import { useState, useEffect } from 'react'

export default function ConceptPanel({ concepts, onApply, onReset }) {
  const [local, setLocal] = useState({})

  // Sync local state when concepts prop changes
  useEffect(() => {
    if (concepts) setLocal({ ...concepts })
  }, [concepts])

  if (!concepts) {
    return (
      <div className="bg-white rounded-xl shadow p-5">
        <h2 className="text-lg font-semibold mb-3">Concept Activations</h2>
        <p className="text-gray-400 italic text-sm">Run a prediction first to see concept activations.</p>
      </div>
    )
  }

  const handleSlider = (concept, value) => {
    setLocal((prev) => ({ ...prev, [concept]: parseFloat(value) }))
  }

  return (
    <div className="bg-white rounded-xl shadow p-5">
      <h2 className="text-lg font-semibold mb-4">Concept Activations</h2>

      <div className="space-y-4">
        {Object.entries(local).map(([name, value]) => (
          <div key={name}>
            <div className="flex justify-between text-sm mb-1">
              <span className="font-medium capitalize">{name.replace(/_/g, ' ')}</span>
              <span className="text-gray-500">{(value * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={value}
              onChange={(e) => handleSlider(name, e.target.value)}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
            />
          </div>
        ))}
      </div>

      <div className="flex gap-3 mt-5">
        <button
          onClick={() => onApply(local)}
          className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-2 rounded-lg transition"
        >
          Apply (Counterfactual)
        </button>
        <button
          onClick={() => {
            onReset()
            if (concepts) setLocal({ ...concepts })
          }}
          className="px-4 py-2 border border-gray-300 text-gray-600 rounded-lg hover:bg-gray-50 transition"
        >
          Reset
        </button>
      </div>
    </div>
  )
}
