import { useState, useEffect, useMemo } from 'react'

export default function ConceptPanel({ concepts, domain, onApply, onReset, onAttributionRequest, activeConcept, strategyLabel }) {
  const [local, setLocal] = useState({})
  const [search, setSearch] = useState('')
  const [sortBy, setSortBy] = useState('name')
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    if (concepts) setLocal({ ...concepts })
  }, [concepts])

  const entries = useMemo(() => {
    if (!local) return []
    let items = Object.entries(local)
    if (search.trim()) {
      const q = search.toLowerCase()
      items = items.filter(([name]) => name.toLowerCase().includes(q))
    }
    if (sortBy === 'activation') {
      items.sort((a, b) => b[1] - a[1])
    } else {
      items.sort((a, b) => a[0].localeCompare(b[0]))
    }
    return items
  }, [local, search, sortBy])

  const totalConcepts = concepts ? Object.keys(concepts).length : 0

  if (!concepts) {
    return (
      <div className="glass-card rounded-2xl shadow-md p-6">
        <h2 className="text-lg font-bold text-gray-800 mb-3">Concept Activations</h2>
        <div className="flex flex-col items-center justify-center py-6 text-center">
          <div className="w-12 h-12 rounded-full bg-indigo-50 flex items-center justify-center mb-3">
            <svg className="w-6 h-6 text-indigo-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          </div>
          <p className="text-gray-400 italic text-sm">Run a prediction first to see concept activations.</p>
        </div>
      </div>
    )
  }

  const handleSlider = (concept, value) => {
    setLocal((prev) => ({ ...prev, [concept]: parseFloat(value) }))
  }

  return (
    <div className="glass-card rounded-2xl shadow-md p-6">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-bold text-gray-800">
          Concept Activations
          <span className="text-sm font-normal text-gray-400 ml-2">({totalConcepts})</span>
          {strategyLabel && strategyLabel !== 'predefined' && (
            <span className="ml-2 text-[10px] bg-purple-100 text-purple-600 px-1.5 py-0.5 rounded-full font-bold uppercase">
              {strategyLabel}
            </span>
          )}
        </h2>
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="text-xs text-gray-400 hover:text-gray-600 font-medium transition"
        >
          {collapsed ? '▼ Expand' : '▲ Collapse'}
        </button>
      </div>

      {!collapsed && (
        <>
          {totalConcepts > 12 && (
            <div className="flex gap-2 mb-3">
              <div className="relative flex-1">
                <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search concepts…"
                  className="w-full border border-gray-200 rounded-xl pl-9 pr-3 py-2 text-sm focus:ring-2 focus:ring-indigo-300 focus:border-indigo-300 focus:outline-none bg-gray-50/50 transition"
                />
              </div>
              <button
                onClick={() => setSortBy(sortBy === 'name' ? 'activation' : 'name')}
                className="px-3 py-2 text-xs font-semibold border border-gray-200 rounded-xl hover:bg-gray-50 transition"
                title={`Sort by ${sortBy === 'name' ? 'activation' : 'name'}`}
              >
                {sortBy === 'name' ? 'A→Z' : '↓ val'}
              </button>
            </div>
          )}

          <p className="text-[11px] text-gray-400 mb-3 font-medium">Click a concept name to see its saliency map</p>

          <div className={`space-y-2.5 styled-scrollbar ${totalConcepts > 12 ? 'max-h-[400px] overflow-y-auto pr-1' : ''}`}>
            {entries.map(([name, value]) => {
              const isActive = activeConcept === name
              const pct = Math.round(value * 100)
              return (
                <div key={name} className={`group rounded-xl transition-all duration-200 ${isActive ? 'bg-indigo-50/80 ring-1 ring-indigo-300 p-3 -mx-1' : 'p-1'}`}>
                  <div className="flex items-center gap-2 text-sm mb-1.5">
                    <button
                      onClick={() => onAttributionRequest && onAttributionRequest(name)}
                      className={`font-medium capitalize flex-1 truncate text-left transition ${
                        isActive
                          ? 'text-indigo-700 font-bold'
                          : 'text-gray-700 hover:text-indigo-600 cursor-pointer'
                      }`}
                      title={`Show saliency for "${name}"`}
                    >
                      {name.replace(/_/g, ' ')}
                    </button>
                    {/* Mini activation bar */}
                    <div className="w-16 h-1.5 bg-gray-100 rounded-full overflow-hidden flex-shrink-0">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-indigo-300 to-indigo-500 transition-all duration-300"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400 w-10 text-right font-mono flex-shrink-0">{pct}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={value}
                    onChange={(e) => handleSlider(name, e.target.value)}
                    className="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                  />
                </div>
              )
            })}
            {entries.length === 0 && search && (
              <p className="text-gray-400 text-sm italic py-2">No concepts matching "{search}"</p>
            )}
          </div>

          <div className="flex gap-3 mt-5">
            <button
              onClick={() => onApply(local)}
              className="flex-1 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 text-white font-semibold py-2.5 rounded-xl transition-all shadow-sm hover:shadow-md"
            >
              Apply (Counterfactual)
            </button>
            <button
              onClick={() => {
                onReset()
                if (concepts) setLocal({ ...concepts })
              }}
              className="px-4 py-2.5 border border-gray-200 text-gray-600 rounded-xl hover:bg-gray-50 transition font-medium"
            >
              Reset
            </button>
          </div>
        </>
      )}
    </div>
  )
}
