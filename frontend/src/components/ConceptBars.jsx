/**
 * ConceptBars — read-only concept activation bars used in the comparison column.
 * Shows each concept as a horizontal bar with optional "reference" marker line
 * from the primary (center) column for side-by-side visual diff.
 *
 * Props:
 *   concepts        – { name: value } map of activations (0-1)
 *   referenceConcepts – optional { name: value } from the other column (shown as marker lines)
 *   strategyLabel   – badge label (e.g. "pca", "clip")
 *   title           – optional header override
 */
export default function ConceptBars({
  concepts,
  referenceConcepts,
  strategyLabel,
  title,
}) {
  if (!concepts) return null

  const entries = Object.entries(concepts).sort((a, b) => b[1] - a[1])

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-base">📊</span>
          <h3 className="font-semibold text-gray-800 text-sm">
            {title || 'Concepts'}
          </h3>
          {strategyLabel && strategyLabel !== 'predefined' && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-purple-100 text-purple-600 font-medium">
              {strategyLabel}
            </span>
          )}
        </div>
        <span className="text-xs text-gray-400">{entries.length} concepts</span>
      </div>

      {/* Legend (only if reference exists) */}
      {referenceConcepts && (
        <div className="px-4 pt-2 flex gap-4 text-[10px] text-gray-500">
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-2 rounded-sm bg-purple-400" />
            Pinned
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-0.5 bg-indigo-500" />
            Current
          </span>
        </div>
      )}

      {/* Bars */}
      <div className="px-4 py-3 space-y-1.5 max-h-[420px] overflow-y-auto styled-scrollbar">
        {entries.map(([name, value]) => {
          const refVal = referenceConcepts?.[name]
          const pct = Math.round(value * 100)
          return (
            <div key={name} className="group">
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-xs text-gray-600 truncate max-w-[60%]">
                  {name.replace(/_/g, ' ')}
                </span>
                <span className="text-[10px] text-gray-400 tabular-nums">
                  {pct}%
                  {refVal != null && (
                    <span
                      className={`ml-1 ${
                        value > refVal
                          ? 'text-emerald-500'
                          : value < refVal
                          ? 'text-rose-500'
                          : 'text-gray-300'
                      }`}
                    >
                      {value > refVal ? '▲' : value < refVal ? '▼' : '='}
                    </span>
                  )}
                </span>
              </div>
              <div className="relative h-2 bg-gray-100 rounded-full overflow-hidden">
                {/* Main bar */}
                <div
                  className="absolute inset-y-0 left-0 rounded-full bg-purple-400 transition-all duration-300"
                  style={{ width: `${pct}%` }}
                />
                {/* Reference marker line */}
                {refVal != null && (
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-indigo-600"
                    style={{ left: `${Math.round(refVal * 100)}%` }}
                    title={`Current: ${Math.round(refVal * 100)}%`}
                  />
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
