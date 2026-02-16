export default function EditResultPanel({ editResult }) {
  if (!editResult) return null

  const original = editResult.original_prediction
  const edited = editResult.edited_prediction
  const conceptDeltas = editResult.concept_deltas || {}

  const changedConcepts = Object.entries(conceptDeltas)
    .filter(([, delta]) => Math.abs(delta) > 0.01)
    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))

  const confDelta = edited ? (edited.confidence - (original?.confidence || 0)) : 0

  return (
    <div className="glass-card rounded-2xl shadow-md p-5">
      <h3 className="text-sm font-bold text-gray-700 flex items-center gap-2 mb-3">
        <svg className="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
        </svg>
        Edit Comparison
      </h3>

      {/* Before/After predictions */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        {/* Original */}
        <div className="bg-gray-50 rounded-xl p-3 text-center">
          <p className="text-[10px] text-gray-400 font-semibold uppercase mb-1">Original</p>
          {original ? (
            <>
              <p className="text-sm font-bold text-gray-700 capitalize">{original.label}</p>
              <p className="text-xs text-gray-500 font-mono">{(original.confidence * 100).toFixed(1)}%</p>
            </>
          ) : (
            <p className="text-xs text-gray-400">N/A</p>
          )}
        </div>

        {/* Edited */}
        <div className="bg-emerald-50 rounded-xl p-3 text-center">
          <p className="text-[10px] text-emerald-600 font-semibold uppercase mb-1">Edited</p>
          {edited ? (
            <>
              <p className="text-sm font-bold text-emerald-700 capitalize">{edited.label}</p>
              <div className="flex items-center justify-center gap-1">
                <span className="text-xs text-emerald-600 font-mono">{(edited.confidence * 100).toFixed(1)}%</span>
                {Math.abs(confDelta) > 0.001 && (
                  <span className={`text-[10px] font-bold ${confDelta >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {confDelta >= 0 ? '▲' : '▼'}{Math.abs(confDelta * 100).toFixed(1)}
                  </span>
                )}
              </div>
            </>
          ) : (
            <p className="text-xs text-gray-400">N/A</p>
          )}
        </div>
      </div>

      {/* Label change indicator */}
      {original && edited && original.label !== edited.label && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mb-3 flex items-center gap-2">
          <span className="text-amber-500">⚡</span>
          <p className="text-xs text-amber-700 font-medium">
            Label changed: <span className="font-bold capitalize">{original.label}</span> → <span className="font-bold capitalize">{edited.label}</span>
          </p>
        </div>
      )}

      {/* Concept deltas */}
      {changedConcepts.length > 0 && (
        <div>
          <p className="text-xs text-gray-500 font-semibold mb-2">Concept Changes</p>
          <div className="space-y-1 max-h-40 overflow-y-auto pr-1">
            {changedConcepts.slice(0, 10).map(([name, delta]) => (
              <div key={name} className="flex items-center justify-between px-2 py-1 rounded-lg bg-gray-50">
                <span className="text-[11px] text-gray-600 truncate mr-2" title={name}>
                  {name.replace(/_/g, ' ')}
                </span>
                <span className={`text-[11px] font-mono font-bold ${
                  delta >= 0 ? 'text-green-600' : 'text-red-500'
                }`}>
                  {delta >= 0 ? '+' : ''}{delta.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Edit description if provided */}
      {editResult.edit_description && (
        <p className="text-[10px] text-gray-400 mt-3 italic">{editResult.edit_description}</p>
      )}
    </div>
  )
}
