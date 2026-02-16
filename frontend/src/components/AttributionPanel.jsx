import { useState } from 'react'

export default function AttributionPanel({ attribution, inputType, loading, error }) {
  const [opacity, setOpacity] = useState(0.5)

  if (loading) {
    return (
      <div className="glass-card rounded-2xl shadow-md p-6 animate-fade-in">
        <h2 className="text-lg font-bold text-gray-800 mb-3 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
          Saliency Map
        </h2>
        <div className="flex items-center gap-2 text-gray-400 text-sm">
          <svg className="animate-spin h-5 w-5 text-indigo-500" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Computing saliency…
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="glass-card rounded-2xl shadow-md p-6 animate-fade-in">
        <h2 className="text-lg font-bold text-gray-800 mb-3">Saliency Map</h2>
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-red-600 text-sm font-medium">Error: {error}</p>
        </div>
      </div>
    )
  }

  if (!attribution || !attribution.data) {
    return (
      <div className="glass-card rounded-2xl shadow-md p-6 animate-fade-in">
        <h2 className="text-lg font-bold text-gray-800 mb-3">Saliency Map</h2>
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="w-12 h-12 rounded-full bg-indigo-50 flex items-center justify-center mb-3">
            <svg className="w-6 h-6 text-indigo-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
          <p className="text-gray-400 italic text-sm">
            Click a concept name to see which parts of the input matter most.
          </p>
        </div>
      </div>
    )
  }

  const { method, type, data, overlay, original_image, heatmap_colored, concept } = attribution
  const conceptLabel = concept ? concept.replace(/_/g, ' ') : ''
  const hasLayeredData = original_image && heatmap_colored
  const opacityPct = Math.round(opacity * 100)

  return (
    <div className="glass-card rounded-2xl shadow-md p-6 animate-fade-in">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-bold text-gray-800">Saliency Map</h2>
        <span className="text-[10px] font-bold text-indigo-500 bg-indigo-50 px-2 py-0.5 rounded-full uppercase tracking-wider">
          {method}
        </span>
      </div>

      {concept && (
        <div className="flex items-center gap-2 mb-4">
          <span className="text-sm text-gray-500">Concept:</span>
          <span className="inline-flex items-center px-3 py-0.5 rounded-full text-xs font-bold bg-gradient-to-r from-indigo-100 to-purple-100 text-indigo-700 capitalize">
            {conceptLabel}
          </span>
        </div>
      )}

      {/* ─── Token‑level highlights (text / toxicity) ─── */}
      {type === 'tokens' && Array.isArray(data) && (
        <div className="space-y-3">
          <div className="flex flex-wrap gap-1 p-4 bg-gray-50/80 rounded-xl text-sm leading-relaxed border border-gray-100">
            {data.map((item, i) => {
              const score = item.score ?? 0
              const absScore = Math.min(Math.abs(score), 1)
              const bg = score >= 0
                ? `rgba(239, 68, 68, ${absScore * 0.7})`
                : `rgba(59, 130, 246, ${absScore * 0.7})`
              const textColor = absScore > 0.5 ? 'white' : 'inherit'
              return (
                <span
                  key={i}
                  className="px-1.5 py-0.5 rounded-md cursor-default transition-all hover:scale-105"
                  style={{ backgroundColor: bg, color: textColor }}
                  title={`${item.token}: ${(score * 100).toFixed(1)}%`}
                >
                  {item.token}
                </span>
              )
            })}
          </div>
          <div className="flex items-center gap-4 text-xs text-gray-400 px-1">
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: 'rgba(239, 68, 68, 0.6)' }} />
              High relevance
            </div>
            <div className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: 'rgba(239, 68, 68, 0.15)' }} />
              Low relevance
            </div>
          </div>
        </div>
      )}

      {/* ─── Heatmap overlay with opacity slider ─── */}
      {type === 'heatmap' && (
        <div className="space-y-4">
          {/* Layered compositing: original + heatmap */}
          {hasLayeredData ? (
            <div className="flex flex-col items-center">
              <div className="relative inline-block rounded-xl overflow-hidden shadow-lg border border-gray-200">
                <img
                  src={`data:image/png;base64,${original_image}`}
                  alt="Original"
                  className="block max-h-72 w-auto"
                  draggable={false}
                />
                <img
                  src={`data:image/png;base64,${heatmap_colored}`}
                  alt="Heatmap overlay"
                  className="absolute inset-0 w-full h-full object-cover pointer-events-none"
                  style={{ opacity }}
                  draggable={false}
                />
              </div>

              {/* Opacity slider */}
              <div className="w-full max-w-xs mt-4 space-y-1">
                <div className="flex items-center justify-between text-[11px] font-semibold text-gray-500 px-0.5">
                  <span>Original</span>
                  <span className="text-indigo-600">{opacityPct}%</span>
                  <span>Heatmap</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={opacity}
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  className="w-full slider-styled appearance-none bg-transparent cursor-pointer"
                />
              </div>
            </div>
          ) : (
            /* Fallback: single pre-blended overlay */
            <div className="flex justify-center">
              <img
                src={`data:image/png;base64,${overlay || data}`}
                alt={`Saliency overlay for ${conceptLabel}`}
                className="max-h-72 rounded-xl border shadow-lg"
              />
            </div>
          )}

          {/* Color legend */}
          <div className="flex items-center justify-center gap-2 text-[11px] text-gray-400 font-medium">
            <span>low</span>
            <div className="w-24 h-2.5 rounded-full bg-gradient-to-r from-blue-500 via-green-400 via-yellow-400 to-red-500 shadow-inner" />
            <span>high</span>
          </div>
        </div>
      )}

      {/* Fallback */}
      {type !== 'tokens' && type !== 'heatmap' && (
        <p className="text-gray-400 text-sm italic">
          Attribution type "{type}" is not yet supported in the UI.
        </p>
      )}
    </div>
  )
}
