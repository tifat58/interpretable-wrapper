export default function PredictionPanel({ prediction, domain }) {
  if (!prediction) {
    return (
      <div className="glass-card rounded-2xl shadow-md p-6">
        <h2 className="text-lg font-bold text-gray-800 mb-3">Prediction</h2>
        <div className="flex flex-col items-center justify-center py-6 text-center">
          <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mb-3">
            <svg className="w-6 h-6 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <p className="text-gray-400 italic text-sm">No prediction yet — submit an input to begin.</p>
        </div>
      </div>
    )
  }

  const { label, confidence } = prediction
  const pct = Math.round(confidence * 100)

  const negativeLabels = ['toxic', 'pneumonia', 'covid-19']
  const isNegative = negativeLabels.includes(label.toLowerCase())
  const badgeColor = isNegative ? 'bg-red-50 text-red-700 border-red-200' : 'bg-emerald-50 text-emerald-700 border-emerald-200'
  const barColor = isNegative ? 'from-red-400 to-red-500' : 'from-emerald-400 to-emerald-500'
  const barBg = isNegative ? 'bg-red-100' : 'bg-emerald-100'

  return (
    <div className="glass-card rounded-2xl shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-gray-800">Prediction</h2>
        {domain && (
          <span className="text-[10px] font-bold uppercase tracking-wider text-gray-400">
            {domain}
          </span>
        )}
      </div>

      <div className="flex items-center gap-3 mb-4">
        <span className={`px-4 py-1.5 rounded-full text-sm font-bold border ${badgeColor} capitalize`}>
          {label}
        </span>
      </div>

      {/* Confidence bar */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs font-semibold">
          <span className="text-gray-500">Confidence</span>
          <span className={isNegative ? 'text-red-600' : 'text-emerald-600'}>{pct}%</span>
        </div>
        <div className={`w-full ${barBg} rounded-full h-3 overflow-hidden`}>
          <div
            className={`h-3 rounded-full bg-gradient-to-r ${barColor} transition-all duration-700 ease-out`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    </div>
  )
}