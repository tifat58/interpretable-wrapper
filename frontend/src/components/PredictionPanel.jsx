export default function PredictionPanel({ prediction }) {
  if (!prediction) {
    return (
      <div className="bg-white rounded-xl shadow p-5">
        <h2 className="text-lg font-semibold mb-3">Prediction</h2>
        <p className="text-gray-400 italic text-sm">No prediction yet — submit an input to begin.</p>
      </div>
    )
  }

  const { label, confidence } = prediction
  const pct = Math.round(confidence * 100)

  // Color coding: red-ish for "toxic"/"pneumonia", green-ish for benign
  const isPositive = ['toxic', 'pneumonia'].includes(label)
  const badgeColor = isPositive ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
  const barColor = isPositive ? 'bg-red-500' : 'bg-green-500'

  return (
    <div className="bg-white rounded-xl shadow p-5">
      <h2 className="text-lg font-semibold mb-3">Prediction</h2>

      <div className="flex items-center gap-3 mb-4">
        <span className={`px-3 py-1 rounded-full text-sm font-semibold ${badgeColor}`}>
          {label}
        </span>
        <span className="text-gray-500 text-sm">{pct}% confidence</span>
      </div>

      {/* Confidence bar */}
      <div className="w-full bg-gray-200 rounded-full h-3">
        <div
          className={`h-3 rounded-full transition-all duration-500 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}