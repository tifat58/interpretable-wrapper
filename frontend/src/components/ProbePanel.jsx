import { useState, useEffect, useCallback } from 'react'

// Local fetch helpers (probe panel is self-contained).
async function getJSON(url) {
  const res = await fetch(url)
  return res.json()
}
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  })
  return res.json()
}

function Gauge({ label, value, color }) {
  const pct = Math.round((value || 0) * 100)
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="font-medium text-gray-600">{label}</span>
        <span className="text-gray-500">{pct}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2.5">
        <div
          className={`h-2.5 rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export default function ProbePanel() {
  const [state, setState] = useState(null)
  const [result, setResult] = useState(null) // retrain before/after
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async () => {
    const s = await getJSON('/probe_state')
    setState(s)
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const handleCorrect = async () => {
    setBusy(true)
    await postJSON('/probe_feedback', { concept: state.concept })
    await refresh()
    setBusy(false)
  }

  const handleRetrain = async () => {
    setBusy(true)
    const r = await postJSON('/probe_retrain', { concept: state?.concept })
    setResult(r.error ? null : r)
    await refresh()
    setBusy(false)
  }

  const handleReset = async () => {
    setBusy(true)
    setResult(null)
    const s = await postJSON('/probe_reset', {})
    setState(s)
    setBusy(false)
  }

  if (!state) {
    return (
      <div className="bg-white rounded-xl shadow p-5">
        <h2 className="text-lg font-semibold mb-3">Concept Probe Update</h2>
        <p className="text-gray-400 italic text-sm">Connecting to probe service…</p>
      </div>
    )
  }

  const fbCount = state.feedback_count
  const preview = (result ? result.after : state).validation_preview || []
  const beforePreview = result ? result.before.validation_preview : null

  return (
    <div className="bg-white rounded-xl shadow p-5">
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg font-semibold">Concept Probe Update</h2>
        <span className="text-xs px-2 py-0.5 bg-amber-100 text-amber-700 rounded-full">
          actionable feedback
        </span>
      </div>
      <p className="text-xs text-gray-500 mb-4">
        The <span className="font-semibold capitalize">{state.concept}</span> probe ships
        miscalibrated. Correct a few readings, then update the probe and watch fidelity recover.
      </p>

      {/* Current example correction */}
      <div className="border border-gray-200 rounded-lg p-3 mb-4">
        <div className="text-xs text-gray-500 mb-1">
          Held-out example #{state.current_example}
        </div>
        <div className="flex items-center justify-between text-sm">
          <span>
            Probe reads <span className="font-medium capitalize">{state.concept}</span> ={' '}
            <span className="font-mono text-red-600">
              {(state.estimated_concepts[state.concept] * 100).toFixed(0)}%
            </span>
          </span>
          <span className="text-gray-400">expert: </span>
          <span className="font-mono text-green-600">
            {(state.expert_value * 100).toFixed(0)}%
          </span>
        </div>
        <button
          onClick={handleCorrect}
          disabled={busy}
          className="mt-3 w-full bg-amber-500 hover:bg-amber-600 disabled:opacity-50 text-white text-sm font-medium py-2 rounded-lg transition"
        >
          ✓ Submit expert correction
        </button>
      </div>

      {/* Live metrics */}
      <div className="space-y-3 mb-4">
        <Gauge label="Surrogate fidelity (vs black box)" value={state.fidelity} color="bg-indigo-600" />
        <Gauge label={`${state.concept} probe accuracy`} value={state.concept_accuracy} color="bg-teal-500" />
      </div>

      <div className="flex items-center justify-between mb-4">
        <span className="text-sm text-gray-600">
          Corrections collected: <span className="font-semibold">{fbCount}</span>
        </span>
        <div className="flex gap-2">
          <button
            onClick={handleRetrain}
            disabled={busy || fbCount === 0}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-sm font-medium px-4 py-2 rounded-lg transition"
          >
            ⟳ Update probe
          </button>
          <button
            onClick={handleReset}
            disabled={busy}
            className="px-3 py-2 border border-gray-300 text-gray-600 text-sm rounded-lg hover:bg-gray-50 transition"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Retrain result: before/after headline */}
      {result && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-4 text-sm">
          <div className="font-semibold text-green-800 mb-1">
            Probe updated from {result.feedback_used} corrections
          </div>
          <div className="grid grid-cols-2 gap-2 text-green-900">
            <div>
              Fidelity:{' '}
              <span className="font-mono">{(result.before.fidelity * 100).toFixed(0)}%</span>
              {' → '}
              <span className="font-mono font-bold">{(result.after.fidelity * 100).toFixed(0)}%</span>
              <span className="text-green-600"> (+{(result.fidelity_gain * 100).toFixed(0)})</span>
            </div>
            <div>
              Concept acc:{' '}
              <span className="font-mono">{(result.before.concept_accuracy * 100).toFixed(0)}%</span>
              {' → '}
              <span className="font-mono font-bold">{(result.after.concept_accuracy * 100).toFixed(0)}%</span>
              <span className="text-green-600"> (+{(result.accuracy_gain * 100).toFixed(0)})</span>
            </div>
          </div>
        </div>
      )}

      {/* Generalization preview: held-out estimate vs truth */}
      <div>
        <div className="text-xs font-medium text-gray-600 mb-2">
          Generalization to other held-out examples (estimate vs expert truth)
        </div>
        <div className="space-y-1.5">
          {preview.map((p, i) => {
            const before = beforePreview ? beforePreview[i] : null
            const err = Math.abs(p.estimate - p.truth)
            const good = err < 0.15
            return (
              <div key={p.id} className="flex items-center gap-2 text-xs">
                <span className="w-8 text-gray-400">#{p.id}</span>
                <div className="flex-1 relative h-3 bg-gray-100 rounded">
                  {/* truth marker */}
                  <div
                    className="absolute top-0 h-3 w-0.5 bg-green-600"
                    style={{ left: `${p.truth * 100}%` }}
                    title={`expert ${(p.truth * 100).toFixed(0)}%`}
                  />
                  {/* estimate dot */}
                  <div
                    className={`absolute top-0.5 h-2 w-2 rounded-full ${good ? 'bg-teal-500' : 'bg-red-500'}`}
                    style={{ left: `calc(${p.estimate * 100}% - 4px)` }}
                    title={`estimate ${(p.estimate * 100).toFixed(0)}%`}
                  />
                </div>
                {before && (
                  <span className="w-16 text-right text-gray-400">
                    err {(Math.abs(before.estimate - before.truth) * 100).toFixed(0)}→{(err * 100).toFixed(0)}
                  </span>
                )}
              </div>
            )
          })}
        </div>
        <div className="flex gap-4 mt-2 text-[10px] text-gray-400">
          <span><span className="inline-block w-2 h-2 bg-green-600 mr-1 align-middle" />expert truth</span>
          <span><span className="inline-block w-2 h-2 rounded-full bg-teal-500 mr-1 align-middle" />accurate estimate</span>
          <span><span className="inline-block w-2 h-2 rounded-full bg-red-500 mr-1 align-middle" />misread</span>
        </div>
      </div>
    </div>
  )
}
