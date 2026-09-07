import { useState, useEffect } from 'react'

function aucColor(auc) {
  if (auc >= 0.9) return 'bg-green-500'
  if (auc >= 0.8) return 'bg-yellow-400'
  return 'bg-red-400'
}

function aucTextColor(auc) {
  if (auc >= 0.9) return 'text-green-600'
  if (auc >= 0.8) return 'text-yellow-600'
  return 'text-red-500'
}

export default function ModelTrustPanel({ domain }) {
  const [metrics, setMetrics] = useState(null)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    setMetrics(null)
    if (!domain) return
    fetch(`/domains/${encodeURIComponent(domain)}/metrics`)
      .then((r) => r.json())
      .then((d) => setMetrics(d.available ? d : null))
      .catch(() => setMetrics(null))
  }, [domain])

  if (!metrics) return null

  const head = metrics.task_head || {}
  const surrogate = metrics.surrogate || {}
  const probes = metrics.probes || {}
  const probeEntries = Object.entries(probes)
    .filter(([, m]) => typeof m.test_auc === 'number')
    .sort((a, b) => b[1].test_auc - a[1].test_auc)

  return (
    <div className="glass-card rounded-2xl shadow-md p-5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between"
      >
        <h3 className="text-sm font-bold text-gray-700 flex items-center gap-2">
          <svg className="w-4 h-4 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Model Trust
          {typeof head.test_accuracy === 'number' && (
            <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded-full font-bold uppercase bg-emerald-100 text-emerald-700">
              acc {(head.test_accuracy * 100).toFixed(0)}%
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
        <div className="mt-3 space-y-4">
          {/* Dataset provenance */}
          <div className="text-[11px] text-gray-400">
            {metrics.dataset} · {metrics.n_train} train / {metrics.n_val} val / {metrics.n_test} test
          </div>

          {/* Task head */}
          <div>
            <p className="text-xs text-gray-500 font-semibold mb-1.5">Task Head (held-out test)</p>
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="bg-gray-50 rounded-lg py-2">
                <p className="text-base font-bold text-gray-800">{(head.test_accuracy * 100).toFixed(1)}%</p>
                <p className="text-[10px] text-gray-400">Accuracy</p>
              </div>
              <div className="bg-gray-50 rounded-lg py-2">
                <p className="text-base font-bold text-gray-800">{head.test_macro_auc?.toFixed(3)}</p>
                <p className="text-[10px] text-gray-400">Macro AUC</p>
              </div>
              <div className="bg-gray-50 rounded-lg py-2">
                <p className="text-base font-bold text-gray-800">{head.val_ece_after?.toFixed(3)}</p>
                <p className="text-[10px] text-gray-400">ECE (calibrated)</p>
              </div>
            </div>
            {head.test_per_class_auc && (
              <div className="flex gap-2 mt-2 flex-wrap">
                {Object.entries(head.test_per_class_auc).map(([cls, auc]) => (
                  <span key={cls} className="text-[10px] px-2 py-1 rounded-full bg-gray-100 text-gray-600">
                    {cls}: <span className={`font-bold ${aucTextColor(auc)}`}>{auc.toFixed(2)}</span>
                  </span>
                ))}
              </div>
            )}
          </div>

          {/* Surrogate fidelity */}
          {typeof surrogate.test_fidelity === 'number' && (
            <div>
              <p className="text-xs text-gray-500 font-semibold mb-1.5">
                Concept Surrogate Fidelity <span className="text-gray-400 font-normal">(agreement with black-box on held-out test)</span>
              </p>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${aucColor(surrogate.test_fidelity)}`}
                    style={{ width: `${surrogate.test_fidelity * 100}%` }}
                  />
                </div>
                <span className={`text-sm font-bold ${aucTextColor(surrogate.test_fidelity)}`}>
                  {(surrogate.test_fidelity * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-[10px] text-gray-400 mt-1">
                Ground-truth accuracy via concepts alone: {(surrogate.test_gt_accuracy * 100).toFixed(1)}%
              </p>
            </div>
          )}

          {/* Per-concept probe reliability */}
          {probeEntries.length > 0 && (
            <div>
              <p className="text-xs text-gray-500 font-semibold mb-1.5">Concept Probe Reliability (test AUC)</p>
              <div className="space-y-1 max-h-56 overflow-y-auto styled-scrollbar pr-1">
                {probeEntries.map(([name, m]) => (
                  <div key={name} className="flex items-center gap-2">
                    <span className="text-[11px] text-gray-600 w-36 truncate" title={name}>{name}</span>
                    <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${aucColor(m.test_auc)}`}
                        style={{ width: `${m.test_auc * 100}%` }}
                      />
                    </div>
                    <span className={`text-[11px] font-semibold w-9 text-right ${aucTextColor(m.test_auc)}`}>
                      {m.test_auc.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
