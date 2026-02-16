import { useState, useCallback, useEffect } from 'react'
import InputPanel from './components/InputPanel'
import PredictionPanel from './components/PredictionPanel'
import ConceptPanel from './components/ConceptPanel'
import ConceptBars from './components/ConceptBars'
import ExplanationPanel from './components/ExplanationPanel'
import AttributionPanel from './components/AttributionPanel'
import ChatAgent from './components/ChatAgent'
import ModelSelector from './components/ModelSelector'
import ConceptStrategySelector from './components/ConceptStrategySelector'
import SurrogatePanel from './components/SurrogatePanel'
import EditResultPanel from './components/EditResultPanel'

async function api(endpoint, body) {
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return res.json()
}

export default function App() {
  const [domains, setDomains] = useState([])
  const [allDomains, setAllDomains] = useState([])
  const [domain, setDomain] = useState('')
  const [lastData, setLastData] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [originalConcepts, setOriginalConcepts] = useState(null)
  const [concepts, setConcepts] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [attribution, setAttribution] = useState(null)
  const [editResult, setEditResult] = useState(null)
  const [selectedModel, setSelectedModel] = useState(null)
  const [attributionLoading, setAttributionLoading] = useState(false)
  const [attributionError, setAttributionError] = useState(null)
  const [showDomainSettings, setShowDomainSettings] = useState(false)
  const [conceptStrategy, setConceptStrategy] = useState('predefined')
  const [customConcepts, setCustomConcepts] = useState(null)
  const [surrogateInfo, setSurrogateInfo] = useState(null)
  const [conceptStrategyLabel, setConceptStrategyLabel] = useState('predefined')

  // ── Comparison slot (pinned snapshot for right column) ──
  const [compSlot, setCompSlot] = useState(null)

  // ── Fetch available domains on mount ──────────────────
  useEffect(() => {
    fetch('/domains')
      .then((r) => r.json())
      .then((d) => {
        setDomains(d.domains)
        if (d.domains.length && !domain) {
          setDomain(d.domains[0].name)
          const defaultModel = d.domains[0].models?.find((m) => m.default)
          if (defaultModel) setSelectedModel(defaultModel.id)
        }
      })
      .catch(() => {})
    fetch('/domains/all')
      .then((r) => r.json())
      .then((d) => setAllDomains(d.domains))
      .catch(() => {})
  }, [])

  const activeDomain = domains.find((d) => d.name === domain)
  const inputType = activeDomain?.input_type ?? 'text'

  // ── Predict ────────────────────────────────────────────
  const handlePredict = useCallback(async (data) => {
    setExplanation(null)
    setAttribution(null)
    setEditResult(null)
    setSurrogateInfo(null)
    setLastData(data)
    const body = { domain, data, model_id: selectedModel }
    if (conceptStrategy !== 'predefined') body.concept_strategy = conceptStrategy
    if (customConcepts) body.custom_concepts = customConcepts
    const result = await api('/predict', body)
    setPrediction({ label: result.label, confidence: result.confidence })
    setOriginalConcepts(result.concepts)
    setConcepts(result.concepts)
    setConceptStrategyLabel(conceptStrategy)
  }, [domain, selectedModel, conceptStrategy, customConcepts])

  // ── Counterfactual ────────────────────────────────────
  const handleCounterfactual = useCallback(async (modifiedConcepts) => {
    if (!originalConcepts || !prediction) return
    // Auto-pin current state to comparison before applying counterfactual
    if (!compSlot) {
      setCompSlot({
        prediction: { ...prediction },
        concepts: { ...concepts },
        strategyLabel: conceptStrategyLabel,
        title: 'Before counterfactual',
        attribution: attribution ? { ...attribution } : null,
      })
    }
    const result = await api('/counterfactual', {
      domain,
      original_concepts: originalConcepts,
      modified_concepts: modifiedConcepts,
      original_confidence: prediction.confidence,
      model_id: selectedModel,
    })
    setPrediction({ label: result.label, confidence: result.confidence })
    setConcepts(modifiedConcepts)
    setExplanation(null)
  }, [domain, originalConcepts, prediction, selectedModel, concepts, conceptStrategyLabel, attribution, compSlot])

  // ── Explain ───────────────────────────────────────────
  const handleExplain = useCallback(async (evidence) => {
    if (!prediction || !concepts) return
    const result = await api('/explain', {
      domain,
      label: prediction.label,
      confidence: prediction.confidence,
      concepts,
      evidence,
      model_id: selectedModel,
    })
    setExplanation(result)
  }, [domain, prediction, concepts, selectedModel])

  // ── Attribution ───────────────────────────────────────
  const handleAttribution = useCallback(async (conceptName) => {
    if (!lastData) {
      setAttributionError('No input data — please run a prediction first')
      return
    }
    setAttributionLoading(true)
    setAttributionError(null)
    setAttribution(null)
    try {
      const res = await fetch('/attribution', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          domain,
          data: lastData,
          concept: conceptName,
          model_id: selectedModel,
        }),
      })
      if (!res.ok) {
        throw new Error(`Server error ${res.status}`)
      }
      const result = await res.json()
      setAttribution(result)
    } catch (err) {
      console.error('Attribution error:', err)
      setAttributionError(err.message || 'Attribution request failed')
    } finally {
      setAttributionLoading(false)
    }
  }, [domain, lastData, selectedModel])

  // ── Chat ──────────────────────────────────────────────
  const handleChat = useCallback(async (message) => {
    const result = await api('/chat', {
      message,
      context: {
        domain,
        label: prediction?.label,
        confidence: prediction?.confidence,
        concepts,
      },
    })
    return result.reply
  }, [domain, prediction, concepts])

  // ── Input editing counterfactual ─────────────────────
  const handleEditInput = useCallback(async (editSpec) => {
    if (!lastData) return
    const result = await api('/edit_input', {
      domain,
      data: lastData,
      edit_spec: editSpec,
      model_id: selectedModel,
    })
    setEditResult(result)
    if (result.edited_prediction) {
      setPrediction(result.edited_prediction)
    }
  }, [domain, lastData, selectedModel])

  // ── Domain toggle ──────────────────────────────────────
  const handleDomainToggle = useCallback(async (domainName, enabled) => {
    await api('/domains/toggle', { domain: domainName, enabled })
    const [enabledRes, allRes] = await Promise.all([
      fetch('/domains').then((r) => r.json()),
      fetch('/domains/all').then((r) => r.json()),
    ])
    setDomains(enabledRes.domains)
    setAllDomains(allRes.domains)
    if (!enabled && domainName === domain && enabledRes.domains.length > 0) {
      handleDomainChange(enabledRes.domains[0].name)
    }
  }, [domain])

  // ── Concept strategy change ───────────────────────────
  const handleStrategyChange = useCallback(({ strategy, custom_concepts }) => {
    setConceptStrategy(strategy)
    setCustomConcepts(custom_concepts || null)
  }, [])

  // ── Fit local surrogate ───────────────────────────────
  const handleFitSurrogate = useCallback(async (surrogateType, nPerturbations) => {
    if (!lastData) return
    const result = await api('/local_surrogate', {
      domain,
      data: lastData,
      model_id: selectedModel,
      concept_strategy: conceptStrategy,
      surrogate_type: surrogateType,
      n_perturbations: nPerturbations,
    })
    setSurrogateInfo(result)
  }, [domain, lastData, selectedModel, conceptStrategy])

  // ── Reset concepts to original ────────────────────────
  const handleResetConcepts = useCallback(() => {
    if (originalConcepts) setConcepts({ ...originalConcepts })
  }, [originalConcepts])

  // ── Handle model change ───────────────────────────────
  const handleModelChange = useCallback((modelId, refresh) => {
    setSelectedModel(modelId)
    if (refresh) {
      fetch('/domains')
        .then((r) => r.json())
        .then((d) => setDomains(d.domains))
        .catch(() => {})
    }
  }, [])

  // ── Clear state on domain change ──────────────────────
  const handleDomainChange = useCallback((d) => {
    setDomain(d)
    setPrediction(null)
    setOriginalConcepts(null)
    setConcepts(null)
    setExplanation(null)
    setAttribution(null)
    setLastData(null)
    setEditResult(null)
    setAttributionError(null)
    setSurrogateInfo(null)
    setConceptStrategy('predefined')
    setCustomConcepts(null)
    setConceptStrategyLabel('predefined')
    setCompSlot(null)
    const domainObj = domains.find((dm) => dm.name === d)
    const defaultModel = domainObj?.models?.find((m) => m.default)
    setSelectedModel(defaultModel?.id ?? null)
  }, [domains])

  // ── Pin / clear comparison ────────────────────────────
  const handlePinToCompare = useCallback(() => {
    if (!prediction || !concepts) return
    setCompSlot({
      prediction: { ...prediction },
      concepts: { ...concepts },
      strategyLabel: conceptStrategyLabel,
      title: `${prediction.label} (${conceptStrategyLabel})`,
      attribution: attribution ? { ...attribution } : null,
    })
  }, [prediction, concepts, conceptStrategyLabel, attribution])

  const handleClearComparison = useCallback(() => {
    setCompSlot(null)
  }, [])

  const domainIcons = { medical: '🩺', toxicity: '🛡️', birds: '🐦', vision: '🔬' }

  return (
    <div className="min-h-screen bg-gray-50 dot-bg">
      {/* ── Compact header ────────────────────────────── */}
      <header className="sticky top-0 z-40 bg-gradient-to-r from-indigo-700 via-indigo-600 to-purple-700 text-white px-4 py-2 shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-white/15 flex items-center justify-center text-base">🔍</div>
            <h1 className="text-lg font-bold tracking-tight">Interpretable Wrapper</h1>
          </div>
          {/* Domain pills + settings */}
          {domains.length > 0 && (
            <div className="flex gap-1.5 items-center">
              {domains.map((d) => (
                <button
                  key={d.name}
                  onClick={() => handleDomainChange(d.name)}
                  className={`px-3 py-1 rounded-full text-xs font-semibold capitalize transition-all duration-200 ${
                    domain === d.name
                      ? 'bg-white text-indigo-700 shadow-md scale-105'
                      : 'bg-white/15 text-indigo-100 hover:bg-white/25'
                  }`}
                  title={d.description}
                >
                  <span className="mr-1">{domainIcons[d.name] || '📊'}</span>
                  {d.name}
                </button>
              ))}
              <div className="relative ml-1">
                <button
                  onClick={() => setShowDomainSettings(!showDomainSettings)}
                  className="w-7 h-7 rounded-full bg-white/15 hover:bg-white/25 flex items-center justify-center transition"
                  title="Domain settings"
                >
                  <svg className="w-4 h-4 text-indigo-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </button>
                {showDomainSettings && (
                  <div className="absolute right-0 top-10 w-72 bg-white rounded-xl shadow-xl border border-gray-100 p-4 z-50">
                    <h4 className="text-sm font-bold text-gray-800 mb-3">Domain Settings</h4>
                    <p className="text-xs text-gray-400 mb-3">Enable or disable domains</p>
                    <div className="space-y-2">
                      {allDomains.map((d) => (
                        <label key={d.name} className="flex items-center justify-between px-3 py-2 rounded-lg hover:bg-gray-50 cursor-pointer transition">
                          <div className="flex items-center gap-2">
                            <span>{domainIcons[d.name] || '📊'}</span>
                            <div>
                              <span className="text-sm font-medium text-gray-700 capitalize">{d.name}</span>
                              <p className="text-[10px] text-gray-400">{d.num_concepts} concepts</p>
                            </div>
                          </div>
                          <div
                            onClick={(e) => {
                              e.preventDefault()
                              handleDomainToggle(d.name, !d.enabled)
                            }}
                            className={`relative w-10 h-5 rounded-full transition cursor-pointer ${d.enabled ? 'bg-indigo-500' : 'bg-gray-200'}`}
                          >
                            <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${d.enabled ? 'translate-x-5' : 'translate-x-0.5'}`} />
                          </div>
                        </label>
                      ))}
                    </div>
                    <button
                      onClick={() => setShowDomainSettings(false)}
                      className="mt-3 w-full text-xs text-gray-500 hover:text-gray-700 py-1 font-medium"
                    >
                      Close
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </header>

      {/* ── 3-column layout ───────────────────────────── */}
      <div className="flex flex-col xl:flex-row gap-4 p-4 min-h-[calc(100vh-48px)]">

        {/* ─── LEFT SIDEBAR (controls) ─────────────────── */}
        <aside className="w-full xl:w-[300px] xl:min-w-[300px] xl:max-w-[300px] xl:sticky xl:top-[48px] xl:self-start xl:max-h-[calc(100vh-48px)] xl:overflow-y-auto styled-scrollbar space-y-3 sidebar-panels">
          <InputPanel
            domain={domain}
            inputType={inputType}
            onPredict={handlePredict}
            lastData={lastData}
            onEditInput={handleEditInput}
          />
          <ModelSelector
            models={activeDomain?.models}
            selectedModel={selectedModel}
            onModelChange={handleModelChange}
            domain={domain}
          />
          <ConceptStrategySelector
            domain={domain}
            onStrategyChange={handleStrategyChange}
            currentStrategy={conceptStrategy}
          />
          <SurrogatePanel
            surrogateInfo={surrogateInfo}
            onFitSurrogate={handleFitSurrogate}
            hasPrediction={!!prediction}
          />
          <ExplanationPanel
            explanation={explanation}
            onExplain={handleExplain}
            hasPrediction={!!prediction}
          />
          <ChatAgent onSend={handleChat} hasPrediction={!!prediction} />
        </aside>

        {/* ─── CENTER (primary results) ────────────────── */}
        <main className="flex-1 min-w-0 space-y-4">
          <PredictionPanel prediction={prediction} domain={domain} />

          <ConceptPanel
            concepts={concepts}
            domain={domain}
            onApply={handleCounterfactual}
            onReset={handleResetConcepts}
            onAttributionRequest={handleAttribution}
            activeConcept={attribution?.concept}
            strategyLabel={conceptStrategyLabel}
          />

          {/* Pin to compare button */}
          {prediction && concepts && (
            <div className="flex justify-center">
              <button
                onClick={handlePinToCompare}
                className="px-4 py-2 text-sm font-medium rounded-full bg-purple-600 text-white hover:bg-purple-700 transition shadow-sm flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                </svg>
                Pin to Compare
              </button>
            </div>
          )}

          <AttributionPanel
            attribution={attribution}
            inputType={inputType}
            loading={attributionLoading}
            error={attributionError}
          />

          {editResult && <EditResultPanel editResult={editResult} />}
        </main>

        {/* ─── RIGHT (comparison) ──────────────────────── */}
        <aside className="w-full xl:flex-1 min-w-0 space-y-4">
          {compSlot ? (
            <>
              {/* Comparison header */}
              <div className="flex items-center justify-between bg-purple-50 border border-purple-200 rounded-2xl px-4 py-2">
                <div className="flex items-center gap-2">
                  <span className="text-purple-600 text-base">📌</span>
                  <span className="text-sm font-semibold text-purple-800 truncate">
                    {compSlot.title}
                  </span>
                </div>
                <button
                  onClick={handleClearComparison}
                  className="text-xs text-purple-400 hover:text-purple-600 font-medium transition"
                >
                  Clear
                </button>
              </div>

              <PredictionPanel
                prediction={compSlot.prediction}
                domain={domain}
              />

              <ConceptBars
                concepts={compSlot.concepts}
                referenceConcepts={concepts}
                strategyLabel={compSlot.strategyLabel}
                title="Pinned Concepts"
              />

              {compSlot.attribution && (
                <AttributionPanel
                  attribution={compSlot.attribution}
                  inputType={inputType}
                  loading={false}
                  error={null}
                />
              )}
            </>
          ) : (
            /* Empty state */
            <div className="h-full flex items-center justify-center">
              <div className="text-center py-16 px-8">
                <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-purple-50 flex items-center justify-center">
                  <svg className="w-8 h-8 text-purple-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                  </svg>
                </div>
                <h3 className="text-sm font-semibold text-gray-500 mb-1">Comparison Column</h3>
                <p className="text-xs text-gray-400 max-w-[200px] mx-auto">
                  Click <strong>Pin to Compare</strong> to snapshot the current results here, then change strategy or apply a counterfactual to compare side by side.
                </p>
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  )
}
