import { useState, useCallback } from 'react'
import InputPanel from './components/InputPanel'
import PredictionPanel from './components/PredictionPanel'
import ConceptPanel from './components/ConceptPanel'
import ExplanationPanel from './components/ExplanationPanel'
import ChatAgent from './components/ChatAgent'

async function api(endpoint, body) {
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return res.json()
}

export default function App() {
  const [inputType, setInputType] = useState('text')
  const [prediction, setPrediction] = useState(null)
  const [originalConcepts, setOriginalConcepts] = useState(null)
  const [concepts, setConcepts] = useState(null)
  const [explanation, setExplanation] = useState(null)

  // ── Predict ────────────────────────────────────────────
  const handlePredict = useCallback(async (type, data) => {
    setInputType(type)
    setExplanation(null)
    const result = await api('/predict', { input_type: type, data })
    setPrediction({ label: result.label, confidence: result.confidence })
    setOriginalConcepts(result.concepts)
    setConcepts(result.concepts)
  }, [])

  // ── Counterfactual ────────────────────────────────────
  const handleCounterfactual = useCallback(async (modifiedConcepts) => {
    if (!originalConcepts || !prediction) return
    const result = await api('/counterfactual', {
      input_type: inputType,
      original_concepts: originalConcepts,
      modified_concepts: modifiedConcepts,
      original_confidence: prediction.confidence,
    })
    setPrediction({ label: result.label, confidence: result.confidence })
    setConcepts(modifiedConcepts)
    setExplanation(null)
  }, [inputType, originalConcepts, prediction])

  // ── Explain ───────────────────────────────────────────
  const handleExplain = useCallback(async (evidence) => {
    if (!prediction || !concepts) return
    const result = await api('/explain', {
      input_type: inputType,
      label: prediction.label,
      confidence: prediction.confidence,
      concepts,
      evidence,
    })
    setExplanation(result)
  }, [inputType, prediction, concepts])

  // ── Chat ──────────────────────────────────────────────
  const handleChat = useCallback(async (message) => {
    const result = await api('/chat', {
      message,
      context: {
        label: prediction?.label,
        confidence: prediction?.confidence,
        concepts,
      },
    })
    return result.reply
  }, [prediction, concepts])

  // ── Reset concepts to original ────────────────────────
  const handleResetConcepts = useCallback(() => {
    if (originalConcepts) setConcepts({ ...originalConcepts })
  }, [originalConcepts])

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-indigo-700 text-white py-4 px-6 shadow-md">
        <h1 className="text-2xl font-bold">Interpretable Wrapper</h1>
        <p className="text-indigo-200 text-sm">Interactive Interpretability for Black-Box AI Models</p>
      </header>

      {/* Main grid */}
      <main className="max-w-7xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left column */}
        <div className="space-y-6">
          <InputPanel onPredict={handlePredict} />
          <ConceptPanel
            concepts={concepts}
            onApply={handleCounterfactual}
            onReset={handleResetConcepts}
          />
        </div>

        {/* Right column */}
        <div className="space-y-6">
          <PredictionPanel prediction={prediction} />
          <ExplanationPanel
            explanation={explanation}
            onExplain={handleExplain}
            hasPrediction={!!prediction}
          />
        </div>

        {/* Full-width bottom */}
        <div className="lg:col-span-2">
          <ChatAgent onSend={handleChat} hasPrediction={!!prediction} />
        </div>
      </main>
    </div>
  )
}
