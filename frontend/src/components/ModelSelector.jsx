import { useState } from 'react'

export default function ModelSelector({ models, selectedModel, onModelChange, domain }) {
  const [showRegister, setShowRegister] = useState(false)
  const [regForm, setRegForm] = useState({ id: '', label: '', model_name: '', model_class: '' })
  const [regError, setRegError] = useState('')

  if (!models || models.length === 0) return null

  const handleRegister = async () => {
    setRegError('')
    if (!regForm.id || !regForm.label || !regForm.model_name || !regForm.model_class) {
      setRegError('All fields are required.')
      return
    }
    try {
      const res = await fetch('/models/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ domain, ...regForm }),
      })
      const data = await res.json()
      if (!res.ok) {
        setRegError(data.error || 'Registration failed.')
        return
      }
      setShowRegister(false)
      setRegForm({ id: '', label: '', model_name: '', model_class: '' })
      onModelChange(regForm.id, true)
    } catch {
      setRegError('Network error.')
    }
  }

  return (
    <div className="glass-card rounded-2xl shadow-md p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-gray-700 flex items-center gap-2">
          <svg className="w-4 h-4 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
          </svg>
          Model
        </h3>
        <button
          onClick={() => setShowRegister(!showRegister)}
          className="text-xs text-indigo-600 hover:text-indigo-800 font-semibold transition"
        >
          {showRegister ? 'Cancel' : '+ Custom'}
        </button>
      </div>

      <div className="space-y-1">
        {models.map((m) => (
          <label
            key={m.id}
            className={`flex items-center gap-2.5 px-3 py-2 rounded-xl cursor-pointer text-sm transition-all duration-150 ${
              selectedModel === m.id
                ? 'bg-indigo-50/80 text-indigo-700 font-semibold ring-1 ring-indigo-200'
                : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            <input
              type="radio"
              name="model"
              value={m.id}
              checked={selectedModel === m.id}
              onChange={() => onModelChange(m.id)}
              className="accent-indigo-600"
            />
            <span>{m.label || m.name}</span>
            {m.default && (
              <span className="ml-auto text-[9px] bg-indigo-100 text-indigo-600 px-1.5 py-0.5 rounded-full font-bold uppercase">
                default
              </span>
            )}
          </label>
        ))}
      </div>

      {showRegister && (
        <div className="mt-3 border-t border-gray-100 pt-3 space-y-2">
          <p className="text-xs text-gray-500 font-semibold">Register a custom model</p>
          <input
            className="w-full text-xs border border-gray-200 rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-indigo-300 focus:outline-none"
            placeholder="Model ID (e.g. my-bert)"
            value={regForm.id}
            onChange={(e) => setRegForm({ ...regForm, id: e.target.value })}
          />
          <input
            className="w-full text-xs border border-gray-200 rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-indigo-300 focus:outline-none"
            placeholder="Display label"
            value={regForm.label}
            onChange={(e) => setRegForm({ ...regForm, label: e.target.value })}
          />
          <input
            className="w-full text-xs border border-gray-200 rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-indigo-300 focus:outline-none"
            placeholder="HuggingFace model name"
            value={regForm.model_name}
            onChange={(e) => setRegForm({ ...regForm, model_name: e.target.value })}
          />
          <select
            className="w-full text-xs border border-gray-200 rounded-lg px-3 py-1.5 focus:ring-2 focus:ring-indigo-300 focus:outline-none"
            value={regForm.model_class}
            onChange={(e) => setRegForm({ ...regForm, model_class: e.target.value })}
          >
            <option value="">Select model class…</option>
            <option value="AutoModelForSequenceClassification">AutoModelForSequenceClassification</option>
            <option value="AutoModelForImageClassification">AutoModelForImageClassification</option>
            <option value="CLIPModel">CLIPModel</option>
          </select>
          {regError && <p className="text-xs text-red-500 font-medium">{regError}</p>}
          <button
            onClick={handleRegister}
            className="w-full text-xs bg-gradient-to-r from-indigo-600 to-indigo-700 text-white rounded-lg py-2 font-semibold hover:from-indigo-700 hover:to-indigo-800 transition-all"
          >
            Register
          </button>
        </div>
      )}
    </div>
  )
}
