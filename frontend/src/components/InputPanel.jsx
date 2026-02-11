import { useState } from 'react'

export default function InputPanel({ onPredict }) {
  const [inputType, setInputType] = useState('text')
  const [text, setText] = useState('')
  const [imageFile, setImageFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)

  const loadSample = async () => {
    const res = await fetch('/samples')
    const { samples } = await res.json()
    if (samples?.length) setText(samples[0])
  }

  const handleImageChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      setImageFile(file)
      setPreviewUrl(URL.createObjectURL(file))
    }
  }

  const handlePredict = async () => {
    setLoading(true)
    try {
      if (inputType === 'text') {
        await onPredict('text', text)
      } else {
        await onPredict('image', imageFile?.name ?? 'sample_xray.png')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-xl shadow p-5">
      <h2 className="text-lg font-semibold mb-3">Input</h2>

      {/* Mode selector */}
      <div className="flex gap-2 mb-4">
        {['text', 'image'].map((t) => (
          <button
            key={t}
            onClick={() => setInputType(t)}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium capitalize transition ${
              inputType === t
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Text mode */}
      {inputType === 'text' && (
        <div className="space-y-2">
          <textarea
            className="w-full border rounded-lg p-3 text-sm min-h-[120px] focus:ring-2 focus:ring-indigo-300 focus:outline-none"
            placeholder="Enter text to classify…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button
            onClick={loadSample}
            className="text-sm text-indigo-600 hover:underline"
          >
            Load sample text
          </button>
        </div>
      )}

      {/* Image mode */}
      {inputType === 'image' && (
        <div className="space-y-3">
          <label className="block">
            <span className="text-sm text-gray-600">Upload an image (or use placeholder)</span>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              className="mt-1 block w-full text-sm file:mr-4 file:py-1.5 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-indigo-50 file:text-indigo-600 hover:file:bg-indigo-100"
            />
          </label>
          {previewUrl && (
            <img
              src={previewUrl}
              alt="Preview"
              className="max-h-40 rounded-lg border"
            />
          )}
        </div>
      )}

      {/* Predict button */}
      <button
        onClick={handlePredict}
        disabled={loading || (inputType === 'text' && !text.trim())}
        className="mt-4 w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-300 text-white font-medium py-2 rounded-lg transition"
      >
        {loading ? 'Predicting…' : 'Predict'}
      </button>
    </div>
  )
}
