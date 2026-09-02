import { useState, useRef, useCallback } from 'react'

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      // strip "data:image/...;base64," prefix — backend expects raw b64
      const b64 = reader.result.split(',')[1]
      resolve(b64)
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

export default function InputPanel({ domain, inputType, onPredict, lastData, onEditInput }) {
  const [text, setText] = useState('')
  const [imageFile, setImageFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  // Input editing state
  const [editMode, setEditMode] = useState(false)
  const [maskedIndices, setMaskedIndices] = useState(new Set())
  const [editLoading, setEditLoading] = useState(false)

  // Image editing state
  const [imgEditMode, setImgEditMode] = useState(false)
  const [maskRegion, setMaskRegion] = useState(null) // {x, y, w, h} in fractions 0-1
  const [dragging, setDragging] = useState(false)
  const [dragStart, setDragStart] = useState(null)
  const imgEditRef = useRef(null)

  const loadSample = async () => {
    const res = await fetch(`/samples?domain=${encodeURIComponent(domain)}`)
    const { samples, type } = await res.json()
    if (!samples?.length) return
    if (type === 'text') {
      setText(samples[0])
    } else if (type === 'image') {
      const item = samples[0]
      const dataUri = typeof item === 'string' ? item : (item.data || item.url)
      setPreviewUrl(dataUri)
      if (dataUri && dataUri.includes(',')) {
        const b64 = dataUri.split(',')[1]
        setImageFile({ _b64: b64 })
      }
    }
  }

  const handleImageFile = (file) => {
    if (file) {
      setImageFile(file)
      setPreviewUrl(URL.createObjectURL(file))
    }
  }

  const handleImageChange = (e) => {
    handleImageFile(e.target.files?.[0])
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith('image/')) {
      handleImageFile(file)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handlePredict = async () => {
    setLoading(true)
    setEditMode(false)
    setMaskedIndices(new Set())
    try {
      if (inputType === 'text') {
        await onPredict(text)
      } else {
        if (imageFile) {
          const b64 = imageFile._b64 || await fileToBase64(imageFile)
          await onPredict(b64)
        }
      }
    } finally {
      setLoading(false)
    }
  }

  const canPredict = inputType === 'text' ? text.trim() : !!imageFile
  const words = (lastData && inputType === 'text' && typeof lastData === 'string')
    ? lastData.split(' ')
    : []

  const toggleWord = (idx) => {
    setMaskedIndices((prev) => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  const handleEditPredict = async () => {
    if (!onEditInput || maskedIndices.size === 0) return
    setEditLoading(true)
    try {
      await onEditInput({
        action: 'mask_tokens',
        indices: [...maskedIndices],
      })
    } finally {
      setEditLoading(false)
    }
  }

  // ── Image mask drawing handlers ───────────────────────
  const getRelativePos = useCallback((e) => {
    const rect = imgEditRef.current?.getBoundingClientRect()
    if (!rect) return null
    return {
      x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
      y: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height)),
    }
  }, [])

  const handleImgMouseDown = useCallback((e) => {
    e.preventDefault()
    const pos = getRelativePos(e)
    if (!pos) return
    setDragging(true)
    setDragStart(pos)
    setMaskRegion(null)
  }, [getRelativePos])

  const handleImgMouseMove = useCallback((e) => {
    if (!dragging || !dragStart) return
    const pos = getRelativePos(e)
    if (!pos) return
    setMaskRegion({
      x: Math.min(dragStart.x, pos.x),
      y: Math.min(dragStart.y, pos.y),
      w: Math.abs(pos.x - dragStart.x),
      h: Math.abs(pos.y - dragStart.y),
    })
  }, [dragging, dragStart, getRelativePos])

  const handleImgMouseUp = useCallback(() => {
    setDragging(false)
  }, [])

  const handleImgEditPredict = async () => {
    if (!onEditInput || !maskRegion) return
    setEditLoading(true)
    try {
      await onEditInput({
        action: 'mask_region',
        region: maskRegion,
      })
    } finally {
      setEditLoading(false)
    }
  }

  return (
    <div className="glass-card rounded-2xl shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-gray-800">Input</h2>
        <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-500 bg-indigo-50 px-2.5 py-1 rounded-full">
          {domain || 'none'} · {inputType}
        </span>
      </div>

      {/* Text mode */}
      {inputType === 'text' && !editMode && (
        <div className="space-y-3">
          <textarea
            className="w-full border border-gray-200 rounded-xl p-4 text-sm min-h-[120px] focus:ring-2 focus:ring-indigo-300 focus:border-indigo-300 focus:outline-none bg-gray-50/50 transition resize-none"
            placeholder={`Enter text to classify (${domain})…`}
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button
            onClick={loadSample}
            className="text-sm text-indigo-600 hover:text-indigo-800 font-medium flex items-center gap-1.5 transition"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Load sample text
          </button>
        </div>
      )}

      {/* Text edit mode */}
      {inputType === 'text' && editMode && words.length > 0 && (
        <div className="space-y-3">
          <p className="text-xs text-gray-500 font-medium">Click words to mask them, then re-predict:</p>
          <div className="flex flex-wrap gap-1.5 p-4 bg-gray-50/80 rounded-xl min-h-[80px] border border-gray-100">
            {words.map((word, i) => (
              <button
                key={i}
                onClick={() => toggleWord(i)}
                className={`px-2 py-1 rounded-lg text-sm transition-all ${ 
                  maskedIndices.has(i)
                    ? 'bg-red-100 text-red-600 line-through font-medium'
                    : 'bg-white border border-gray-200 hover:bg-indigo-50 hover:border-indigo-200'
                }`}
              >
                {maskedIndices.has(i) ? '[MASK]' : word}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleEditPredict}
              disabled={editLoading || maskedIndices.size === 0}
              className="flex-1 bg-amber-500 hover:bg-amber-600 disabled:bg-gray-300 text-white font-semibold py-2.5 rounded-xl transition text-sm shadow-sm"
            >
              {editLoading ? 'Re-predicting…' : `Re-predict (${maskedIndices.size} masked)`}
            </button>
            <button
              onClick={() => { setEditMode(false); setMaskedIndices(new Set()) }}
              className="px-4 py-2.5 border border-gray-200 text-gray-600 rounded-xl hover:bg-gray-50 text-sm font-medium transition"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Image mode with drag-and-drop */}
      {inputType === 'image' && (
        <div className="space-y-3">
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={() => setDragOver(false)}
            onClick={() => fileInputRef.current?.click()}
            className={`relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-200 ${
              dragOver
                ? 'border-indigo-400 bg-indigo-50/50 scale-[1.01]'
                : previewUrl
                  ? 'border-gray-200 bg-gray-50/30'
                  : 'border-gray-300 bg-gray-50/50 hover:border-indigo-300 hover:bg-indigo-50/30'
            }`}
          >
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Preview"
                className="max-h-52 mx-auto rounded-lg shadow-sm"
              />
            ) : (
              <div className="py-4">
                <div className="w-14 h-14 mx-auto mb-3 rounded-full bg-indigo-50 flex items-center justify-center">
                  <svg className="w-7 h-7 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <p className="text-sm font-medium text-gray-600">
                  Drop an image here or <span className="text-indigo-600">browse</span>
                </p>
                <p className="text-xs text-gray-400 mt-1">
                    {domain === 'medical' ? 'Chest X-ray' : domain === 'vision' ? 'Bird photo' : 'Image file'}
                </p>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              className="hidden"
            />
          </div>
          <button
            onClick={loadSample}
            className="text-sm text-indigo-600 hover:text-indigo-800 font-medium flex items-center gap-1.5 transition"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Load sample image
          </button>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2 mt-4">
        <button
          onClick={handlePredict}
          disabled={loading || !canPredict || editMode || imgEditMode}
          className="flex-1 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 disabled:from-gray-300 disabled:to-gray-300 text-white font-semibold py-2.5 rounded-xl transition-all shadow-sm hover:shadow-md"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Predicting…
            </span>
          ) : 'Predict'}
        </button>
        {lastData && inputType === 'text' && !editMode && (
          <button
            onClick={() => setEditMode(true)}
            className="px-4 py-2.5 border border-amber-300 text-amber-600 rounded-xl hover:bg-amber-50 transition text-sm font-semibold"
          >
            Edit Input
          </button>
        )}
        {lastData && inputType === 'image' && !imgEditMode && previewUrl && (
          <button
            onClick={() => setImgEditMode(true)}
            className="px-4 py-2.5 border border-amber-300 text-amber-600 rounded-xl hover:bg-amber-50 transition text-sm font-semibold"
          >
            Mask Region
          </button>
        )}
      </div>

      {/* Image mask editing overlay */}
      {imgEditMode && previewUrl && (
        <div className="mt-3 space-y-3 border-t border-gray-100 pt-3">
          <p className="text-xs text-gray-500 font-medium">Draw a rectangle on the image to mask a region, then re-predict:</p>
          <div
            ref={imgEditRef}
            className="relative inline-block cursor-crosshair select-none"
            onMouseDown={handleImgMouseDown}
            onMouseMove={handleImgMouseMove}
            onMouseUp={handleImgMouseUp}
            onMouseLeave={handleImgMouseUp}
          >
            <img src={previewUrl} alt="Edit" className="max-h-60 rounded-lg" draggable={false} />
            {maskRegion && (
              <div
                className="absolute border-2 border-red-500 bg-red-500/20 rounded"
                style={{
                  left: `${maskRegion.x * 100}%`,
                  top: `${maskRegion.y * 100}%`,
                  width: `${maskRegion.w * 100}%`,
                  height: `${maskRegion.h * 100}%`,
                }}
              />
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleImgEditPredict}
              disabled={editLoading || !maskRegion}
              className="flex-1 bg-amber-500 hover:bg-amber-600 disabled:bg-gray-300 text-white font-semibold py-2.5 rounded-xl transition text-sm shadow-sm"
            >
              {editLoading ? 'Re-predicting…' : 'Re-predict with mask'}
            </button>
            <button
              onClick={() => { setImgEditMode(false); setMaskRegion(null) }}
              className="px-4 py-2.5 border border-gray-200 text-gray-600 rounded-xl hover:bg-gray-50 text-sm font-medium transition"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
