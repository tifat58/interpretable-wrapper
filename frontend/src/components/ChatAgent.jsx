import { useState, useRef, useEffect, useCallback } from 'react'

export default function ChatAgent({ onSend, hasPrediction }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'Hi! I can explain the model\'s predictions. Try asking "why this prediction?" or "what if I change insult?"' },
  ])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = async () => {
    const msg = input.trim()
    if (!msg) return

    setMessages((prev) => [...prev, { role: 'user', text: msg }])
    setInput('')
    setSending(true)

    try {
      const reply = await onSend(msg)
      setMessages((prev) => [...prev, { role: 'assistant', text: reply }])
    } catch {
      setMessages((prev) => [...prev, { role: 'assistant', text: 'Sorry, something went wrong.' }])
    } finally {
      setSending(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleReset = useCallback(async () => {
    try {
      await fetch('/chat/reset', { method: 'POST' })
    } catch {}
    setMessages([
      { role: 'assistant', text: 'Conversation reset. Ask me anything about the model\'s predictions!' },
    ])
  }, [])

  return (
    <div className="glass-card rounded-2xl shadow-md p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-gray-800 flex items-center gap-2">
          <span className="w-7 h-7 rounded-lg bg-indigo-100 flex items-center justify-center text-sm">🤖</span>
          Chat Agent
        </h2>
        {messages.length > 1 && (
          <button
            onClick={handleReset}
            className="text-xs text-gray-400 hover:text-gray-600 font-medium transition"
          >
            Reset
          </button>
        )}
      </div>

      {/* Message list */}
      <div className="h-56 overflow-y-auto border border-gray-200 rounded-xl p-4 mb-4 space-y-3 bg-gray-50/50 styled-scrollbar">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex items-end gap-2 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {m.role === 'assistant' && (
              <span className="w-6 h-6 rounded-full bg-indigo-100 flex items-center justify-center text-[10px] flex-shrink-0 mb-0.5">🤖</span>
            )}
            <div
              className={`max-w-[75%] px-4 py-2.5 text-sm leading-relaxed ${
                m.role === 'user'
                  ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 text-white rounded-2xl rounded-br-md shadow-sm'
                  : 'bg-white border border-gray-200 text-gray-800 rounded-2xl rounded-bl-md shadow-sm'
              }`}
            >
              {m.text}
            </div>
            {m.role === 'user' && (
              <span className="w-6 h-6 rounded-full bg-indigo-600 flex items-center justify-center text-[10px] text-white flex-shrink-0 mb-0.5">👤</span>
            )}
          </div>
        ))}
        {sending && (
          <div className="flex items-end gap-2 justify-start">
            <span className="w-6 h-6 rounded-full bg-indigo-100 flex items-center justify-center text-[10px] flex-shrink-0">🤖</span>
            <div className="bg-white border border-gray-200 px-4 py-2.5 rounded-2xl rounded-bl-md text-sm text-gray-400 shadow-sm">
              <span className="flex items-center gap-1">
                <span className="animate-bounce" style={{ animationDelay: '0ms' }}>·</span>
                <span className="animate-bounce" style={{ animationDelay: '150ms' }}>·</span>
                <span className="animate-bounce" style={{ animationDelay: '300ms' }}>·</span>
              </span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={hasPrediction ? 'Ask about the prediction…' : 'Run a prediction first…'}
          disabled={!hasPrediction || sending}
          className="flex-1 border border-gray-200 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-300 focus:border-indigo-300 focus:outline-none disabled:bg-gray-100 bg-gray-50/50 transition"
        />
        <button
          onClick={handleSend}
          disabled={!hasPrediction || sending || !input.trim()}
          className="bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 disabled:from-gray-300 disabled:to-gray-300 text-white font-semibold px-5 py-2.5 rounded-xl transition-all text-sm shadow-sm hover:shadow-md"
        >
          Send
        </button>
      </div>
    </div>
  )
}
