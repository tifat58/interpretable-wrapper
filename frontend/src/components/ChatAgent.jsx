import { useState, useRef, useEffect } from 'react'

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

  return (
    <div className="bg-white rounded-xl shadow p-5">
      <h2 className="text-lg font-semibold mb-3">Chat Agent</h2>

      {/* Message list */}
      <div className="h-56 overflow-y-auto border rounded-lg p-3 mb-3 space-y-3 bg-gray-50">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] px-3 py-2 rounded-lg text-sm leading-relaxed ${
                m.role === 'user'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-white border text-gray-800'
              }`}
            >
              {m.text}
            </div>
          </div>
        ))}
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
          className="flex-1 border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-300 focus:outline-none disabled:bg-gray-100"
        />
        <button
          onClick={handleSend}
          disabled={!hasPrediction || sending || !input.trim()}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-300 text-white font-medium px-4 py-2 rounded-lg transition text-sm"
        >
          Send
        </button>
      </div>
    </div>
  )
}
