import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/predict': 'http://localhost:5000',
      '/counterfactual': 'http://localhost:5000',
      '/explain': 'http://localhost:5000',
      '/chat': 'http://localhost:5000',
      '/samples': 'http://localhost:5000',
      '/probe_state': 'http://localhost:5000',
      '/probe_feedback': 'http://localhost:5000',
      '/probe_retrain': 'http://localhost:5000',
      '/probe_reset': 'http://localhost:5000',
    },
  },
})
