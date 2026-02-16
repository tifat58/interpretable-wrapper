import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/predict': 'http://localhost:5000',
      '/counterfactual': 'http://localhost:5000',
      '/explain': 'http://localhost:5000',
      '/chat': 'http://localhost:5000',
      '/samples': 'http://localhost:5000',
      '/domains': 'http://localhost:5000',
      '/attribution': 'http://localhost:5000',
      '/edit_input': 'http://localhost:5000',
      '/models': 'http://localhost:5000',
      '/rag': 'http://localhost:5000',
      '/concept_strategies': 'http://localhost:5000',
      '/local_surrogate': 'http://localhost:5000',
    },
  },
})
