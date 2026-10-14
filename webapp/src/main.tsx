import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { appBase } from './api'
import App from './App'
import './index.css'

// A project's pages live under /p/<project>/ (see api.ts): the router works below that, so links stay as they are.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter basename={appBase || undefined}>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
