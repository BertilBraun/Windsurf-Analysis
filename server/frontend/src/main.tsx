import React from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'
import { App } from './ui/App'
import { SingleInstanceGuard } from './ui/components/SingleInstanceGuard'

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <SingleInstanceGuard>
            <App />
        </SingleInstanceGuard>
    </React.StrictMode>
)
