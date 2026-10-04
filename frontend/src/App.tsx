import { useState, useEffect } from 'react'
import { Database, Server } from 'lucide-react'

function App() {
  const [backendStatus, setBackendStatus] = useState<string>('checking...')

  useEffect(() => {
    fetch('http://localhost:8000/api/health')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'ok') {
          setBackendStatus('connected')
        } else {
          setBackendStatus('error')
        }
      })
      .catch(() => {
        setBackendStatus('disconnected')
      })
  }, [])

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center justify-center p-4">
      <div className="max-w-md w-full bg-gray-800 rounded-xl shadow-lg p-8 border border-gray-700">
        <div className="flex items-center justify-center mb-6">
          <Database className="w-12 h-12 text-blue-500 mr-4" />
          <h1 className="text-3xl font-bold text-white">RAGLab</h1>
        </div>
        
        <p className="text-gray-400 text-center mb-8">
          Offline RAG Experimentation Workbench
        </p>

        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-gray-700/50 rounded-lg">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-3 animate-pulse"></div>
              <span className="font-medium">Frontend</span>
            </div>
            <span className="text-sm text-green-400">Running</span>
          </div>

          <div className="flex items-center justify-between p-4 bg-gray-700/50 rounded-lg">
            <div className="flex items-center">
              <Server className="w-5 h-5 text-gray-400 mr-3" />
              <span className="font-medium">Backend API</span>
            </div>
            <span className={`text-sm ${
              backendStatus === 'connected' ? 'text-green-400' : 
              backendStatus === 'checking...' ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {backendStatus}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
