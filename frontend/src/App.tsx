import { useState, useEffect } from 'react'
import { Database, Server, Upload, FileText, CheckCircle, AlertCircle } from 'lucide-react'

interface IngestionResponse {
  filename: string
  file_type: string
  total_characters: number
  total_chunks: number
  embedded_chunks: number
  collection: string
}

function App() {
  const [backendStatus, setBackendStatus] = useState<string>('checking...')
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<IngestionResponse | null>(null)

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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0])
      setError(null)
      setResult(null)
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setUploading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('http://localhost:8000/api/documents/ingest', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => null)
        throw new Error(errData?.detail || `Upload failed with status ${res.status}`)
      }

      const data: IngestionResponse = await res.json()
      setResult(data)
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred during upload.')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center p-8">
      
      {/* Header */}
      <div className="flex items-center justify-center mb-8">
        <Database className="w-10 h-10 text-blue-500 mr-4" />
        <h1 className="text-4xl font-bold text-white tracking-tight">RAGLab</h1>
      </div>

      <div className="max-w-2xl w-full space-y-6">
        
        {/* Status Section */}
        <div className="bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-700 flex gap-4">
          <div className="flex-1 flex items-center justify-between p-4 bg-gray-700/50 rounded-lg">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-3 animate-pulse"></div>
              <span className="font-medium">Frontend</span>
            </div>
            <span className="text-sm text-green-400">Running</span>
          </div>

          <div className="flex-1 flex items-center justify-between p-4 bg-gray-700/50 rounded-lg">
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

        {/* Upload Section */}
        <div className="bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-700">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Upload className="w-5 h-5 mr-2 text-blue-400" />
            Document Ingestion
          </h2>
          
          <div className="space-y-4">
            <div className="flex items-center justify-center w-full">
              <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-600 border-dashed rounded-lg cursor-pointer bg-gray-700/30 hover:bg-gray-700/50 transition-colors">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <FileText className="w-8 h-8 text-gray-400 mb-2" />
                  <p className="mb-2 text-sm text-gray-300">
                    <span className="font-semibold">Click to select a file</span>
                  </p>
                  <p className="text-xs text-gray-500">PDF, TXT, MD, DOCX</p>
                </div>
                <input 
                  type="file" 
                  className="hidden" 
                  accept=".pdf,.txt,.md,.docx"
                  onChange={handleFileChange}
                />
              </label>
            </div>

            {file && (
              <div className="flex items-center justify-between bg-gray-700 p-3 rounded-md">
                <span className="text-sm text-gray-200 truncate pr-4">{file.name}</span>
                <button
                  onClick={handleUpload}
                  disabled={uploading}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:text-gray-400 text-white text-sm font-medium rounded-md transition-colors whitespace-nowrap"
                >
                  {uploading ? 'Uploading...' : 'Upload & Index'}
                </button>
              </div>
            )}

            {error && (
              <div className="flex items-center p-3 mt-4 text-sm text-red-400 bg-red-900/30 rounded-lg border border-red-800">
                <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Result Section */}
        {result && (
          <div className="bg-gray-800 rounded-xl shadow-lg p-6 border border-green-700/50">
            <h2 className="text-xl font-semibold mb-4 flex items-center text-green-400">
              <CheckCircle className="w-5 h-5 mr-2" />
              Ingestion Successful
            </h2>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-700/50 p-4 rounded-lg border border-gray-700">
                <p className="text-sm text-gray-400 mb-1">Filename</p>
                <p className="font-medium text-gray-100 truncate" title={result.filename}>{result.filename}</p>
              </div>
              <div className="bg-gray-700/50 p-4 rounded-lg border border-gray-700">
                <p className="text-sm text-gray-400 mb-1">File Type</p>
                <p className="font-medium text-gray-100 uppercase">{result.file_type}</p>
              </div>
              <div className="bg-gray-700/50 p-4 rounded-lg border border-gray-700">
                <p className="text-sm text-gray-400 mb-1">Total Characters</p>
                <p className="font-medium text-gray-100">{result.total_characters.toLocaleString()}</p>
              </div>
              <div className="bg-gray-700/50 p-4 rounded-lg border border-gray-700">
                <p className="text-sm text-gray-400 mb-1">Collection</p>
                <p className="font-medium text-gray-100">{result.collection}</p>
              </div>
              <div className="bg-gray-700/50 p-4 rounded-lg border border-gray-700">
                <p className="text-sm text-gray-400 mb-1">Total Chunks</p>
                <p className="font-medium text-gray-100">{result.total_chunks}</p>
              </div>
              <div className="bg-gray-700/50 p-4 rounded-lg border border-gray-700">
                <p className="text-sm text-gray-400 mb-1">Embedded Chunks</p>
                <p className="font-medium text-green-400">{result.embedded_chunks}</p>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  )
}

export default App
