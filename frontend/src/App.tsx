import { useState, useEffect } from 'react'
import {
  Database, Server, Upload, FileText, CheckCircle, AlertCircle,
  Search, Loader2, MessageSquare, BookOpen
} from 'lucide-react'

interface IngestionResponse {
  filename: string
  file_type: string
  total_characters: number
  total_chunks: number
  embedded_chunks: number
  collection: string
}

interface RetrievalResult {
  text: string
  score: number
  metadata: {
    document_id: string
    filename: string
    file_type: string
    page: number | null
    chunk_index: number
    char_start: number
    char_end: number
  }
}

interface RAGResponse {
  answer: string
  sources: RetrievalResult[]
}

function App() {
  const [backendStatus, setBackendStatus] = useState<string>('checking...')

  // Ingestion state
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<IngestionResponse | null>(null)

  // RAG state
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [askError, setAskError] = useState<string | null>(null)
  const [ragResult, setRagResult] = useState<{ query: string; data: RAGResponse } | null>(null)

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

  const handleAsk = async () => {
    if (!question.trim()) return

    setAsking(true)
    setAskError(null)
    setRagResult(null)

    const currentQuery = question.trim()

    try {
      const res = await fetch('http://localhost:8000/api/rag/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: currentQuery,
          top_k: 5
        })
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => null)
        throw new Error(errData?.detail || `Request failed with status ${res.status}`)
      }

      const data: RAGResponse = await res.json()
      setRagResult({ query: currentQuery, data })
      setQuestion('')
    } catch (err: any) {
      setAskError(err.message || 'An unexpected error occurred.')
    } finally {
      setAsking(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center p-8">

      {/* Header */}
      <div className="flex items-center justify-center mb-8">
        <Database className="w-10 h-10 text-blue-500 mr-4" />
        <h1 className="text-4xl font-bold text-white tracking-tight">RAGLab</h1>
      </div>

      <div className="max-w-3xl w-full space-y-6">

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

        {/* RAG Ask Section */}
        <div className="bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-700">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Search className="w-5 h-5 mr-2 text-blue-400" />
            Ask your documents
          </h2>

          <div className="space-y-4">
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Ask a question about your documents..."
                className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500 placeholder-gray-400"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleAsk() }}
              />
              <button
                onClick={handleAsk}
                disabled={asking || !question.trim()}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:text-gray-400 text-white font-medium rounded-lg transition-colors whitespace-nowrap flex items-center"
              >
                {asking ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Thinking...
                  </>
                ) : (
                  'Ask'
                )}
              </button>
            </div>

            {askError && (
              <div className="flex items-center p-3 text-sm text-red-400 bg-red-900/30 rounded-lg border border-red-800">
                <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
                {askError}
              </div>
            )}

            {ragResult && (
              <div className="mt-6 space-y-4 border-t border-gray-700 pt-6">
                <div className="bg-gray-700/50 rounded-lg p-4">
                  <p className="text-sm text-gray-400 mb-1">Question</p>
                  <p className="text-white font-medium">{ragResult.query}</p>
                </div>

                <div className="bg-gray-700/50 rounded-lg p-4 border border-blue-500/30">
                  <p className="text-sm text-blue-400 mb-2 flex items-center">
                    <MessageSquare className="w-4 h-4 mr-2" />
                    Answer
                  </p>
                  <div className="text-white whitespace-pre-wrap">{ragResult.data.answer}</div>
                </div>

                {ragResult.data.sources && ragResult.data.sources.length > 0 && (
                  <div>
                    <p className="text-sm text-gray-400 mb-3 flex items-center">
                      <BookOpen className="w-4 h-4 mr-2" />
                      Sources
                    </p>
                    <div className="space-y-3">
                      {ragResult.data.sources.map((source, idx) => (
                        <div key={idx} className="bg-gray-700/30 rounded-lg p-4 border border-gray-700 text-sm">
                          <div className="flex flex-wrap gap-x-4 gap-y-2 mb-2 text-gray-400">
                            <span className="flex items-center"><FileText className="w-3 h-3 mr-1"/> {source.metadata.filename}</span>
                            {source.metadata.page !== null && <span>Page: {source.metadata.page}</span>}
                            <span>Chunk: {source.metadata.chunk_index}</span>
                            <span>Score: {source.score.toFixed(3)}</span>
                          </div>
                          <p className="text-gray-300 italic">"{source.text}"</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  )
}

export default App
