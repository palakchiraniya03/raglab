import { useState, useEffect, useRef, useMemo } from 'react'
import {
  Database, FileText, CheckCircle, AlertCircle,
  Loader2, Plus, ArrowUp, MessageSquare, Activity
} from 'lucide-react'

interface IngestionResponse {
  filename: string
  file_type: string
  total_characters: number
  total_chunks: number
  embedded_chunks: number
  collection: string
  already_indexed?: boolean
}

interface DocumentItem {
  document_id: string
  filename: string
  file_type: string
  chunk_count: number
}

interface RetrievalResult {
  text: string
  score: number
  semantic_score?: number
  lexical_boost?: number
  final_score?: number
  selected?: boolean
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

interface ChatMessage {
  role: "user" | "assistant"
  content: string
  sources?: RetrievalResult[]
}

interface ChatSession {
  id: string
  title: string
  createdAt: number
  messages: ChatMessage[]
}

interface RAGResponse {
  answer: string
  sources: RetrievalResult[]
}

const getRelevantPassage = (chunk: string, query: string) => {
  const MAX_LENGTH = 300
  if (chunk.length <= MAX_LENGTH) {
    return { passage: chunk, found: false }
  }

  let normChunk = ''
  const normToOrig: number[] = []

  let inSpace = false
  for (let i = 0; i < chunk.length; i++) {
    const char = chunk[i]
    if (/\s/.test(char)) {
      if (!inSpace) {
        normToOrig.push(i)
        normChunk += ' '
        inSpace = true
      }
    } else {
      normToOrig.push(i)
      normChunk += char.toLowerCase()
      inSpace = false
    }
  }
  normToOrig.push(chunk.length)

  const STOP_WORDS = new Set(['what', 'is', 'the', 'of', 'a', 'an', 'are', 'was', 'were', 'to', 'in', 'on', 'for', 'and', 'or', 'how', 'why', 'which'])

  const cleanQuery = query.toLowerCase().replace(/[^\w\s]/g, '')
  const terms = cleanQuery.split(/\s+/).filter(t => t.length > 2 && !STOP_WORDS.has(t))

  if (terms.length === 0) {
    return {
      passage: chunk.substring(0, MAX_LENGTH) + '...',
      found: false
    }
  }

  const phrase = terms.join(' ')

  const scoreWindowOrig = (origStart: number) => {
    const origEnd = Math.min(chunk.length, origStart + MAX_LENGTH)

    const windowOrig = chunk.substring(origStart, origEnd)
    const windowNorm = windowOrig.toLowerCase().replace(/\s+/g, ' ')

    let score = 0
    let termsFound = 0
    for (const term of terms) {
      if (windowNorm.includes(term)) {
        score += 10
        termsFound++
      }
    }

    if (termsFound === 0) score -= 500
    if (windowNorm.includes(phrase)) score += 1000

    const codePatterns = ['np.', 'nx.', 'def ', 'return ', 'toarray', '=', '_', '{', '}', '[', ']', 'λ', 'diag', 'matrix', 'import ']
    const lowerWindowOrig = windowOrig.toLowerCase()
    for (const pat of codePatterns) {
      score -= (lowerWindowOrig.split(pat).length - 1) * 3
    }

    return { score, origStart, origEnd, termsFound }
  }

  let bestResult = null
  let pos = normChunk.indexOf(phrase)

  if (pos !== -1) {
    while (pos !== -1) {
      const matchOrigStart = normToOrig[pos]
      // Start exactly at the matched concept to avoid pulling in preceding code
      const windowStartOrig = matchOrigStart

      const res = scoreWindowOrig(windowStartOrig)
      if (!bestResult || res.score > bestResult.score) {
        bestResult = res
      }
      pos = normChunk.indexOf(phrase, pos + 1)
    }
  } else {
    // Slide over original chunk only if no exact phrase match was found
    for (let i = 0; i <= chunk.length; i += 50) {
      const res = scoreWindowOrig(i)
      if (!bestResult || res.score > bestResult.score) {
        bestResult = res
      }
    }
  }

  if (bestResult && bestResult.termsFound > 0) {
    let passage = chunk.substring(bestResult.origStart, bestResult.origEnd)
    if (pos === -1 && bestResult.origStart > 0) passage = '...' + passage
    if (bestResult.origEnd < chunk.length) passage = passage + '...'
    return { passage, found: true }
  }

  return {
    passage: chunk.substring(0, MAX_LENGTH) + '...',
    found: false
  }
}

function SourceCard({ source, query }: { source: RetrievalResult, query: string }) {
  const [expanded, setExpanded] = useState(false)
  const isLong = source.text.length > 300

  const { passage, found } = useMemo(() => {
     return getRelevantPassage(source.text, query)
  }, [source.text, query])

  return (
    <div className="bg-[#262626] border border-white/5 rounded-xl p-4 flex flex-col gap-3">
       <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-gray-500 border-b border-white/5 pb-3">
          <span className="font-medium text-gray-300 flex items-center gap-1.5">
            <FileText className="w-3.5 h-3.5" />
            {source.metadata.filename}
          </span>
          {source.metadata.page !== null && <span>Page {source.metadata.page}</span>}
          <span>Chunk {source.metadata.chunk_index}</span>
          <span className="text-emerald-500/80 font-medium">Score {source.score.toFixed(3)}</span>
       </div>

       <div>
          {!expanded ? (
             <>
                <div className="text-xs font-medium text-gray-400 mb-2">
                   {found ? "Relevant passage" : "Retrieved passage"}
                </div>
                <div className="text-gray-300 text-[13.5px] leading-[1.7] whitespace-pre-wrap break-words">
                   {passage}
                </div>
             </>
          ) : (
             <>
                <div className="text-xs font-medium text-gray-400 mb-2">
                   Full chunk context
                </div>
                <div className="text-gray-400 text-[13.5px] leading-[1.7] whitespace-pre-wrap break-words">
                   {source.text}
                </div>
             </>
          )}
       </div>

       {isLong && (
          <button
             onClick={() => setExpanded(!expanded)}
             className="text-xs font-medium text-gray-500 hover:text-gray-300 transition-colors self-start mt-1"
          >
             {expanded ? "Show less" : "Show surrounding context"}
          </button>
       )}
    </div>
  )
}

function RetrievalInspector({ query, sources }: { query: string, sources: RetrievalResult[] }) {
  const [expanded, setExpanded] = useState(false)

  if (!sources || sources.length === 0) return null

  return (
    <div className="mt-5 mb-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 text-[12px] font-medium text-gray-500 hover:text-gray-300 transition-colors"
      >
        <Activity className="w-3.5 h-3.5" />
        Retrieval details
      </button>

      {expanded && (
        <div className="mt-3 bg-[#1c1c1c] border border-white/5 rounded-xl p-5 text-[13px] text-gray-400">
           <div className="mb-4 pb-4 border-b border-white/5">
             <div className="text-gray-500 mb-1">Query</div>
             <div className="text-gray-200">"{query}"</div>
             <div className="mt-3 text-emerald-500/80 font-medium">{sources.length} selected chunk{sources.length === 1 ? '' : 's'}</div>
           </div>

           <div className="flex flex-col gap-6">
             {sources.map((src, idx) => (
               <div key={idx} className="flex flex-col">
                  <div className="text-gray-300 font-medium mb-3">
                    {src.metadata.filename} {src.metadata.page !== null ? `· Page ${src.metadata.page}` : ''} · Chunk {src.metadata.chunk_index}
                  </div>
                  <div className="font-mono text-[12.5px] flex flex-col gap-1.5">
                    <div className="flex justify-between max-w-[280px]">
                      <span>Semantic similarity</span>
                      <span>{(src.semantic_score ?? src.score).toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between max-w-[280px]">
                      <span>Lexical boost</span>
                      <span>+{(src.lexical_boost ?? 0).toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between max-w-[280px] text-gray-300">
                      <span>Final score</span>
                      <span>{(src.final_score ?? src.score).toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between max-w-[280px]">
                      <span>Threshold</span>
                      <span>0.500</span>
                    </div>
                    <div className="mt-1 text-emerald-500/80">
                      {src.selected !== false ? '✓ Selected' : '✗ Dropped'}
                    </div>
                  </div>
               </div>
             ))}
           </div>
        </div>
      )}
    </div>
  )
}

function App() {
  const [backendStatus, setBackendStatus] = useState<string>('checking...')

  // Ingestion state
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<IngestionResponse | null>(null)
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [documentsLoading, setDocumentsLoading] = useState(false)

  // Chat state
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [sessionsLoaded, setSessionsLoaded] = useState(false)

  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [askError, setAskError] = useState<string | null>(null)

  const messagesEndRef = useRef<HTMLDivElement>(null)

  const fetchDocuments = async () => {
    setDocumentsLoading(true)
    try {
      const res = await fetch('http://localhost:8000/api/documents')
      if (res.ok) {
        const data = await res.json()
        setDocuments(data)
      }
    } catch (err) {
      console.error("Failed to fetch documents", err)
    } finally {
      setDocumentsLoading(false)
    }
  }

  useEffect(() => {
    if (backendStatus === 'connected') {
      fetchDocuments()
    }
  }, [backendStatus])

  // Load backend status
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

  // Load chat sessions from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem('raglab_chat_sessions')
      if (saved) {
        const parsed = JSON.parse(saved)
        if (Array.isArray(parsed)) {
          setSessions(parsed)
          if (parsed.length > 0) {
            setActiveSessionId(parsed[0].id)
          }
        }
      }
    } catch (err) {
      console.error("Failed to load sessions", err)
    } finally {
      setSessionsLoaded(true)
    }
  }, [])

  // Save chat sessions to localStorage whenever they change
  useEffect(() => {
    if (sessionsLoaded) {
      localStorage.setItem('raglab_chat_sessions', JSON.stringify(sessions))
    }
  }, [sessions, sessionsLoaded])

  // Auto scroll to bottom
  const activeSession = sessions.find(s => s.id === activeSessionId)
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [activeSession?.messages])

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
      setFile(null)
      fetchDocuments()
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred during upload.')
    } finally {
      setUploading(false)
    }
  }

  const generateTitle = (q: string) => q.length > 40 ? q.substring(0, 40) + '...' : q

  const handleAsk = async () => {
    if (!question.trim()) return

    setAsking(true)
    setAskError(null)

    const currentQuery = question.trim()
    setQuestion('')

    let targetSessionId = activeSessionId
    if (!targetSessionId) {
      targetSessionId = Date.now().toString()
      setActiveSessionId(targetSessionId)
    }
    const finalSessionId = targetSessionId

    setSessions(prev => {
      const idx = prev.findIndex(s => s.id === finalSessionId)
      if (idx === -1) {
        const newSession: ChatSession = {
          id: finalSessionId,
          title: generateTitle(currentQuery),
          createdAt: Date.now(),
          messages: [{ role: "user", content: currentQuery }]
        }
        return [newSession, ...prev]
      } else {
        const newSessions = [...prev]
        const updatedSession = { ...newSessions[idx] }
        if (updatedSession.messages.length === 0) {
          updatedSession.title = generateTitle(currentQuery)
        }
        updatedSession.messages = [...updatedSession.messages, { role: "user", content: currentQuery }]
        newSessions[idx] = updatedSession
        return newSessions
      }
    })

    try {
      const res = await fetch('http://localhost:8000/api/rag/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: currentQuery, top_k: 5 })
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => null)
        throw new Error(errData?.detail || `Request failed with status ${res.status}`)
      }

      const data: RAGResponse = await res.json()

      setSessions(prev => {
        const newSessions = [...prev]
        const idx = newSessions.findIndex(s => s.id === finalSessionId)
        if (idx !== -1) {
          const updatedSession = { ...newSessions[idx] }
          updatedSession.messages = [
            ...updatedSession.messages,
            { role: "assistant", content: data.answer, sources: data.sources }
          ]
          newSessions[idx] = updatedSession
        }
        return newSessions
      })
    } catch (err: any) {
      setAskError(err.message || 'An unexpected error occurred.')
    } finally {
      setAsking(false)
    }
  }

  const hasMessages = activeSession && activeSession.messages.length > 0

  return (
    <div className="flex h-screen bg-[#212121] text-gray-100 font-sans overflow-hidden selection:bg-white/20">

      {/* DESKTOP SIDEBAR */}
      <aside className="w-[260px] bg-[#171717] flex-col shrink-0 md:flex hidden relative">
        <div className="p-4 flex flex-col h-full">

           {/* Logo / Header */}
           <div className="flex items-center gap-3 mb-6 px-2">
             <div className="w-8 h-8 rounded-full bg-white flex items-center justify-center shrink-0">
               <Database className="w-5 h-5 text-black" />
             </div>
             <div className="flex flex-col">
               <span className="font-semibold text-[15px] text-gray-100 tracking-tight">RAGLab</span>
               <span className="text-xs text-gray-400">Offline RAG Workbench</span>
             </div>
           </div>

           {/* New Chat Action */}
           <button
             onClick={() => setActiveSessionId(null)}
             className="flex items-center gap-3 px-3 py-2.5 rounded-lg border border-white/10 hover:bg-white/5 transition-colors cursor-pointer mb-6 group w-full"
           >
              <Plus className="w-4 h-4 text-gray-300" />
              <span className="text-[14px] font-medium text-gray-200">New chat</span>
           </button>

           {/* Chats List */}
           <div className="flex-1 overflow-y-auto min-h-0 mb-4">
             <div className="px-2 mb-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">Recent</div>
             {sessions.length === 0 ? (
               <div className="px-3 text-[13px] text-gray-500">No conversations yet</div>
             ) : (
               <div className="flex flex-col gap-1">
                 {sessions.map(s => (
                   <button
                     key={s.id}
                     onClick={() => setActiveSessionId(s.id)}
                     className={`flex items-center w-full px-3 py-2 rounded-lg text-left text-[13px] transition-colors ${activeSessionId === s.id ? 'bg-[#2f2f2f] text-gray-200' : 'text-gray-400 hover:bg-[#212121]'}`}
                   >
                     <MessageSquare className={`w-3.5 h-3.5 mr-2 shrink-0 ${activeSessionId === s.id ? 'text-gray-300' : 'text-gray-500'}`} />
                     <span className="truncate">{s.title}</span>
                   </button>
                 ))}
               </div>
             )}
           </div>

           {/* Knowledge Base */}
           <div className="shrink-0 border-t border-white/5 pt-4 mb-2">
             <div className="px-2 mb-3 text-xs font-semibold text-gray-500 uppercase tracking-wider flex justify-between items-center">
                Knowledge Base
                <label className="cursor-pointer text-gray-400 hover:text-gray-200 transition-colors" title="Add document">
                   <Plus className="w-3.5 h-3.5" />
                   <input type="file" className="hidden" accept=".pdf,.txt,.md,.docx" onChange={handleFileChange} />
                </label>
             </div>

             {/* Uploading Status */}
             {file && !uploading && !result && (
                <div className="px-3 py-3 rounded-lg bg-[#212121] border border-white/10 mb-2 flex flex-col gap-3">
                   <div className="flex items-center gap-2 overflow-hidden">
                      <FileText className="w-4 h-4 text-gray-400 shrink-0" />
                      <span className="text-[13px] text-gray-200 truncate">{file.name}</span>
                   </div>
                   <button
                      onClick={handleUpload}
                      className="w-full py-1.5 bg-white text-black text-[13px] font-medium rounded hover:bg-gray-200 transition-colors"
                    >
                      Upload & Index
                    </button>
                </div>
             )}

             {uploading && (
               <div className="px-3 py-3 rounded-lg bg-[#212121] border border-white/10 mb-2 flex items-center gap-3">
                  <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
                  <span className="text-[13px] text-gray-300">Indexing...</span>
               </div>
             )}

             {error && (
               <div className="px-3 py-3 rounded-lg bg-red-900/20 border border-red-900/30 mb-2 flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                  <span className="text-[13px] text-red-400 leading-tight">{error}</span>
               </div>
             )}

             {documentsLoading && documents.length === 0 ? (
               <div className="px-3 py-2 flex items-center gap-2 text-[13px] text-gray-500">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading...
               </div>
             ) : documents.length > 0 ? (
               <div className="flex flex-col gap-1 overflow-y-auto max-h-[30vh]">
                 {documents.map((doc, idx) => (
                   <div key={idx} className="px-3 py-2.5 rounded-lg hover:bg-[#212121] transition-colors cursor-default group flex flex-col gap-1.5 mx-1">
                      <div className="flex items-center gap-2">
                         <FileText className="w-4 h-4 text-gray-400 shrink-0" />
                         <span className="text-[14px] text-gray-200 truncate" title={doc.filename}>{doc.filename}</span>
                      </div>
                      <div className="flex items-center gap-3 text-xs text-gray-500 pl-6">
                         <span className="uppercase">{doc.file_type}</span>
                         <span className="flex items-center gap-1"><CheckCircle className="w-3 h-3 text-emerald-500/80"/> {doc.chunk_count} chunks</span>
                      </div>
                   </div>
                 ))}
               </div>
             ) : (!file && !uploading && (
               <div className="px-3 text-[13px] text-gray-500">No documents indexed</div>
             ))}
           </div>

           {/* Footer: Status */}
           <div className="pt-4 border-t border-white/5 flex items-center gap-2 px-2 text-[13px]">
             {backendStatus === 'connected' ? (
                <div className="flex items-center gap-2 text-gray-400">
                  <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                  Backend connected
                </div>
              ) : backendStatus === 'checking...' ? (
                <div className="flex items-center gap-2 text-gray-500">
                  <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                  Checking status
                </div>
              ) : (
                <div className="flex items-center gap-2 text-gray-500">
                  <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                  Backend offline
                </div>
              )}
           </div>
        </div>
      </aside>

      {/* MOBILE HEADER */}
      <div className="md:hidden flex items-center justify-between p-4 bg-[#212121] absolute top-0 w-full z-10 border-b border-white/5">
         <div className="font-semibold text-gray-200">RAGLab</div>
         <button onClick={() => setActiveSessionId(null)} className="text-gray-300">
           <Plus className="w-5 h-5" />
         </button>
      </div>

      {/* MAIN WORKSPACE */}
      <main className="flex-1 flex flex-col min-w-0 bg-[#212121] relative h-full">

         <div className="flex-1 overflow-y-auto">
            <div className="max-w-3xl mx-auto pt-20 md:pt-10 pb-40 px-4 md:px-6">

               {/* Empty State */}
               {!hasMessages && !asking && (
                 <div className="flex flex-col items-center justify-center h-[60vh] text-center">
                    <div className="w-12 h-12 bg-white flex items-center justify-center rounded-full mb-6">
                       <Database className="w-6 h-6 text-black" />
                    </div>
                    <h2 className="text-2xl font-semibold text-gray-200 mb-2">Ask your documents</h2>
                    <p className="text-[15px] text-gray-400">Upload a document and ask questions grounded in its contents.</p>
                 </div>
               )}

               {/* Active Conversation */}
               {hasMessages && (
                 <div className="flex flex-col gap-8 animate-in fade-in duration-500">
                    {activeSession!.messages.map((msg, msgIdx) => {
                       if (msg.role === 'user') {
                          return (
                            <div key={msgIdx} className="flex justify-end mt-4">
                               <div className="bg-[#2f2f2f] px-5 py-3.5 rounded-3xl max-w-[85%] text-[15px] text-gray-100 whitespace-pre-wrap">
                                  {msg.content}
                               </div>
                            </div>
                          )
                       } else {
                          return (
                            <div key={msgIdx} className="flex gap-4">
                               <div className="w-8 h-8 rounded-full bg-white flex flex-shrink-0 items-center justify-center mt-1">
                                  <Database className="w-5 h-5 text-black" />
                               </div>
                               <div className="flex flex-col min-w-0 w-full">

                                  {/* Answer Text */}
                                  <div className="text-[16px] text-gray-200 leading-relaxed whitespace-pre-wrap">
                                     {msg.content}
                                  </div>

                                  {/* Retrieval Inspector */}
                                  {msg.sources && msg.sources.length > 0 && (
                                     <RetrievalInspector query={activeSession.messages[msgIdx - 1]?.content || ''} sources={msg.sources} />
                                  )}

                                  {/* Sources Block */}
                                  {msg.sources && msg.sources.length > 0 && (
                                     <div className="mt-6 border-t border-white/5 pt-5">
                                        <div className="text-[12px] font-semibold text-gray-500 uppercase tracking-wider mb-4">Sources</div>
                                        <div className="flex flex-col gap-3">
                                            {msg.sources.map((source, idx) => (
                                               <SourceCard key={idx} source={source} query={activeSession!.messages[msgIdx - 1]?.content || ''} />
                                            ))}
                                        </div>
                                     </div>
                                  )}
                               </div>
                            </div>
                          )
                       }
                    })}
                    {asking && (
                       <div className="flex gap-4">
                          <div className="w-8 h-8 rounded-full bg-white flex flex-shrink-0 items-center justify-center mt-1">
                             <Database className="w-5 h-5 text-black" />
                          </div>
                          <div className="flex items-center text-gray-400">
                             <Loader2 className="w-5 h-5 animate-spin" />
                          </div>
                       </div>
                    )}
                    <div ref={messagesEndRef} />
                 </div>
               )}
            </div>
         </div>

         {/* COMPOSER AT BOTTOM */}
         <div className="absolute bottom-0 left-0 w-full bg-gradient-to-t from-[#212121] via-[#212121] to-transparent pt-10 pb-6 px-4 md:px-6">
            <div className="max-w-3xl mx-auto relative">

               {askError && (
                 <div className="mb-3 px-4 py-3 bg-red-900/20 border border-red-900/30 rounded-lg text-[13px] text-red-400 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    <span>{askError}</span>
                 </div>
               )}

               <div className="relative flex items-end bg-[#2f2f2f] rounded-[24px] focus-within:ring-1 ring-gray-400 shadow-md">
                  <input
                    type="text"
                    placeholder={documents.length > 0 ? "Ask something about your documents..." : "Upload a document to ask questions..."}
                    className="w-full bg-transparent text-[15px] text-gray-100 placeholder-gray-400 px-5 py-4 pr-14 focus:outline-none rounded-[24px]"
                    value={question}
                    disabled={documents.length === 0 || asking}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        e.preventDefault()
                        handleAsk()
                      }
                    }}
                  />
                  <div className="absolute right-2.5 bottom-2.5">
                     <button
                       onClick={handleAsk}
                       disabled={asking || !question.trim() || documents.length === 0}
                       className="w-9 h-9 rounded-full bg-white text-black hover:bg-gray-200 disabled:bg-[#404040] disabled:text-gray-500 transition-colors flex items-center justify-center"
                     >
                       {asking ? <Loader2 className="w-4 h-4 animate-spin" /> : <ArrowUp className="w-4 h-4" />}
                     </button>
                  </div>
               </div>
               <div className="text-center mt-3 text-[11px] text-gray-500">
                  RAGLab can make mistakes. Verify important information with the sources.
               </div>
            </div>
         </div>
      </main>
    </div>
  )
}

export default App
