import { useEffect, useRef, useState } from "react"
import type { FormEvent, PointerEvent as ReactPointerEvent } from "react"
import { MicIcon } from "lucide-react"

import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { consumeEvents, detailMessage, encodeWav, safeUrl } from "@/lib/audio"
import { LEVELS, type ChatMessage, type Level, type Reading, type ReadingSummary } from "@/lib/types"
import { cn } from "@/lib/utils"

type RecordMode = "hold" | "talk"
type Recorder = {
  stream: MediaStream
  audioCtx: AudioContext
  processor: ScriptProcessorNode
  source: MediaStreamAudioSourceNode
  chunks: Float32Array[]
  state: { heardSpeech: boolean; silence: number; elapsed: number; stopping: boolean }
}

export default function App() {
  const [topic, setTopic] = useState("")
  const [level, setLevel] = useState<Level>("B1")
  const [readings, setReadings] = useState<ReadingSummary[]>([])
  const [reading, setReading] = useState<Reading | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [draft, setDraft] = useState("")
  const [healthMessage, setHealthMessage] = useState("")
  const [error, setError] = useState("")
  const [status, setStatus] = useState("")
  const [busy, setBusyState] = useState(false)
  const [searching, setSearching] = useState(false)
  const [recording, setRecording] = useState(false)
  const [talk, setTalk] = useState(false)

  const busyRef = useRef(false)
  const readingIdRef = useRef<number | null>(null)
  const recordingRef = useRef(false)
  const recordModeRef = useRef<RecordMode | null>(null)
  const recorderRef = useRef<Recorder | null>(null)
  const cancelStartRef = useRef(false)
  const stopRequestedRef = useRef(false)
  const talkRef = useRef(false)
  const playCtxRef = useRef<AudioContext | null>(null)
  const playSourceRef = useRef<AudioBufferSourceNode | null>(null)
  const bottomRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    talkRef.current = talk
  }, [talk])

  useEffect(() => {
    readingIdRef.current = reading?.id ?? null
  }, [reading])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" })
  }, [messages, status])

  useEffect(() => {
    const onPointerUp = () => {
      if (recordModeRef.current === "hold") stopRecording()
    }
    window.addEventListener("pointerup", onPointerUp)
    return () => window.removeEventListener("pointerup", onPointerUp)
  }, [])

  useEffect(() => {
    let active = true
    void (async () => {
      await refreshHealth()
      const items = await refreshReadings()
      if (!active || items.length === 0) return
      await openReading(items[0].id)
    })()
    return () => {
      active = false
    }
  }, [])

  function setBusy(value: boolean) {
    busyRef.current = value
    setBusyState(value)
  }

  function ensureAudio() {
    if (!playCtxRef.current) playCtxRef.current = new AudioContext()
    if (playCtxRef.current.state === "suspended") void playCtxRef.current.resume()
  }

  async function refreshHealth() {
    const response = await fetch("/api/health")
    const health = (await response.json()) as { ollama_up: boolean; model_ready: boolean; model: string }
    if (!health.ollama_up) {
      setHealthMessage(`O Ollama não está no ar. Abra o aplicativo e rode ollama pull ${health.model}.`)
      return
    }
    if (!health.model_ready) {
      setHealthMessage(`O modelo ${health.model} ainda não foi baixado. Rode ollama pull ${health.model}.`)
      return
    }
    setHealthMessage("")
  }

  async function refreshReadings() {
    const response = await fetch("/api/readings")
    const items = (await response.json()) as ReadingSummary[]
    setReadings(items)
    return items
  }

  async function openReading(id: number) {
    const response = await fetch(`/api/readings/${id}`)
    const payload = (await response.json()) as Reading & { detail?: string }
    if (!response.ok) {
      setError(detailMessage(payload))
      return
    }
    setError("")
    setReading(payload)
    setMessages(payload.messages ?? [])
    await refreshReadings()
  }

  async function search(event: FormEvent) {
    event.preventDefault()
    if (busyRef.current) return
    setBusy(true)
    setSearching(true)
    setError("")
    try {
      const response = await fetch("/api/readings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic, level }),
      })
      const payload = (await response.json()) as Reading & { detail?: string }
      if (!response.ok) {
        setError(detailMessage(payload))
        return
      }
      payload.messages = payload.messages ?? []
      setReading(payload)
      setMessages([])
      await refreshReadings()
    } catch {
      setError("Não consegui buscar o texto.")
    } finally {
      setSearching(false)
      setBusy(false)
    }
  }

  async function sendMessage(text: string, fromVoice: boolean) {
    const content = text.trim()
    if (!content) return
    const readingId = readingIdRef.current
    if (!readingId) {
      setError("Abra um texto antes de conversar.")
      return
    }
    if (busyRef.current) return
    ensureAudio()
    stopRequestedRef.current = false
    setError("")
    setBusy(true)
    setMessages((current) => [
      ...current,
      { role: "user", content },
      { role: "assistant", content: "" },
    ])
    setStatus("Pensando…")
    const sentences: string[] = []
    try {
      const response = await fetch(`/api/readings/${readingId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      })
      if (!response.ok || !response.body) {
        const payload: unknown = await response.json()
        setError(detailMessage(payload))
        setMessages((current) => current.slice(0, -2))
        return
      }
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      while (true) {
        const chunk = await reader.read()
        if (chunk.done) break
        buffer += decoder.decode(chunk.value, { stream: true })
        buffer = consumeEvents(buffer, (event) => {
          if (event.status) setStatus(event.status)
          if (event.delta) {
            setMessages((current) => {
              const next = current.slice()
              const last = next[next.length - 1]
              if (last?.role === "assistant") next[next.length - 1] = { ...last, content: last.content + event.delta }
              return next
            })
          }
          if (event.sentence) sentences.push(event.sentence)
          if (event.error) {
            setError(event.error)
            setMessages((current) => {
              const last = current[current.length - 1]
              if (last?.role === "assistant" && !last.content) return current.slice(0, -1)
              return current
            })
          }
        })
      }
      await speakAll(sentences)
    } catch {
      setError("A conversa foi interrompida.")
    } finally {
      setBusy(false)
      setStatus("")
    }
    if (fromVoice && talkRef.current && !stopRequestedRef.current) void startRecording("talk")
  }

  async function speakAll(sentences: string[]) {
    if (!sentences.length || stopRequestedRef.current) return
    setStatus("Falando…")
    for (const sentence of sentences) {
      if (stopRequestedRef.current) return
      const response = await fetch("/api/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: sentence }),
      })
      if (!response.ok) {
        setError(detailMessage(await response.json().catch(() => ({}))))
        return
      }
      const bytes = await response.arrayBuffer()
      if (stopRequestedRef.current) return
      await playBuffer(bytes)
    }
  }

  function playBuffer(bytes: ArrayBuffer) {
    return new Promise<void>((resolve, reject) => {
      ensureAudio()
      const context = playCtxRef.current
      if (!context) {
        resolve()
        return
      }
      void context.decodeAudioData(bytes.slice(0)).then((audio) => {
        if (stopRequestedRef.current) {
          resolve()
          return
        }
        const source = context.createBufferSource()
        playSourceRef.current = source
        source.buffer = audio
        source.connect(context.destination)
        source.onended = () => resolve()
        source.start()
      }).catch(reject)
    })
  }

  async function startRecording(kind: RecordMode) {
    if (recordingRef.current || busyRef.current || !readingIdRef.current) return
    ensureAudio()
    cancelStartRef.current = false
    recordingRef.current = true
    recordModeRef.current = kind
    setRecording(true)
    setStatus("Ouvindo…")
    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      recordingRef.current = false
      recordModeRef.current = null
      setRecording(false)
      setError("Não consegui usar o microfone.")
      return
    }
    if (cancelStartRef.current) {
      stream.getTracks().forEach((track) => track.stop())
      return
    }
    const audioCtx = new AudioContext()
    const source = audioCtx.createMediaStreamSource(stream)
    const processor = audioCtx.createScriptProcessor(4096, 1, 1)
    const silent = audioCtx.createGain()
    silent.gain.value = 0
    const chunks: Float32Array[] = []
    const state = { heardSpeech: false, silence: 0, elapsed: 0, stopping: false }
    processor.onaudioprocess = (event) => {
      const data = new Float32Array(event.inputBuffer.getChannelData(0))
      chunks.push(data)
      let energy = 0
      for (let index = 0; index < data.length; index += 1) energy += data[index] * data[index]
      const rms = Math.sqrt(energy / data.length)
      const seconds = data.length / audioCtx.sampleRate
      state.elapsed += seconds
      if (rms > 0.02) {
        state.heardSpeech = true
        state.silence = 0
      } else if (state.heardSpeech) {
        state.silence += seconds
      }
      const timedOut = state.elapsed > 30
      const quietAfterSpeech = recordModeRef.current === "talk" && state.heardSpeech && state.silence > 1.2
      const neverSpoke = recordModeRef.current === "talk" && !state.heardSpeech && state.elapsed > 8
      if (!state.stopping && (timedOut || quietAfterSpeech || neverSpoke)) {
        state.stopping = true
        window.setTimeout(() => stopRecording(), 0)
      }
    }
    source.connect(processor)
    processor.connect(silent)
    silent.connect(audioCtx.destination)
    recorderRef.current = { stream, audioCtx, processor, source, chunks, state }
  }

  function stopRecording() {
    if (!recordingRef.current) return
    recordingRef.current = false
    recordModeRef.current = null
    setRecording(false)
    const current = recorderRef.current
    recorderRef.current = null
    if (!current) {
      cancelStartRef.current = true
      setStatus("")
      return
    }
    const sampleRate = current.audioCtx.sampleRate
    current.processor.onaudioprocess = null
    current.processor.disconnect()
    current.source.disconnect()
    current.stream.getTracks().forEach((track) => track.stop())
    void current.audioCtx.close()
    if (!current.state.heardSpeech) {
      setStatus("Não ouvi nada. Tente falar de novo.")
      return
    }
    void transcribeBlob(encodeWav(current.chunks, sampleRate))
  }

  async function transcribeBlob(blob: Blob) {
    setBusy(true)
    setStatus("Ouvindo…")
    let text = ""
    try {
      const body = new FormData()
      body.append("file", blob, "speech.wav")
      const response = await fetch("/api/transcribe", { method: "POST", body })
      const payload: unknown = await response.json()
      if (!response.ok) {
        setError(detailMessage(payload))
        return
      }
      text = (payload as { text?: string }).text ?? ""
    } catch {
      setError("Não consegui transcrever a fala.")
      return
    } finally {
      setBusy(false)
    }
    if (text) await sendMessage(text, true)
  }

  function stopAll() {
    stopRequestedRef.current = true
    try {
      playSourceRef.current?.stop()
    } catch {
      /* playback already ended */
    }
    stopRecording()
    setStatus("")
  }

  function holdMic(event: ReactPointerEvent<HTMLButtonElement>) {
    if (talk || recording || busy || !reading) return
    event.preventDefault()
    void startRecording("hold")
  }

  function clickMic() {
    if (!talk || busy || !reading) return
    if (recording) stopRecording()
    else void startRecording("talk")
  }

  function submitDraft(event: FormEvent) {
    event.preventDefault()
    const text = draft
    setDraft("")
    void sendMessage(text, false)
  }

  const paragraphs = (reading?.text ?? "").split(/\n+/).map((item) => item.trim()).filter(Boolean)
  const href = reading ? safeUrl(reading.url) : ""

  return (
    <div className="min-h-svh">
      <header className="flex flex-col gap-4 border-b bg-card/70 px-4 py-4 md:flex-row md:items-end md:justify-between md:px-6">
        <div>
          <h1 className="text-3xl tracking-tight">Sala de leitura</h1>
          <p className="text-sm text-muted-foreground">Um texto de cada vez. A conversa fica na sua máquina.</p>
        </div>
        <form className="flex flex-wrap items-center gap-2" onSubmit={(event) => void search(event)}>
          <Label className="sr-only" htmlFor="topic">Tema</Label>
          <Input
            id="topic"
            value={topic}
            maxLength={200}
            placeholder="Tema, por exemplo rain"
            className="w-full sm:w-64"
            onChange={(event) => setTopic(event.target.value)}
          />
          <Select value={level} onValueChange={(value) => setLevel(value as Level)}>
            <SelectTrigger className="w-24" aria-label="Nível">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {LEVELS.map((item) => (
                <SelectItem key={item} value={item}>{item}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button type="submit" disabled={busy}>{searching ? "Buscando…" : "Buscar texto"}</Button>
        </form>
      </header>

      <div className="space-y-2 px-4 pt-3 md:px-6">
        {healthMessage ? (
          <Alert>
            <AlertDescription>{healthMessage}</AlertDescription>
          </Alert>
        ) : null}
        {error ? (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : null}
      </div>

      <main className="grid gap-4 px-4 py-4 md:px-6 lg:h-[calc(100svh-8.5rem)] lg:grid-cols-[240px_minmax(0,1fr)_380px]">
        <Card className="min-h-48 lg:min-h-0">
          <CardHeader>
            <CardTitle className="text-xs tracking-wide text-muted-foreground uppercase">Textos</CardTitle>
          </CardHeader>
          <CardContent className="min-h-0 flex-1">
            <ScrollArea className="h-40 lg:h-[calc(100svh-14rem)]">
              {readings.length === 0 ? (
                <p className="text-sm text-muted-foreground">Nenhum texto ainda.</p>
              ) : (
                <div className="flex flex-col gap-2 pr-3">
                  {readings.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      className={cn(
                        "rounded-lg px-3 py-2 text-left hover:bg-muted",
                        reading?.id === item.id && "bg-muted ring-1 ring-foreground/10",
                      )}
                      onClick={() => void openReading(item.id)}
                    >
                      <span className="block font-medium">{item.title}</span>
                      <span className="text-xs text-muted-foreground">{item.level} · {item.topic}</span>
                    </button>
                  ))}
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>

        <Card className="min-h-[50vh] lg:min-h-0">
          <CardContent className="min-h-0 flex-1">
            <ScrollArea className="h-[50vh] lg:h-[calc(100svh-12rem)]">
              {reading ? (
                <article className="prose-reading max-w-[66ch] pr-4">
                  <h2 className="text-4xl leading-tight tracking-tight">{reading.title}</h2>
                  <p className="mt-3 mb-6">
                    {href ? (
                      <a className="text-primary underline underline-offset-4" href={href} target="_blank" rel="noopener noreferrer">
                        {reading.source || reading.url}
                      </a>
                    ) : (
                      <span>{reading.source}</span>
                    )}
                  </p>
                  <div className="space-y-4 text-lg leading-relaxed" lang="en">
                    {paragraphs.map((paragraph, index) => (
                      <p key={index}>{paragraph}</p>
                    ))}
                  </div>
                </article>
              ) : (
                <div className="prose-reading max-w-xl">
                  <h2 className="text-4xl tracking-tight">Escolha um tema</h2>
                  <p className="mt-3 text-muted-foreground">
                    O texto aparece aqui. Depois você lê e conversa sobre ele em inglês, por voz ou por escrito.
                  </p>
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>

        <Card className="min-h-[70vh] lg:min-h-0">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-xs tracking-wide text-muted-foreground uppercase">Conversa</CardTitle>
              {reading ? <Badge variant="secondary">{reading.level}</Badge> : null}
            </div>
          </CardHeader>
          <CardContent className="flex min-h-0 flex-1 flex-col gap-3">
            <ScrollArea className="h-64 lg:h-[calc(100svh-28rem)]">
              <ol className="flex flex-col gap-2 pr-3" lang="en">
                {messages.map((message, index) => (
                  <li
                    key={`${message.role}-${index}`}
                    className={cn(
                      "max-w-[92%] rounded-2xl px-3 py-2 whitespace-pre-wrap",
                      message.role === "user" ? "ml-auto bg-primary/10" : "bg-muted",
                      message.role === "assistant" && !message.content && "hidden",
                    )}
                  >
                    {message.content}
                  </li>
                ))}
                <div ref={bottomRef} />
              </ol>
            </ScrollArea>
            <form className="flex flex-col gap-2" onSubmit={submitDraft}>
              <Label className="sr-only" htmlFor="draft">Mensagem em inglês</Label>
              <Textarea
                id="draft"
                value={draft}
                placeholder="Ask about the text in English"
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault()
                    event.currentTarget.form?.requestSubmit()
                  }
                }}
              />
              <div className="flex flex-wrap items-center gap-2">
                <Button type="submit" disabled={busy}>Enviar</Button>
                <Button
                  type="button"
                  variant={recording ? "destructive" : "outline"}
                  aria-pressed={recording}
                  onPointerDown={holdMic}
                  onClick={clickMic}
                >
                  <MicIcon />
                  Falar
                </Button>
                <Button type="button" variant="ghost" onClick={stopAll}>Parar</Button>
              </div>
              <div className="flex items-center gap-2">
                <Switch id="talk" checked={talk} onCheckedChange={setTalk} />
                <Label htmlFor="talk">Modo conversa</Label>
              </div>
            </form>
            <p className="text-xs text-muted-foreground">
              Segure Falar para gravar. No modo conversa, o microfone abre de novo quando a resposta termina.
            </p>
            <p className="min-h-5 text-sm text-primary" aria-live="polite">{status}</p>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
