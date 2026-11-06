export const LEVELS = ["A2", "B1", "B2", "C1"] as const

export type Level = (typeof LEVELS)[number]

export type ReadingSummary = {
  id: number
  topic: string
  level: string
  title: string
  url: string
  source: string
  created_at: string
}

export type ChatMessage = {
  id?: number
  role: "user" | "assistant"
  content: string
}

export type Reading = ReadingSummary & {
  text: string
  messages: ChatMessage[]
}

export type Health = {
  ollama_up: boolean
  model_ready: boolean
  model: string
}

export type ChatEvent = {
  delta?: string
  sentence?: string
  error?: string
  status?: string
  done?: boolean
}
