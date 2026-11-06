import type { ChatEvent } from "@/lib/types"

export function detailMessage(payload: unknown): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = payload.detail
    if (typeof detail === "string") return detail
  }
  return "Algo deu errado."
}

export function safeUrl(url: string): string {
  try {
    const parsed = new URL(url)
    if (parsed.protocol === "http:" || parsed.protocol === "https:") return parsed.href
  } catch {
    return ""
  }
  return ""
}

export function consumeEvents(buffer: string, onEvent: (event: ChatEvent) => void): string {
  const parts = buffer.split("\n\n")
  const rest = parts.pop() ?? ""
  for (const part of parts) {
    const line = part.split("\n").find((entry) => entry.startsWith("data:"))
    if (!line) continue
    onEvent(JSON.parse(line.slice(5).trim()) as ChatEvent)
  }
  return rest
}

export function encodeWav(chunks: Float32Array[], sampleRate: number): Blob {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0)
  const buffer = new ArrayBuffer(44 + length * 2)
  const view = new DataView(buffer)
  const write = (offset: number, text: string) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index))
    }
  }
  write(0, "RIFF")
  view.setUint32(4, 36 + length * 2, true)
  write(8, "WAVE")
  write(12, "fmt ")
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  write(36, "data")
  view.setUint32(40, length * 2, true)
  let offset = 44
  for (const chunk of chunks) {
    for (let index = 0; index < chunk.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, chunk[index]))
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
      offset += 2
    }
  }
  return new Blob([buffer], { type: "audio/wav" })
}
