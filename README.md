# Sala de leitura

Pratique inglês lendo e conversando sobre um texto por vez. Você escolhe um tema e um nível, o app busca um texto atual na web e depois conversa com você sobre ele, por texto ou por voz.

A conversa, a transcrição da fala e a voz rodam na sua máquina. Só a busca do texto sai para a internet.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Licença GPL-3.0](https://img.shields.io/badge/licen%C3%A7a-GPL--3.0-green)

## Como funciona

| Etapa | Ferramenta | Onde roda |
| --- | --- | --- |
| Busca do texto e de fatos atuais | [OpenRouter](https://openrouter.ai) (`openai/gpt-4o-mini:online`) | Nuvem |
| Conversa | [Ollama](https://ollama.com) com Llama 3.2 | Local |
| Fala para texto | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`small.en`) | Local |
| Texto para fala | [Piper](https://github.com/OHF-Voice/piper1-gpl) (`en_US-lessac-medium`) | Local |
| Servidor | FastAPI + SQLite | Local |
| Interface | React, Vite e shadcn/ui | Navegador |

Níveis disponíveis: A2, B1, B2 e C1. Os textos e as conversas ficam salvos em `data/readings.db`.

Quando você pergunta algo que o modelo local não sabe, como o tempo hoje ou uma notícia, o app faz uma busca extra no OpenRouter e passa o resumo para o Llama responder.

## Requisitos

- Python 3.9 ou mais novo
- [Ollama](https://ollama.com/download) com o modelo `llama3.2`
- Uma chave do [OpenRouter](https://openrouter.ai/keys)
- Node.js 20, só se for alterar a interface

## Instalação

```bash
git clone https://github.com/tiagoJ-collab/learn-english-agent.git
cd learn-english-agent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Abra o `.env` e preencha `OPENROUTER_API_KEY`.

Baixe o modelo da conversa uma vez:

```bash
ollama pull llama3.2
```

O primeiro uso do microfone baixa o Whisper `small.en`. A primeira resposta falada baixa a voz Piper. Os dois ficam em `data/`.

## Uso

Em um terminal, deixe o Ollama rodando:

```bash
ollama serve
```

Em outro, na pasta do projeto:

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8765
```

Abra http://127.0.0.1:8765, busque um tema e leia o texto. Segure **Falar** para conversar em inglês. O **Modo conversa** reabre o microfone quando a resposta termina.

## Configuração

Todas as opções ficam no `.env`:

| Variável | Padrão | Para que serve |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | vazio | Chave da busca na web |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini:online` | Modelo da busca |
| `OLLAMA_MODEL` | `llama3.2` | Modelo da conversa |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Endereço do Ollama |
| `APP_HOST` | `127.0.0.1` | Endereço do servidor |
| `APP_PORT` | `8765` | Porta do servidor |
| `WHISPER_MODEL` | `small.en` | Modelo de transcrição |
| `PIPER_VOICE` | `en_US-lessac-medium` | Voz das respostas |
| `DATA_DIR` | `data/` | Banco, modelos e vozes baixados |

O servidor não tem autenticação. Mantenha `APP_HOST=127.0.0.1` para que ele fique acessível só no seu computador.

## Interface

O código do front fica em `frontend/`. O servidor entrega os arquivos gerados em `app/static/`. Depois de mudar a interface, gere de novo:

```bash
cd frontend
npm install
npm run build
```

## Testes

```bash
pytest
```

Os testes não chamam OpenRouter, Ollama, Whisper nem Piper, então rodam sem rede e sem modelos baixados.

## Estrutura

```
app/
  main.py         rotas da API e entrega da interface
  openrouter.py   busca do texto e de fatos atuais
  ollama.py       conversa com o Llama em streaming
  stt.py          transcrição com faster-whisper
  tts.py          síntese de voz com Piper
  store.py        banco SQLite
  static/         interface já compilada
frontend/         código da interface em React
tests/            testes com pytest
```

## Contribuindo

Veja [CONTRIBUTING.md](CONTRIBUTING.md).

## Licença

Distribuído sob a [GNU GPL v3.0 ou posterior](LICENSE). O projeto usa o `piper-tts`, que é GPL-3.0, por isso a mesma licença.

Os modelos baixados em tempo de uso têm licenças próprias e não fazem parte deste repositório: [Llama 3.2](https://www.llama.com/llama3_2/license/), [Whisper](https://github.com/openai/whisper/blob/main/LICENSE) e as [vozes do Piper](https://huggingface.co/rhasspy/piper-voices). Os textos buscados pertencem às fontes originais e ficam só no seu computador.
