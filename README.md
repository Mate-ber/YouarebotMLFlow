# Echo Bot

A simple echo bot for the HumanOrBot project that replies to any received message with the same text.

## Overview

This service provides a FastAPI-based API endpoint that receives messages and echoes them back. It is designed to work with the HumanOrBot service, responding to each message with the same text.

## Running the Service

Go to the project directory:

### On Linux/macOS

```bash
chmod +x run_all_linux.sh
```

```bash
./run_all_linux.sh
```

### On Windows

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\run_all_windows.ps1
```

#### These scripts will:
1. Install Poetry (if needed)
2. Install project dependencies
3. Set up an SSH tunnel to the remote host
4. Start the FastAPI application on port 6782

---

## Microservice Architecture (Docker Compose)

Responsibilities are split across separate services. Inside the Compose
network services call each other by **service name**, never `localhost`.

| Service        | Role                                                   | Internal address                       |
|----------------|--------------------------------------------------------|----------------------------------------|
| `mlflow`       | Tracking server + model registry                       | `http://mlflow:5000`                   |
| `classifier`   | FastAPI; loads champion model from MLflow, `POST /predict` | `http://classifier:8000/predict`   |
| `llm`          | llama.cpp server (OpenAI-style chat API)               | `http://llm:8080/v1/chat/completions`  |
| `orchestrator` | Public gateway; **only routes**, runs no model         | `http://localhost:8090`                |

The `orchestrator` forwards:
- `POST /predict` → `http://classifier:8000/predict`
- `POST /get_message` → `http://llm:8080/v1/chat/completions`

**LLM port note:** the `llm` service uses port **8080** (llama.cpp default), so
the orchestrator forwards `/get_message` to `http://llm:8080/...`, not 11434.

The classifier registers the champion into MLflow on first start (from
`app/artifacts/`) and then loads `models:/bot-classifier@champion`.

### Run

```bash
docker compose up --build
```

- Public API: http://localhost:8090
- MLflow UI:  http://localhost:5001  (host 5001 → container 5000, so it won't clash with a local MLflow on 5000)

Requires the model file `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` locally (large, not committed).

### Test

`/predict` — returns a bot probability in `[0, 1]`:

```bash
curl -s -X POST http://localhost:8090/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "как я могу вам помочь?", "dialog_id": "11111111-1111-1111-1111-111111111111", "id": "22222222-2222-2222-2222-222222222222", "participant_index": 0}'
```

`/get_message` — returns a real answer from the LLM:

```bash
curl -s -X POST http://localhost:8090/get_message \
  -H "Content-Type: application/json" \
  -d '{"dialog_id": "11111111-1111-1111-1111-111111111111", "last_msg_text": "Привет! Как дела?", "last_message_id": "33333333-3333-3333-3333-333333333333"}'
```

No tokens or API keys are required or committed.
