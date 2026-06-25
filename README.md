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
