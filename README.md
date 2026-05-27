# 🤖 Enterprise Chatbot — Backend

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green?style=for-the-badge)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-ff0000?style=for-the-badge)
![Redis](https://img.shields.io/badge/Redis-Queue_%26_Cache-dc382d?style=for-the-badge&logo=redis&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)

> **Important Data Privacy Notice** ⚠️  
> For data privacy reasons, the actual knowledge base data (`md/` folder markdown files) and the vector database snapshots (`snapshots/` folder) have been excluded from this public repository. Originally, these contained proprietary client information. The empty directories have been kept to retain the project structure, but they require your own markdown and snapshot files to populate the database. Additionally, the system prompt in `backend/ai.py` has been changed to mask the client's identity.

## 📖 Project Overview
This repository holds the backend for the enterprise chatbot. The API provides a **Retrieval-Augmented Generation (RAG)** pipeline designed to answer queries about the client's programs and platform. It receives a user question, retrieves matching textual content dynamically from a vector database (Qdrant), streams an AI-generated response via Server-Sent Events (SSE), and asynchronously logs queries into MySQL via a background worker queue leveraging Redis.

## ✨ Key Features & Capabilities
*   **Streaming Responses**: Streams real-time tokens to the frontend via Server-Sent Events (SSE), enabling instant user feedback without freezing the UI.
*   **Hybrid Search RAG**: Combines Dense Vector Search (Qdrant) and Lexical Search (BM25) with Reciprocal Rank Fusion (RRF) to retrieve semantic and exact-matched context.
*   **Agentic LangGraph Workflow**: Intelligent agent logic ensuring no assumptions or fabrications, utilizing context-fetching tools seamlessly.
*   **Asynchronous Processing**: Background job queuing powered by ARQ and Redis for non-blocking MySQL database logging.
*   **Rate Limiting**: Integrated connection throttling via SlowAPI to prevent abuse (default 20 requests/session/day).
*   **Conversation Memory**: Chat history contextualized in Redis, ensuring accurate multi-turn questions.
*   **Dockerized Infrastructure**: Complete containerized setup for Qdrant, MySQL, and Redis ensuring reliable local execution.

## 🏗 System Architecture

The core flow works as follows:

1. **Ingestion**: Raw `.md` files are chunked (`chunk.py`), embedded via Google Gemini Embeddings, and upserted into Qdrant.
2. **Serving**: A query hits `main.py` (`FastAPI`).
3. **Queueing**: A background job is dispatched to Redis via ARQ to log the query without slowing down response times. `worker.py` eventually processes this job to write into MySQL permanently.
4. **Retrieval**: `ai.py` fetches the top matches using `fetchContext` (semantic Qdrant search + BM25). 
5. **Generation**: Final context is passed to the agent powered by DeepSeek API, generating the response.
6. **Streaming**: Response is routed safely back to the user via SSE chunks.

```
MD Files → chunk.py → Qdrant (vectors + metadata)
                    ↓
           (snapshot exported)
                    ↓
        ingest_new_data.py → seeds fresh Qdrant

User Query → main.py (FastAPI)
               ├── enqueue log job → Redis (fast drop-off)
               │                        ↓
               │                   worker.py → MySQL (permanent log)
               └── ai.py (fetchContext tool)
                       ├── Qdrant dense search (semantic)
                       ├── BM25 search (keyword, in-memory)
                       ├── RRF fusion → top 5 chunks
                       └── DeepSeek LLM → streamed SSE response → User
```

## 🛠 Tech Stack
* **Framework**: FastAPI
* **LLM Orchestration**: LangChain, LangGraph
* **Models**: DeepSeek API (OpenAI-compatible) for chat generation, Google Generative AI (`gemini-embedding-001`) for dense embeddings.
* **Vector Database**: Qdrant (HNSW Indexing, Cosine Distance)
* **Message Queue & Checkpoint Memory**: Redis, ARQ
* **Relational Database**: MySQL 8.0
* **Rate Limiting**: SlowAPI
* **Containerization**: Docker & Docker Compose

## 📦 Prerequisites
* **Python**: `3.x`
* **Docker & Docker Compose**: To run isolated services locally.
* **API Keys Required**:
  - `OPENAI_API_KEY` (Used for Deepseek as an OpenAI drop-in parameter)
  - `GOOGLE_API_KEY` (For Gemini Text Embeddings)
  - `QDRANT_API_KEY` (If targeting a cloud instance; otherwise omit for local Docker)

## 🐳 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd chatbot-backend
   ```

2. **Configure Environment Variables:**
   Create a `.env` file in the root `backend` directory. See the *Configuration* section below for the required values.

3. **Spin up Docker Containers:**
   Start the Qdrant, Redis, and MySQL components via `docker-compose`:
   ```bash
   cd backend
   docker-compose up -d
   ```
   *(Wait ~15-20 seconds to allow MySQL's init scripts to configure the schema from `schema.sql` the very first time)*

4. **Install Python Dependencies:**
   Create a virtual environment and install packages:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # Or `source .venv/bin/activate` on MacOS/Linux
   pip install -r requirements.txt
   ```

5. **Populate Vector DB (Optional/First time):**
   If you had `.md` files present in `md/` you would run `python chunk.py` to ingest them. Alternatively, if you have a `.snapshot` in the `snapshots/` folder, run `python ingest_new_data.py` to load previous data natively without recalling Google embedding credits.

## 🚀 Running Locally

Once infrastructure and dependencies are installed, you need dual processes:

**Terminal 1 — Run the ARQ Worker (DB Logger)**
```bash
cd backend
arq worker.WorkerSettings
```

**Terminal 2 — Run the FastAPI Server**
```bash
cd backend
uvicorn main:app --reload --port 8080
```
> The API will now stream RAG chatbot queries at `http://127.0.0.1:8080/query`.

## 🌐 API Endpoints

### `POST /query`
Performs query retrieval and streams AI tokens.
*   **Headers:**
    - `Content-Type: application/json`
    - `x-session-id: <unique-session-id>` (Used for Rate Limiting / 20 requests per day threshold)
*   **Payload:**
    ```json
    {
       "userQuery": "What are the rules regarding unfair means?",
       "userId": "usr_995",     // Optional: logged-in user identifier
       "anonId": "anon_239xx"   // Optional: guest user identifier
    }
    ```
*   **Response (Streaming `text/event-stream`):**
    ```text
    data: {"token": "The"}
    
    data: {"token": " rules"}
    
    data: {"token": " state"}
    
    event: done
    ```

## 🧠 RAG Pipeline Workflows

*   **Ingestion Setup**: `chunk.py` implements a LangGraph pipeline to use `RecursiveCharacterTextSplitter` on Markdown files. It sets up 1500-sized chunks with 300 overlaps, applies summarization metadata directly into points payload, creates a deterministic ID via SHA-256 for idempotency, and pushes batches via Qdrant's REST API. 
*   **Query Lifecycle**: `ai.py` sets up in-memory BM25 lexical structures mapping all fetched chunks at runtime along with Qdrant Dense Similarity requests filtering. Both returned outputs traverse a **Reciprocal Rank Fusion (RRF)** blending metric algorithm prior to hitting the Deepseek agent prompt system logic as augmented context.

## 📂 Folder Structure

```
chatbot-backend/
├── backend/
│   ├── ai.py                 # Core RAG, Retriever & LangGraph agent setup
│   ├── chunk.py              # Ingestion code parsing Markdown to Vector Embeddings
│   ├── config.py             # Global constants parsing dot-env values centrally
│   ├── docker-compose.yml    # Spawns Redis, MySQL, Qdrant on bridged ports
│   ├── ingest_new_data.py    # Loads exported database snapshot to Qdrant without embedding API costs
│   ├── main.py               # FastAPI router handling SSE and rate limiting setup
│   ├── middleware.py         # Implements SlowAPI throttles
│   ├── requirements.txt      # Python dependencies
│   ├── schema.sql            # MySQL schema tracking historical questions asked
│   ├── setup.sh              # Bash utility for quick deployment routines (if any)
│   ├── test_retrieval.py     # Local retrieval debugger saving token testing
│   ├── worker.py             # ARQ Background dispatcher executing inserts to MySQL
│   ├── md/                   # (EMPTY) Origin folder for pure textual context (Deleted for privacy)
│   └── snapshots/            # (EMPTY) Origin folder for frozen Qdrant databases (Deleted for privacy)
├── frontend/
│   └── index.html            # Example client interface parsing SSE data chunks
└── README.md                 # Project documentation
```

## ⚙️ Configuration Variables
Populate a local `.env` inside `backend/`:

| Variable | Description |
|---|---|
| `QDRANT_HOST` | Local or remote host for Vector DB (e.g. `127.0.0.1`) |
| `QDRANT_PORT` | Port pointing to Qdrant (e.g. `6333`) |
| `REDIS_HOST` / `PORT` / `DB` | Connection defaults (e.g., `127.0.0.1`, `6380`, `0`) |
| `MYSQL_HOST` / `PORT` | Connection strings targeting mysql (e.g., `127.0.0.1`, `3307`) |
| `MYSQL_USER` / `PASSWORD` / `DATABASE`| Credentials for history insert worker |
| `OPENAI_API_KEY` | Provides Deepseek compatibility calls via `ChatOpenAI` wrapper |
| `GOOGLE_API_KEY` | Provides access strictly for `gemini-embedding-001` ingestion |

## 🛡️ Rate Limiting & Session Management
The API safeguards limits strictly by implementing `SlowAPI`. Unverified hits or registered tokens carry a hard cap of `20 questions per day` keyed by the `x-session-id` header (fallback to Client IP). Reaching this limit triggers a standard JSON HTTP `429 Too Many Requests` envelope, safely preventing AI abuse. 

Sessions are **not** cookie-dependent. They are purely managed via the frontend passing session IDs inside request bodies or headers. Redis Checkpointer threads track conversation histories locally tied to those unique identifiers directly from API layers.
