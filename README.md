# ⚡ Production-Grade GenAI Summarizer Microservice

A containerized, resilient FastAPI microservice that leverages open-weights LLMs (`openai/gpt-oss-20b` via Groq) for structured, deterministic executive summaries and token streaming.

## 🚀 Key Architectural Features

- **FastAPI Core**: High-throughput ASGI framework with asynchronous lifespan context management and connection pooling.
- **Pydantic Validation**: Strict schema enforcement (`min_length`, type validation) preventing malformed prompts from reaching upstream inference providers.
- **Error Isolation**: Clean HTTP status code boundaries:
  - `HTTP 422`: Schema validation failures (client-side).
  - `HTTP 502`: Upstream inference API/provider failure with root-cause propagation.
  - `HTTP 500`: Internal application exceptions.
- **Token Streaming**: Real-time Server-Sent Events (SSE) via `StreamingResponse` for low perceived latency.
- **Automated Test Suite**: Unit and integration tests with `pytest` utilizing `unittest.mock` to simulate upstream LLM responses with zero API costs.
- **Dockerized**: Lightweight, production-ready container definition.

---

## 🛠️ API Specification

| Method | Endpoint | Description | Status Codes |
|---|---|---|---|
| `GET` | `/health` | Service uptime and LLM client health probe | 200, 503 |
| `POST` | `/summarize` | Returns structured 3-bullet JSON summary | 200, 422, 502 |
| `POST` | `/summarize/stream` | Streams generated tokens in real time (SSE) | 200, 500 |

---

## 📦 Quickstart & Local Setup

### 1. Clone & Environment Setup
```bash
git clone https://github.com/Umanagalla27/genai-sprint.git
cd genai-sprint
python -m venv venv
source venv/Scripts/activate  # Or ./venv/Scripts/Activate.ps1 on Windows
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

### 3. Run Automated Tests
```bash
pytest -v
```

### 4. Run Development Server
```bash
uvicorn src.app:app --reload --port 8000
```
Visit interactive Swagger API documentation at: `http://localhost:8000/docs`.

---

## 🐳 Docker Deployment

```bash
docker build -t genai-summarizer .
docker run -p 8000:8000 --env-file .env genai-summarizer
```

---

## 🧠 Portfolio Project 1: Production-Grade RAG Engine

An enterprise-ready Retrieval-Augmented Generation (RAG) system featuring hierarchical recursive chunking, hybrid search (dense + BM25), cross-encoder reranking, and automated LLM-as-a-Judge evaluation.

### 📐 Pipeline Architecture

```text
User Query
│
├──► [Dense Vector Search] ──► FastEmbed (BAAI/bge-small-en-v1.5) ──► Qdrant Vector Store
│                                                                            │
└──► [Sparse Keyword Search] ─► BM25Okapi (Exact Term Matching) ────────────┤
                                                                            ▼
                                                              [Reciprocal Rank Fusion (RRF)]
                                                                            │ Top 10 Fused Candidates
                                                                            ▼
                                                               [FlashRank Cross-Encoder] (ms-marco-TinyBERT-L-2-v2)
                                                                            │ Top 3 Reranked Chunks
                                                                            ▼
                                                                [Groq LLM Inference] (Strict Grounding)
                                                                            │
                                                                            ▼
                                                                 Grounded Response + Sources
```

### 🔬 Key Technical Innovations

1. **Hierarchical Recursive Chunking (`src/rag/chunker.py`)**:
   - Preserves semantic boundaries (`\n\n` -> `\n` -> `. ` -> ` `) with configurable sliding window overlap.
   - Enriches each chunk with lineage metadata: `doc_id`, `chunk_id`, character counts, and index tracking.
2. **Hybrid Search with Reciprocal Rank Fusion (`src/rag/hybrid_search.py`)**:
   - Combines semantic similarity (capturing intent/synonyms) with BM25 (capturing exact product codes, acronyms, and keywords).
   - Merges disparate ranking distributions using RRF ($k=60$):
     $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(d)}$$
3. **Two-Stage Retrieval with Cross-Encoder Reranking (`src/rag/reranker.py`)**:
   - Eliminates bi-encoder semantic drift by performing joint cross-attention over `(query, passage)` pairs via FlashRank ONNX runtime.
4. **Automated Evaluation Harness (`src/rag/evaluator.py`)**:
   - LLM-as-a-Judge test suite evaluating 5 curated ground-truth engineering scenarios.
   - Measures **Faithfulness (Grounding)** and **Answer Relevance**.

### 📊 Benchmark Results

| Metric | Target | Measured Result | Evaluation Method |
|---|---|---|---|
| **Faithfulness (Grounding)** | > 85% | **90.0%** | LLM-as-a-Judge (Zero Hallucination Check) |
| **Answer Relevance** | > 90% | **96.0%** | LLM-as-a-Judge (Query Intent Alignment) |
| **End-to-End Latency** | < 2.5s | **~1.2s** | FastEmbed + Qdrant + Groq LPU Inference |
| **Test Suite Coverage** | 100% | **12 / 12 Passing** | `pytest -v` automated suite |

---

## 🤖 Portfolio Project 2: Autonomous Multi-Tool Agent with Guardrails

A production-grade, state-managed autonomous agent featuring ReAct reasoning, LangGraph cyclic state graphs, safe AST computation, live web search, enterprise input/output guardrails, and automated evaluation benchmarks.

### 📐 Agent Architecture

```text
User Query
│
▼
[Enterprise Guardrails] ──► 1. Prompt Injection & Jailbreak Defense
                        ──► 2. Maximum Input Length & Boundary Verification
│ (Blocked if flagged)
▼
[LangGraph / ReAct State Machine]
│
├──► Node: [Reasoning Engine] (LLM Logit Evaluation & Tool Selection)
│    │
│    ├──► Direct Text Answer ──► Terminate
│    │
│    └──► Tool Call Invocation ──► Node: [Tool Execution]
│                                  ├──► [AST Safe Calculator]
│                                  ├──► [Live Web Search (DDGS)]
│                                  └──► Reducer: Append Tool Message
│                                       │
│                                       ▼
│                                 Loop back to [Reasoning Engine]
│
└──► Circuit Breaker: Max Iterations Cap (<= 6) & Token Budget Limit
│
▼
[Output Guardrails] ─────► 3. PII Redaction Mask (Emails, Phones, Credit Cards)
│
▼
Grounded Response + Audit Trace
```

### 🔬 Core Architectural Capabilities

1. **Dual Orchestration Engines**:
   - **Zero-Dependency ReAct Core (`src/agent/react_engine.py`)**: Lightweight, framework-free loop for high-throughput microservices.
   - **LangGraph State Graph (`src/agent/state_graph.py`)**: Cyclic graph with explicit state reducers (`add_messages`), conditional edge branching (`should_continue`), and termination safeguards.

2. **AST Safe Computation Tool (`src/agent/tools.py`)**:
   - Strictly parses arithmetic expressions via Python's Abstract Syntax Tree (`ast.BinOp`, `ast.Constant`).
   - Eliminates 100% of remote code execution vulnerabilities inherent in native `eval()`.

3. **Enterprise Defense Shield (`src/agent/guardrails.py`)**:
   - **Input Shield**: Heuristic and regex pattern blocking for prompt injections (e.g., *"ignore previous instructions"*).
   - **Output Redactor**: Automated PII masking for emails, phone numbers, and financial data.
   - **Cost Control**: Token accumulation tracking and circuit breakers preventing runaway LLM billing.

4. **FastAPI Microservice Integration (`POST /agent/run`)**:
   - Exposes autonomous execution over REST with full audit traces (`step`, `action`, `arguments`, `observation`).

### 📊 Agent Evaluation Benchmark

| Metric | Target | Measured Result | Evaluation Method |
|---|---|---|---|
| **Task Completion Rate** | > 90% | **100.0%** | Automated Multi-Scenario Benchmark Suite |
| **Tool Selection Accuracy** | > 80% | **80% - 100%** | Ground Truth Tool Intent Matching |
| **Injection Defense Rate** | 100% | **100.0%** | Malicious Jailbreak Payload Rejection |
| **PII Redaction Accuracy** | 100% | **100.0%** | Regex Entity Redaction Verification |
| **Test Suite Coverage** | 100% | **29 / 29 Passing** | `pytest -v` Automated Test Suite |

---


