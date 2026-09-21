import os
import json
from dataclasses import asdict
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Generator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from groq import Groq, APIError
from src.rag.service import RAGService
from src.agent.multi_tool_agent import create_production_agent
from src.llmops.semantic_cache import SemanticCache
from src.llmops.tracer import LLMOpsTracer

load_dotenv()

client: Groq | None = None
rag_service: RAGService | None = None
semantic_cache: SemanticCache | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, rag_service, semantic_cache
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    client = Groq(api_key=api_key)
    print("INFO: Groq client initialized successfully.")
    rag_service = RAGService()
    print("INFO: RAG service initialized.")
    semantic_cache = SemanticCache(similarity_threshold=0.88)
    print("INFO: Semantic Cache initialized.")
    yield
    print("INFO: Shutting down service.")

app = FastAPI(
    title="Executive Summarizer API",
    description="Production-grade API for LLM-powered summarization.",
    version="1.0.0",
    lifespan=lifespan,
)

class SummarizeRequest(BaseModel):
    topic: str = Field(
        ..., 
        min_length=3, 
        max_length=500, 
        description="The topic or text to summarize."
    )
    model: str = Field(
        default="openai/gpt-oss-20b",
        description="Target model identifier."
    )

class SummarizeResponse(BaseModel):
    topic: str
    summary: str
    model_used: str
    status: str = "success"


class IngestRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=100)
    text: str = Field(..., min_length=20)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    status: str = "success"


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    context_used: str = ""
    sources: List[Dict[str, Any]]

@app.get("/health", tags=["Monitoring"])
def health_check() -> Dict[str, str]:
    if client is None:
        raise HTTPException(status_code=503, detail="LLM Client uninitialized")
    return {"status": "healthy", "service": "summarizer-api"}

@app.post(
    "/summarize", 
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    tags=["GenAI"]
)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM Client is not initialized."
        )

    system_prompt = (
        "You are an executive AI assistant. Summarize the user's topic in exactly "
        "3 concise, high-impact bullet points. Each bullet must be under 25 words."
    )
    user_prompt = f"Topic to summarize: {request.topic}"

    try:
        completion = client.chat.completions.create(
            model=request.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        summary_text = completion.choices[0].message.content
        if not summary_text:
            raise HTTPException(status_code=500, detail="Empty response received from LLM.")

        return SummarizeResponse(
            topic=request.topic,
            summary=summary_text,
            model_used=request.model,
            status="success"
        )
    except APIError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Groq upstream error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal processing failure: {str(e)}"
        )

def stream_llm_tokens(topic: str, model: str) -> Generator[str, None, None]:
    """Generates chunks of text as they arrive from the LLM."""
    system_prompt = (
        "You are an executive AI assistant. Summarize the user's topic in exactly "
        "3 concise, high-impact bullet points. Each bullet must be under 25 words."
    )
    user_prompt = f"Topic to summarize: {topic}"

    try:
        stream = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,  # Enables chunk-by-chunk streaming
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                yield token
    except Exception as e:
        yield f"\n[Streaming Error: {str(e)}]"

@app.post("/summarize/stream", tags=["GenAI"])
def summarize_stream(request: SummarizeRequest):
    """Stream summary tokens in real time using Server-Sent Events."""
    if client is None:
        raise HTTPException(status_code=500, detail="LLM Client uninitialized")

    return StreamingResponse(
        stream_llm_tokens(request.topic, request.model),
        media_type="text/event-stream"
    )


@app.post("/rag/ingest", response_model=IngestResponse, tags=["RAG"])
def ingest_document(req: IngestRequest) -> IngestResponse:
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    count = rag_service.ingest_document(req.text, req.doc_id, req.metadata)
    return IngestResponse(doc_id=req.doc_id, chunks_indexed=count)


@app.post("/rag/query", response_model=RAGQueryResponse, tags=["RAG"])
def query_rag(req: RAGQueryRequest) -> RAGQueryResponse:
    if rag_service is None or client is None:
        raise HTTPException(status_code=500, detail="Services not initialized")
    result = rag_service.answer_query(req.query, llm_client=client, top_k=req.top_k)
    return RAGQueryResponse(**result)


class AgentExecutionStep(BaseModel):
    step: int
    action: str
    arguments: Dict[str, Any]
    observation: str


class AgentRunRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Query or instruction for the autonomous agent.",
        examples=["Calculate total cost for 15 units at $85 each with 8.5% sales tax."],
    )
    max_iterations: int = Field(default=6, ge=1, le=10)


class AgentRunResponse(BaseModel):
    query: str
    final_answer: str
    steps_taken: int
    execution_trace: List[AgentExecutionStep]
    status: str


@app.post(
    "/agent/run",
    response_model=AgentRunResponse,
    status_code=status.HTTP_200_OK,
    tags=["Agentic AI"],
)
def run_agent(req: AgentRunRequest) -> AgentRunResponse:
    """
    Execute the Autonomous Multi-Tool Agent with ReAct reasoning,
    circuit breakers, and input/output guardrails.
    """
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM Client is not initialized.",
        )

    # Initialize agent with production toolset and requested loop limit
    agent = create_production_agent(client=client, max_iterations=req.max_iterations)
    result = agent.run(req.query)

    return AgentRunResponse(
        query=result["query"],
        final_answer=result["final_answer"],
        steps_taken=result["steps_taken"],
        execution_trace=[
            AgentExecutionStep(
                step=s["step"],
                action=s["action"],
                arguments=s["arguments"],
                observation=str(s["observation"]),
            )
            for s in result["execution_trace"]
        ],
        status=result["status"],
    )


class CompilerRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language query to compile into structured catalog filter JSON.",
        examples=["Show Nike running shoes under 120 dollars in stock"]
    )
    model: str = Field(default="openai/gpt-oss-20b")
    use_cache: bool = Field(default=True, description="Enable semantic caching.")


class CompilerResponse(BaseModel):
    query: str
    compiled_filter: Dict[str, Any]
    cache_hit: bool
    similarity_score: float
    latency_ms: float
    cost_usd: float
    model_used: str


class CacheStatsResponse(BaseModel):
    model_config = {"extra": "ignore"}
    total_lookups: int
    hits: int
    misses: int
    hit_rate_pct: float
    total_latency_saved_ms: float


@app.post(
    "/llmops/compile",
    response_model=CompilerResponse,
    status_code=status.HTTP_200_OK,
    tags=["Fine-Tuning & LLMOps"]
)
def compile_query_with_telemetry(req: CompilerRequest) -> CompilerResponse:
    """
    Compile natural language to JSON filter with Semantic Caching,
    latency profiling, and token cost telemetry.
    """
    if client is None or semantic_cache is None:
        raise HTTPException(status_code=500, detail="Services not initialized")

    tracer = LLMOpsTracer(trace_id="req_compile", model_name=req.model)
    tracer.start_span("cache_lookup")

    # 1. Check Semantic Cache
    if req.use_cache:
        is_hit, cached_response, sim_score = semantic_cache.lookup(req.query)
        if is_hit:
            tracer.end_span("cache_lookup")
            profile = tracer.get_latency_profile()
            return CompilerResponse(
                query=req.query,
                compiled_filter=json.loads(cached_response),
                cache_hit=True,
                similarity_score=sim_score,
                latency_ms=profile.total_latency_ms,
                cost_usd=0.0,  # $0 cost for cache hit!
                model_used="semantic-cache",
            )
    else:
        is_hit, sim_score = False, 0.0

    tracer.end_span("cache_lookup")

    # 2. Cache Miss -> Invoke LLM with System Instruction
    tracer.start_span("llm_inference")
    system_prompt = (
        "You are an expert Text-to-Filter compiler. Convert natural language queries "
        "into strict JSON matching fields: category (str), brand (str), price_min (float), "
        "price_max (float), in_stock (bool), sort_by (str). Return ONLY the raw JSON object."
    )

    try:
        response = client.chat.completions.create(
            model=req.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.query},
            ],
        )
        tracer.end_span("llm_inference")
        content = response.choices[0].message.content or "{}"
        clean_json = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed_output = json.loads(clean_json)

        # Store in cache
        if req.use_cache:
            semantic_cache.set(req.query, json.dumps(parsed_output))

        # Calculate cost & latency
        prompt_tokens = response.usage.prompt_tokens if hasattr(response, "usage") and response.usage else 0
        completion_tokens = response.usage.completion_tokens if hasattr(response, "usage") and response.usage else 0
        cost_profile = tracer.calculate_cost(prompt_tokens, completion_tokens)
        latency_profile = tracer.get_latency_profile()

        return CompilerResponse(
            query=req.query,
            compiled_filter=parsed_output,
            cache_hit=False,
            similarity_score=sim_score,
            latency_ms=latency_profile.total_latency_ms,
            cost_usd=cost_profile.total_cost_usd,
            model_used=req.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compilation error: {str(e)}")


@app.get(
    "/llmops/cache/stats",
    response_model=CacheStatsResponse,
    tags=["Fine-Tuning & LLMOps"]
)
def get_cache_telemetry() -> CacheStatsResponse:
    """Return real-time semantic cache performance metrics."""
    if semantic_cache is None:
        raise HTTPException(status_code=500, detail="Semantic cache uninitialized")
    stats = semantic_cache.get_stats()
    return CacheStatsResponse(**asdict(stats))

