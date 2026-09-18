import os
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Generator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from groq import Groq, APIError
from src.rag.service import RAGService

load_dotenv()

client: Groq | None = None
rag_service: RAGService | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, rag_service
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    client = Groq(api_key=api_key)
    print("INFO: Groq client initialized successfully.")
    rag_service = RAGService()
    print("INFO: RAG service initialized.")
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

