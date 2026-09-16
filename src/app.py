import os
from contextlib import asynccontextmanager
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from groq import Groq, APIError

load_dotenv()

client: Groq | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")
    client = Groq(api_key=api_key)
    print("INFO: Groq client initialized successfully.")
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
