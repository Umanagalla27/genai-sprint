import json
import os
import sys

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from groq import Groq

from src.rag.service import RAGService
from src.rag.evaluator import LLMJudgeEvaluator

load_dotenv()

def run_benchmark():
    print("=" * 60)
    print("[*] Starting RAG Production Benchmark: LLM-as-a-Judge")
    print("=" * 60)

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    client = Groq(api_key=groq_key)
    judge = LLMJudgeEvaluator(client=client)
    rag = RAGService(collection_name="benchmark_rag_eval")

    # Load golden dataset
    with open("data/golden_eval_set.json", "r") as f:
        dataset = json.load(f)

    # 1. Ingest all ground truth documents
    print(f"\n[1/3] Ingesting {len(dataset)} documents into knowledge base...")
    for item in dataset:
        rag.ingest_document(text=item["doc_text"], doc_id=item["id"])
    print("[OK] Ingestion complete.\n")

    # 2. Run evaluation
    print("[2/3] Executing questions through RAG pipeline & grading with LLM Judge...")
    results = []

    for item in dataset:
        qid = item["id"]
        question = item["question"]

        # Run through full pipeline (Hybrid + FlashRank Rerank)
        rag_output = rag.answer_query(query=question, llm_client=client)
        answer = rag_output["answer"]
        context_used = rag_output.get("context_used", "")

        # Grade with Judge
        faithfulness = judge.evaluate_faithfulness(context=context_used, answer=answer)
        relevance = judge.evaluate_answer_relevance(question=question, answer=answer)

        results.append({
            "id": qid,
            "question": question[:50] + "...",
            "faithfulness": faithfulness,
            "relevance": relevance,
            "sources_found": len(rag_output["sources"]),
        })
        print(f"  Question {qid}: Faithfulness={faithfulness:.2f} | Relevance={relevance:.2f}")

    # 3. Summary Report
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["relevance"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Questions Evaluated: {len(results)}")
    print(f"Average Faithfulness (Grounding): {avg_faithfulness * 100:.1f}%")
    print(f"Average Answer Relevance:        {avg_relevance * 100:.1f}%")
    print("=" * 60)

    # Save results to data/eval_results.json
    with open("data/eval_results.json", "w") as f:
        json.dump({
            "summary": {
                "avg_faithfulness": avg_faithfulness,
                "avg_relevance": avg_relevance,
                "total_queries": len(results),
            },
            "details": results,
        }, f, indent=2)
    print("[OK] Detailed metrics saved to data/eval_results.json")


if __name__ == "__main__":
    run_benchmark()
