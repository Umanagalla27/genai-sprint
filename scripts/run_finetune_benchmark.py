import os
import sys
import json
from dotenv import load_dotenv

# Configure UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from groq import Groq
from src.finetuning.benchmark import ModelComparisonBenchmark

load_dotenv()

def main():
    print("=" * 70)
    print("🔬 Benchmarking Zero-Shot vs Few-Shot Token & Latency Trade-offs")
    print("=" * 70)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    benchmark = ModelComparisonBenchmark(client=client)

    # Load validation split
    with open("data/finetune_val.jsonl", "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(val_records)} validation samples.\n")

    zero_shot_results = []
    few_shot_results = []

    for idx, rec in enumerate(val_records):
        query = rec["messages"][1]["content"]
        expected_target = json.loads(rec["messages"][2]["content"])

        print(f"[{idx+1}/{len(val_records)}] Evaluating query: '{query}'")

        # 1. Zero-Shot
        res_zero = benchmark.evaluate_zero_shot(query)
        zero_shot_results.append(res_zero)

        # 2. Few-Shot
        res_few = benchmark.evaluate_few_shot(query)
        few_shot_results.append(res_few)

    # Aggregate Metrics
    def calc_averages(results):
        total = len(results)
        valid_count = sum(1 for r in results if r["is_schema_valid"])
        avg_prompt_tokens = sum(r["prompt_tokens"] for r in results) / total
        avg_latency = sum(r["latency_ms"] for r in results) / total
        return {
            "valid_pct": (valid_count / total) * 100,
            "avg_prompt_tokens": round(avg_prompt_tokens, 1),
            "avg_latency_ms": round(avg_latency, 2),
        }

    zero_metrics = calc_averages(zero_shot_results)
    few_metrics = calc_averages(few_shot_results)

    print("\n" + "=" * 70)
    print("📊 BENCHMARK COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Zero-Shot Baseline':<18} | {'Few-Shot In-Context':<18}")
    print("-" * 70)
    print(f"{'Schema Compliance Rate':<30} | {zero_metrics['valid_pct']:<17.1f}% | {few_metrics['valid_pct']:<17.1f}%")
    print(f"{'Avg Prompt Tokens / Query':<30} | {zero_metrics['avg_prompt_tokens']:<18} | {few_metrics['avg_prompt_tokens']:<18}")
    print(f"{'Avg Inference Latency':<30} | {zero_metrics['avg_latency_ms']:<15} ms | {few_metrics['avg_latency_ms']:<15} ms")
    
    token_savings = ((few_metrics['avg_prompt_tokens'] - zero_metrics['avg_prompt_tokens']) / few_metrics['avg_prompt_tokens']) * 100
    print("-" * 70)
    print(f"💡 Architectural Takeaway: Zero-Shot/Fine-Tuned reduces prompt token overhead by {token_savings:.1f}%, cutting API billing and latency at scale.")
    print("=" * 70)

    # Save to data/finetune_benchmark_results.json
    with open("data/finetune_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "zero_shot_metrics": zero_metrics,
            "few_shot_metrics": few_metrics,
            "token_savings_pct": token_savings,
        }, f, indent=2)
    print("📁 Benchmark results saved to data/finetune_benchmark_results.json")


if __name__ == "__main__":
    main()
