import os
import json
import time
import sys
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="NIMBench — NVIDIA NIM LLM Speed Benchmark")
group = parser.add_mutually_exclusive_group()
group.add_argument("--resume", action="store_true", help="Continue from last completed model (default)")
group.add_argument("--restart", action="store_true", help="Delete results.json and start from scratch")
args = parser.parse_args()

API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"
PROMPT = "What is the capital of India? Please answer in one sentence."
RESULTS_FILE = "results.json"
RATE_LIMIT_WAIT = 12  # seconds to wait on 429
REQUEST_TIMEOUT = 60  # seconds before giving up on a hung model

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=REQUEST_TIMEOUT)

with open("models.json") as f:
    models = json.load(f)

# Load existing results so we can resume
if args.restart and os.path.exists(RESULTS_FILE):
    os.remove(RESULTS_FILE)
    print("🗑  Deleted results.json — starting fresh.")

try:
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    done_ids = {r["id"] for r in results}
    print(f"Resuming — {len(done_ids)} models already done, {len(models) - len(done_ids)} remaining.")
except FileNotFoundError:
    results = []
    done_ids = set()
    print(f"Starting fresh — {len(models)} models to benchmark.")


def benchmark_model(model_id, label):
    print(f"\n→ {label} ({model_id})")
    try:
        start = time.perf_counter()
        token_count = 0
        first_token_time = None

        stream = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.7,
            max_tokens=300,
            stream=True,
        )

        full_text = ""
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and getattr(delta, "content", None):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                full_text += delta.content
                token_count += 1
                print(".", end="", flush=True)

        elapsed = time.perf_counter() - start
        ttft = (first_token_time - start) if first_token_time else None
        tps = token_count / elapsed if elapsed > 0 else 0

        print(f"\n   {token_count} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")
        return {
            "id": model_id,
            "label": label,
            "status": "ok",
            "total_time_s": round(elapsed, 3),
            "tokens": token_count,
            "tokens_per_second": round(tps, 2),
            "ttft_s": round(ttft, 3) if ttft else None,
            "response_preview": full_text[:120],
        }

    except Exception as e:
        msg = str(e)
        print(f"\n   ERROR: {msg[:120]}")
        return {
            "id": model_id,
            "label": label,
            "status": "error",
            "error": msg[:200],
            "total_time_s": None,
            "tokens": 0,
            "tokens_per_second": 0,
            "ttft_s": None,
        }


for model in models:
    mid = model["id"]
    if mid in done_ids:
        print(f"  skip (done): {model['label']}")
        continue

    result = benchmark_model(mid, model["label"])
    results.append(result)

    # Save after every model so we don't lose progress
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Rate limit handling: wait between every call
    if result["status"] == "error" and "429" in result.get("error", ""):
        print(f"   Rate limited — waiting {RATE_LIMIT_WAIT * 2}s...")
        time.sleep(RATE_LIMIT_WAIT * 2)
    else:
        time.sleep(RATE_LIMIT_WAIT)

print(f"\n\nDone! Results saved to {RESULTS_FILE}")
print("\nTop 5 by tokens/second:")
ranked = sorted([r for r in results if r["status"] == "ok"], key=lambda x: x["tokens_per_second"], reverse=True)
for i, r in enumerate(ranked[:5], 1):
    print(f"  {i}. {r['label']}: {r['tokens_per_second']} tok/s")
