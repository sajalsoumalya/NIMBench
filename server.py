import os
import json
import time
import asyncio
from contextlib import asynccontextmanager
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"
RESULTS_FILE = "results.json"
PROMPT = "What is the capital of India? Please answer in one sentence."
RATE_LIMIT_WAIT = 12
REQUEST_TIMEOUT = 60

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=REQUEST_TIMEOUT)

benchmark_queue: asyncio.Queue = asyncio.Queue()
validate_queue: asyncio.Queue = asyncio.Queue()
is_running = False
is_validating = False
current_model = ""
progress = {"total": 0, "done": 0, "status": "idle"}

# Model availability cache: {model_id: "unknown"|"available"|"unavailable"}
model_status: dict[str, str] = {}
validation_done = False

# In-memory results cache — serves GET /api/results without touching disk
cached_results: list = []
results_loaded = False


def ensure_results_cached():
    global cached_results, results_loaded
    if not results_loaded:
        cached_results = load_results_from_disk()
        results_loaded = True


def load_results_from_disk():
    try:
        with open(RESULTS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_results_to_disk(results):
    tmp = RESULTS_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, RESULTS_FILE)
    except Exception:
        pass


def make_label(model_id: str) -> str:
    return model_id.split("/")[-1].replace("-", " ").replace("_", " ").title()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_results():
    try:
        with open(RESULTS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def benchmark_model_sync(model_id: str, label: str, prompt: str) -> dict:
    start = time.perf_counter()
    token_count = 0
    first_token_time = None

    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
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

        elapsed = time.perf_counter() - start
        ttft = (first_token_time - start) if first_token_time else None
        tps = token_count / elapsed if elapsed > 0 else 0

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


def check_model_available_sync(model_id: str) -> str:
    """Returns 'available', 'unavailable', or 'unknown'."""
    try:
        stream = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            max_tokens=10,
            temperature=0.1,
            stream=True,
        )
        for _ in stream:
            break
        return "available"
    except Exception as e:
        msg = str(e)
        # 404 / 410 = model genuinely gone
        if "404" in msg or "410" in msg:
            return "unavailable"
        # Everything else (429 rate limit, 503, timeout, etc.) = uncertain
        return "unknown"


async def run_validation(model_list: list[dict]):
    global is_validating, model_status, validation_done
    is_validating = True
    model_status = {m["id"]: "unknown" for m in model_list}
    total = len(model_list)

    await validate_queue.put({"type": "validate_start", "total": total})

    sem = asyncio.Semaphore(3)

    async def validate_one(model_id: str):
        async with sem:
            current = model_status.get(model_id)
            if current != "unknown":
                return
            model_status[model_id] = "checking"
            result = await asyncio.to_thread(check_model_available_sync, model_id)
            model_status[model_id] = result
            await validate_queue.put({
                "type": "validate_result",
                "model_id": model_id,
                "status": result,
            })
            await asyncio.sleep(0.3)

    tasks = [validate_one(m["id"]) for m in model_list]
    await asyncio.gather(*tasks)

    validation_done = True
    is_validating = False
    await validate_queue.put({"type": "validate_done"})


async def run_benchmarks(model_ids: list, prompt: str):
    global is_running, current_model, progress, cached_results
    ensure_results_cached()
    results = cached_results[:]
    done_ids = {r["id"] for r in results}

    valid_ids = [mid for mid in model_ids if model_status.get(mid) in ("available", "unknown")]
    if not valid_ids:
        await benchmark_queue.put({
            "type": "error",
            "message": "None of the selected models are available. Try validating first.",
        })
        await benchmark_queue.put({"type": "done", "total": 0})
        is_running = False
        progress["status"] = "done"
        return

    total = len(valid_ids)
    skipped = len(model_ids) - len(valid_ids)
    progress = {"total": total, "done": 0, "status": "running"}

    label_map = {}
    try:
        resp = client.models.list()
        for m in resp.data:
            label_map[m.id] = m.id
    except Exception:
        pass

    await benchmark_queue.put({"type": "start", "total": total, "skipped": skipped})

    for i, mid in enumerate(valid_ids):
        current_model = mid
        if mid in done_ids:
            await benchmark_queue.put({
                "type": "skip", "model_id": mid,
                "done": i + 1, "total": total,
            })
            continue

        label = label_map.get(mid, make_label(mid))

        await benchmark_queue.put({
            "type": "progress", "model_id": mid, "label": label,
            "done": i, "total": total, "status": "testing",
        })

        result = await asyncio.to_thread(benchmark_model_sync, mid, label, prompt)
        results.append(result)
        cached_results = results[:]
        save_results_to_disk(results)
        progress["done"] = i + 1

        await benchmark_queue.put({
            "type": "result", **result, "done": i + 1, "total": total,
        })

        if result["status"] == "error" and "429" in result.get("error", ""):
            await asyncio.sleep(RATE_LIMIT_WAIT * 2)
        else:
            await asyncio.sleep(RATE_LIMIT_WAIT)

    await benchmark_queue.put({"type": "done", "total": total})
    is_running = False
    progress["status"] = "done"
    current_model = ""


@app.get("/api/models")
async def list_models():
    global model_status
    try:
        resp = client.models.list()
        models_list = resp.data
    except Exception:
        try:
            with open("models.json") as f:
                static = json.load(f)
            for m in static:
                model_status.setdefault(m["id"], "unknown")
            return {"models": static, "source": "fallback", "statuses": model_status}
        except FileNotFoundError:
            return {"models": [], "source": "none", "statuses": {}}

    models = [
        {"id": m.id, "owned_by": getattr(m, "owned_by", ""), "created": getattr(m, "created", 0)}
        for m in models_list
    ]

    # Seed status for any new model IDs
    for m in models:
        model_status.setdefault(m["id"], "unknown")

    # Kick off background validation if not already done/doing
    if not is_validating and not validation_done:
        asyncio.create_task(run_validation(models))

    return {"models": models, "source": "api", "statuses": model_status}


@app.get("/api/models/status")
async def get_model_status():
    return {
        "statuses": model_status,
        "validating": is_validating,
        "done": validation_done,
    }


@app.get("/api/models/validate/events")
async def validate_events(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(validate_queue.get(), timeout=1.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "validate_done":
                    break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/models/validate/start")
async def trigger_validation():
    global validation_done, is_validating
    if is_validating:
        return {"status": "already_validating"}
    validation_done = False
    try:
        resp = client.models.list()
        models = [{"id": m.id} for m in resp.data]
    except Exception:
        return {"status": "error", "message": "Cannot fetch model list"}
    asyncio.create_task(run_validation(models))
    return {"status": "started", "count": len(models)}


@app.post("/api/benchmark/start")
async def start_benchmark(data: dict):
    global is_running
    model_ids = data.get("model_ids", [])
    prompt = data.get("prompt", PROMPT)

    if not model_ids:
        return {"error": "No models selected"}
    if is_running:
        return {"error": "Benchmark already running"}

    is_running = True
    while not benchmark_queue.empty():
        try:
            benchmark_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

    asyncio.create_task(run_benchmarks(model_ids, prompt))
    return {"status": "started", "count": len(model_ids)}


@app.get("/api/benchmark/events")
async def benchmark_events(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(benchmark_queue.get(), timeout=1.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/benchmark/status")
async def get_benchmark_status():
    return {
        "running": is_running,
        "current_model": current_model,
        "progress": progress,
    }


@app.get("/api/results")
async def get_results():
    ensure_results_cached()
    return {"results": cached_results}


@app.delete("/api/results")
async def clear_results():
    global cached_results, results_loaded
    cached_results = []
    results_loaded = True
    if os.path.exists(RESULTS_FILE):
        try:
            os.remove(RESULTS_FILE)
        except Exception:
            pass
    if os.path.exists(RESULTS_FILE + ".tmp"):
        try:
            os.remove(RESULTS_FILE + ".tmp")
        except Exception:
            pass
    return {"status": "cleared"}


@app.get("/")
async def serve_index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>index.html not found</h1>")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
