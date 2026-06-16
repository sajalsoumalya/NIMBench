# NVIDIA NIM — Model Speed Benchmark

Dynamically discovers available NVIDIA NIM models, benchmarks their latency and throughput, and displays results in an interactive web dashboard.

## Setup

```bash
pip install -r requirements.txt
```

Add your API key to `.env`:
```
NVIDIA_API_KEY=your_key_here
```

## Usage

### Web App (recommended)

```bash
python server.py
# open http://localhost:8000
```

Three tabs:
1. **Models** — Browse all available NVIDIA NIM models (dynamically fetched), search and select which to test
2. **Benchmark** — Configure prompt, run the benchmark, watch real-time progress via SSE
3. **Results** — View TPS/TTFT bar charts, summary cards, and a detailed results table

### CLI (original)

```bash
python benchmark.py --restart   # fresh run
python benchmark.py --resume    # resume interrupted run
```

## Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI web server (model discovery + benchmarking + UI) |
| `index.html` | Web dashboard frontend |
| `benchmark.py` | Original CLI benchmark script |
| `models.json` | Static model list (fallback if API fails) |
| `results.json` | Benchmark output (auto-generated) |
