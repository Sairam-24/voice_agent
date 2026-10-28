# Deployment guide

## Status — read this first

* **There is no public deployment of this project.** No URL exists, and none should be claimed anywhere until you have deployed it yourself and checked it.
* The app has been run and tested **locally on Windows 10** (see the README for exactly what was run).
* The `Dockerfile` and `docker-compose.yml` were **written but never built or run**: the development machine has no Docker. I checked the parts I could check without Docker (the base image tags `python:3.13-slim` and `ollama/ollama:0.40.11` exist on Docker Hub; PyPI publishes Linux x86_64 and aarch64 wheels for every compiled dependency, so no compiler should be needed). Treat the first `docker compose up --build` as an experiment and fix what it shows.

## What a deployment has to solve

| Concern | Why it matters here | What to do |
|---|---|---|
| **HTTPS** | Browsers only allow microphone access on `https://` pages or `localhost`. A plain `http://server-ip:8000` page will not be able to record. | Put a TLS-terminating reverse proxy in front (Caddy gets certificates automatically; nginx works too). |
| **WebSocket** | `/ws/voice` is a WebSocket. | Caddy's `reverse_proxy` handles upgrades. For nginx add `proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";` and a long `proxy_read_timeout`. |
| **No authentication** | Anyone who can reach the server can spend your CPU and, with Gemini, your API quota. The in-process limits (`MAX_INFLIGHT_TURNS`, timeouts) are global, not per user. | Do not expose it directly. Put it behind the proxy's basic auth or an OAuth proxy, set a spending cap/quota on the Gemini key, and keep the compose port bound to `127.0.0.1` (already the default). |
| **CPU-bound inference** | Whisper, Piper and (with Ollama) the LLM all run on the CPU and compete for the same cores. See `docs/design-decisions.md`. | Benchmark on the target host (`scripts/benchmark.py`) before promising any latency. Numbers from one machine do not transfer to another. |
| **Model files** | About 140 MB (Whisper `base.en`) + 60 MB (voice) + about 1.3 GB (`llama3.2:1b`). They are not in the image. | Keep them on a persistent volume (compose does: `voice-models`). On hosts with non-persistent disk they are re-downloaded at every start. |
| **Licences** | See the README "Licensing" section. In particular `piper-tts` is GPL-3.0-or-later, and the Llama licence has redistribution conditions. | Do not publish an image that contains model weights unless you have checked each licence. |

## Option A — your own machine (verified)

```
python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

Bind to `127.0.0.1` so only you can reach it.

## Option B — Docker Compose (not run by the author)

```
cp .env.example .env              # PowerShell: Copy-Item .env.example .env
docker compose run --rm app python scripts/download_models.py   # fills the voice-models volume
docker compose up --build
```

* Ollama on the **host**: leave `DOCKER_OLLAMA_URL` unset; the container reaches it at `http://host.docker.internal:11434`. On Linux Ollama must listen beyond loopback for that to work (`OLLAMA_HOST=0.0.0.0` for the Ollama service); that exposes Ollama to your network, so firewall it.
* Ollama in a **container**: `docker compose --profile ollama up --build`, then `docker compose exec ollama ollama pull llama3.2:1b`, and set `DOCKER_OLLAMA_URL=http://ollama:11434` in `.env`.
* `DOCKER_OLLAMA_URL` is deliberately not called `OLLAMA_BASE_URL`: your `.env` usually sets that to `localhost`, which inside a container means the container itself.
* GPU: nothing in the compose file requests one. See "GPU" below.

## Option C — a small VPS or container host with a hosted LLM

The simplest hosted shape is the app container with Gemini doing the answering, so no LLM runs on the server. Leave `GEMINI_API_KEY` empty and visitors enter their own key on the System page (each uses their own quota), or set it to pay for everyone. Without Ollama on the server, a visitor with no key gets an error instead of an answer; Whisper and the local voices are still needed, as the fallbacks. Serve it over HTTPS, since visitors type keys into the page. Prices below were read from each vendor's page on **2026-10-26**; they change, so check the links before relying on them.

| Host | What the vendor page says | Fit for this app |
|---|---|---|
| **Fly.io** ([pricing](https://docs.fly.io/about/pricing)) | `shared-cpu-2x` with 2 GB: $13.39 / 30 days; extra RAM $6.00/GB/month; volumes $0.15/GB/month; "new organizations don't have a free tier". | Plausible. Shared CPUs are the weakest option for inference, so measure. A 1 GB volume for models costs about $0.15/month. |
| **Hetzner Cloud** ([cloud](https://www.hetzner.com/cloud/)) | Shared plans "from €5.99/month" (cost-optimised) and "from €6.49/month" (regular), excluding VAT. The page also showed a "currently not available" notice, and did not list vCPU/RAM per plan when I read it. | Likely the cheapest way to get a normal VM. Check the specs and availability yourself. |
| **Hugging Face Spaces** ([docs](https://huggingface.co/docs/hub/spaces-overview)) | Docker Spaces need a paid plan to create; CPU Basic is 2 vCPU / 16 GB with no hourly cost; free hardware sleeps when unused; the 50 GB disk is not persistent; outbound requests are limited to ports 80, 443 and 8080. | Works only with a hosted LLM (Gemini over 443); a separate Ollama on port 11434 would be blocked. Models re-download after every restart unless you add storage. The sleep behaviour means a cold start for the next visitor. |

Memory: during the benchmark runs the app process had a working set of about 650 MB on Windows (one snapshot, not a profile). Ollama with `llama3.2:1b` needs its own gigabyte or two on top. With Gemini as the LLM, 2 GB of RAM is a reasonable starting point, but I have not tested on any of these hosts.

## GPU

`STT_DEVICE=cuda` with `STT_COMPUTE_TYPE=float16` runs Whisper on an NVIDIA GPU through CTranslate2. It needs an NVIDIA driver and the CUDA 12 runtime libraries (cuBLAS and cuDNN 9) installed alongside the Python packages; the faster-whisper README documents the exact versions, and they change, so follow it. Piper can use `onnxruntime-gpu`, which this project does not wire up. **None of the GPU paths were tested here**: the development machine has only an Intel integrated GPU. Hugging Face lists a small T4 Space at $0.40/hour on the page above.

## Checklist before putting it on the internet

- [ ] TLS in front, WebSocket upgrade working (`wss://` connects).
- [ ] Authentication in front (the app has none).
- [ ] Gemini key stored as a secret in the host, never in the image or repository; quota or budget cap set on the key.
- [ ] `ALLOWED_ORIGINS` only if the page is served from a different origin than the API.
- [ ] Logs reviewed: they contain request ids and timings, not audio or transcripts. Confirm that still holds on your version.
- [ ] A benchmark run on the target host, saved, and the README updated with those measured numbers (not the laptop's).
- [ ] You have read the model licences and are not redistributing weights you may not redistribute.
