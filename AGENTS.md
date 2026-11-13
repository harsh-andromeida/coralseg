# AGENTS.md

## Project Rules for ReefVision

Inherits all global rules with the following mandatory project-specific rules:

### 1. Push to GitHub on Completion of Every Step
On completion of every development step, task, bug fix, or milestone:
- Verify the implementation first (**Implement → Test → Fix → Re-test**).
- Check `git status` to ensure clean staging without unintended files.
- Commit all changes with a clear, descriptive commit message.
- Push immediately to GitHub:
  ```bash
  git push origin main
  ```
  *(or the active working branch).*

### 2. Workspace Structure
- Standalone Tauri v2 Desktop Application pairing a React 19 / Vite UI with a high-performance local FastAPI backend in `backend/`.
- `backend/`: FastAPI inference engine, ONNX models management, dataset export, and hardware acceleration logic.
- `src-tauri/`: Rust desktop shell, process lifecycle management, and native integration.
- `src/`: Modern React 19 UI with Tailwind CSS and Canvas annotation tools.
- `docs/`: Architecture documentation and legacy reference guides.
- `.agents/skills/`: Workspace-specific agent skills.

### 3. Binary & Weight Handling
Never commit heavy machine learning model weights (`*.onnx`, `*.pth`, `*.pt`, `*.bin`), installers (`*.dmg`, `*.pkg`, `*.exe`), or virtual environments/dependencies (`node_modules/`, `.venv/`, `target/`). Keep `.gitignore` strictly maintained.

### 4. Testing & Pipeline Execution Policy (Do NOT Run Full Pipeline Locally)
**STRICT REQUIREMENT**: Do **NOT** run the complete heavy pipeline on the user's PC for testing every time.
- **On every MAJOR change**:
  1. **Primary (Cloud Colab)**: Test and benchmark first on **Google Colab** (via Colab CLI GPU session). Colab GPU is the default environment and can be used anytime.
  2. **Alternative (Scaled-down Local)**: If testing locally, make or run a **smaller, lightweight version** (e.g. minimal sample dataset, 1-2 video frames, or lightweight mock stages) that can be run safely and quickly on the user's PC without overloading local system resources.
- Reserve local testing strictly for fast smoke checks, unit tests, or scaled-down test cases—never full-scale, end-to-end heavy pipeline processing.
- Use Modal only when explicitly requested.

### 5. Remote Cloud GPU Testing via Google Colab CLI (`colab`)
The Google Colab CLI allows provisioning cloud runtimes with GPU accelerators (T4) and running remote tests/benchmarks directly from the terminal without manual browser interaction.

- **Binary Location**: `/Users/harsh/.local/bin/colab` (or `colab`).
- **Authentication**: Uses Application Default Credentials (ADC) or OAuth2. Check active sessions and auth with:
  ```bash
  colab sessions
  colab whoami
  ```
- **Ephemeral Script Execution (`colab run`)**:
  Provisions a fresh VM, executes the script with arguments, and automatically tears down the VM upon completion:
  ```bash
  colab run --gpu T4 script.py [args...]
  ```
  *(Pass `--keep` if the session should persist after script completion).*
- **Stateful Session Workflow**:
  ```bash
  # 1. Allocate session with desired GPU accelerator
  colab new -s bench-session --gpu T4

  # 2. Install dependencies on the remote VM
  colab install -s bench-session -r backend/requirements.txt

  # 3. Execute script against the live remote kernel
  colab exec -s bench-session -f path/to/script.py

  # 4. Check status and logs
  colab status -s bench-session
  colab log -s bench-session -n 20

  # 5. ALWAYS stop session when finished to release compute resources
  colab stop -s bench-session
  ```
- **Agent Guidelines**:
  - Never run interactive commands (`colab repl`, `colab auth`, `colab drivemount` without piping) inside agent turns as they require a TTY.
  - Remote working directory is `/content`. Use absolute paths for remote file transfers (`colab upload`/`colab download`).
  - Always clean up any provisioned session with `colab stop -s <name>` to prevent wasting compute units.

### 6. Remote Cloud GPU Testing & Serverless Benchmarking via Modal CLI (`modal`)
Modal provides serverless cloud execution for horizontal scaling and high-end GPUs (e.g. NVIDIA H100, A100, L4).

- **Binary Location**: `/Users/harsh/.local/bin/modal` (or `modal`).
- **Profile & Authentication**: Profile `ctrlxharsh`. Verify connection with:
  ```bash
  modal profile current
  modal app list
  ```
- **Running Functions & Entrypoints (`modal run`)**:
  Execute remote functions or benchmarks defined with Modal decorators:
  ```bash
  modal run script.py [--arg value]
  ```
- **Interactive Container Debugging (`modal shell`)**:
  Inspect a container environment with:
  ```bash
  modal shell script.py
  ```
- **App Lifecycle & Dev Server**:
  ```bash
  modal serve app.py     # Live web endpoint development with hot-reloading
  modal deploy app.py    # Persistent app deployment
  modal app list         # View running or deployed apps
  modal app stop <app_id> # Terminate running app
  ```
- **GPU Allocation Pattern in Python**:
  ```python
  import modal

  app = modal.App("reefvision-gpu-bench")
  image = modal.Image.debian_slim().pip_install("torch", "onnxruntime-gpu", "opencv-python-headless")

  @app.function(image=image, gpu="H100", timeout=600)
  def run_remote_benchmark():
      import torch
      return f"Running on: {torch.cuda.get_device_name(0)}"
  ```
- **Agent Guidelines**:
  - Run commands without an attached terminal (no TTY) so output streams as clean log lines without spinner ANSI escape codes.
  - Mount persistent data or weights using `modal.Volume` instead of uploading large files per run.

### 7. Code Map (read this before searching)

Search tip: `.ignore` hides `dist/`, `.agents/`, `demo_images/`, lock files. Do not read `.agents/skills/` unless a task names a skill; it is vendored third-party material, not project code.

**Backend (`backend/`, Python/FastAPI, run via `backend/run_server.py`)**

| Path | Responsibility |
|---|---|
| `app/main.py` | App factory, CORS, router mounting, serves `dist/` |
| `app/state.py` | Module-level singletons: image store, caches, `BatchProcessor`, `VideoProcessor`, demo-dir resolution |
| `app/schemas.py` | Pydantic request models |
| `app/routers/system.py` | Health, device select, model status/download/delete |
| `app/routers/images.py` | Samples, register image, segment, enrich, overlay, SAM prompt, save annotations, COCO/CSV export |
| `app/routers/batch.py` | Batch start/pause/resume/cancel/status, folder export (not `core/batch.py`) |
| `app/routers/video.py` | Video upload, process, status/frames, pause/cancel, exports |
| `core/models.py` | ONNX session loading, HF download, model locks, active-job counter |
| `core/device.py` | Hardware tier resolution (CUDA/CoreML/CPU), memory budget, session options |
| `core/segmentation.py` | CoralSCOP SAM automatic mask generation + NMS (`run_segmentation`) |
| `core/interactive_segmentation.py` | Point/box prompt SAM, mask<->polygon, `SAM_SESSION` embedding cache |
| `core/taxonomy.py` | BioCLIP genus/growth-form + bleaching classification, `enrich_masks_with_taxonomy_and_bleaching` |
| `core/visualization.py` | Overlay rendering, distinct colours |
| `core/batch.py` | `BatchProcessor` worker thread (image batches); the route layer is `app/routers/batch.py` |
| `core/export.py` | COCO JSON/zip/dataset export for images |
| `core/video/engine.py` | `VideoProcessor`; `_run_job_pipeline` is the whole video pipeline (prepare, discover, track, classify, persist, export) |
| `core/video/store.py` | `VideoJobStore` SQLite (WAL) persistence for jobs/chunks/frames/metrics |
| `core/video/scheduler.py` | Chunk planning, worker admission, identity reconciliation across chunks |
| `core/video/preprocessor.py` | 25 fps standardisation, bounded frame extraction |
| `core/video/keyframe_selector.py` | Ego-motion keyframe selection |
| `core/video/tracker.py` | `CoralTrack`, `BenthicColonyTracker` (IoU identity matching) |
| `core/video/sam2_tracker.py`, `efficienttam_tracker.py` | Neural temporal mask propagation backends |
| `core/video/export.py` | Annotated MP4, transect CSV, COCO video export (not `core/export.py`) |
| `test_api.py`, `test_video_pipeline.py` | Existing tests; benchmark/Colab scripts also live in `backend/` |

**Frontend (`src/`, React 19 + Vite, dev port 1425)**

| Path | Responsibility |
|---|---|
| `services/api.ts` | Every backend call (50 functions); base URL logic in `getApiBase` |
| `types/index.ts` | Shared API/domain types |
| `views/UploadView.tsx` | Upload, model/device setup, starting image batches and video jobs |
| `views/AnalysisView.tsx` | Image results browsing and editing |
| `views/VideoAnalysisView.tsx` | Video results playback, timeline, export |
| `components/segmentation/` | Konva editor: canvas, hotkeys, history, segment manager |
| `components/upload/`, `components/video/` | Sub-panels for those views |
| `components/ui/` | shadcn/Radix primitives; do not hand-edit |
| `styles/` | Global CSS and theme tokens |

**Desktop shell (`src-tauri/src/lib.rs`)**: spawns `backend/run_server.py` from `.venv`, health-checks `127.0.0.1:8000`, native save dialog command.

**Data flow**
- Image: `register` -> `/api/analysis/segment` (SAM) -> `/enrich` (BioCLIP + bleaching) -> `/overlay`; batch mode runs the same steps in `core/batch.py`. Results live in `app/state.py` caches (in memory only).
- Video: `/api/video/upload` -> `/process` -> `VideoProcessor` thread -> frames written to SQLite -> UI polls `/api/video/status` and frame deltas -> export endpoints.

**Known name collisions**: `batch.py` (routers vs core), `export.py` (core vs core/video). Always use the full path.
