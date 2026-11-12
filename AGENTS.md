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

### 4. GPU usage for local inference and testing
NOTE: use Colab GPU as default you can use it anytime.. but use modal only when asked. 

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

### 5. Remote Cloud GPU Testing & Serverless Benchmarking via Modal CLI (`modal`)
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
