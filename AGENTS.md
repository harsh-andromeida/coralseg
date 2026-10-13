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
