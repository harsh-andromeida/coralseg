---
name: project-core
description: Core project workflow and development standards for ReefVision / coralseg. Enforces committing and pushing to GitHub upon completion of every step.
---

# Project Core Workflow & Guidelines

## 1. Push to GitHub on Completion of Every Step
**MANDATORY RULE**: After completing each development step, task, bug fix, or milestone:
1. **Verify**: Test and verify the requested behavior before considering the step complete.
2. **Review Status**: Run `git status` to verify changed files and ensure no temporary files, dependencies (`node_modules/`, `.venv/`), or large model weights (`*.onnx`, `*.pth`, `*.dmg`, `target/`) are staged.
3. **Commit**: Stage all relevant changes (`git add <files>`) and write a clear, descriptive commit message (`git commit -m "..."`).
4. **Push**: Immediately push to GitHub:
   ```bash
   git push origin main
   ```
   *(or the current feature branch).*

## 2. Project Architecture
- **Desktop Application**: Tauri v2 desktop application pairing a React/Vite UI with a high-performance local FastAPI inference engine in `backend/`.
- **Remote Repository**: `https://github.com/Andromeida-Tech/Andromeida-ReefVision.git`

## 3. Clean Repository Standards
- Always ensure `.gitignore` excludes heavy artifacts:
  - Python: `.venv/`, `__pycache__/`, `build/`, `dist/`
  - Node/Rust: `node_modules/`, `target/`, `dist/`
  - Models/Binaries: `*.onnx`, `*.pth`, `*.pt`, `*.bin`, `*.npy`, `*.dmg`, `*.pkg`
