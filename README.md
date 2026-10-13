# Andromeida Reef Vision Studio 🪸

A high-performance, standalone desktop application for AI-powered coral reef segmentation, taxonomic classification, bleaching condition assessment, and automated COCO dataset generation.

Built with **Tauri v2**, **React 19**, **FastAPI**, and **ONNX Runtime (Zero PyTorch)**.

---

## 🌟 Key Features

- **Dense Instance Segmentation**: Ultra-fast ONNX-accelerated foundation models for pixel-precise coral boundary segmentation.
- **Interactive Point & Box Prompting**: Segment-Anything (SAM) powered interactive prompt refinement with real-time Moore-Neighbor contour tracing and RDP polygon simplification.
- **Taxonomic & Morphological Classification**: Zero-shot BioCLIP embeddings identifying coral genera and growth forms (Branching, Massive, Encrusting, Foliose, Plate, Solitary).
- **Bleaching & Health Assessment**: Real-time condition estimation classifying healthy vs. bleached coral colonies.
- **High-Throughput Batch Processing**: Queue entire directories of underwater imagery with pause, resume, prioritize, and multi-thread hardware acceleration controls.
- **Standard COCO Dataset Export**: Export complete dataset archives with verified polygons, category definitions, and summary reports ready for computer vision pipelines.

---

## 🏗️ Architecture

```
coralseg/
├── src/                  # React 19 + TypeScript Frontend (Tailwind CSS, Konva, Radix UI)
├── src-tauri/            # Tauri v2 Desktop Core (Rust shell, process lifecycle, IPC)
├── backend/              # Local FastAPI Vision Engine (Zero-PyTorch ONNX Runtime)
│   ├── app/              # FastAPI routers, endpoints, and schemas
│   ├── core/             # Segmentation, taxonomy, batch engine, hardware detection
│   └── requirements.txt  # Dedicated Python backend requirements
├── demo_images/          # Bundled sample coral reef imagery
├── docs/                 # Architectural specifications and legacy documentation
└── start_backend.sh      # Self-bootstrapping FastAPI runner script
```

---

## 🚀 Getting Started

### Prerequisites

- **Node.js**: >= 18.x
- **Python**: >= 3.10
- **Rust & Cargo**: >= 1.77+ (for Tauri builds)

### 1. Install Dependencies

```bash
# Install frontend dependencies
npm install

# Initialize Python backend environment (auto-creates .venv and installs requirements)
npm run setup:backend
```

### 2. Run in Development Mode

```bash
# Starts both the FastAPI backend and Vite frontend concurrently
npm run dev

# Or launch as native desktop application in dev mode:
npm run tauri dev
```

### 3. Production Build

```bash
# Build production bundle
npm run build

# Package desktop installer (macOS .dmg / Windows .exe / Linux .deb)
npm run tauri build
```

---

## 🧪 Testing

Run the full FastAPI vision engine test suite:

```bash
npm run test:backend
# or
python3 backend/test_api.py
```

---

## 📄 License

Proprietary — Andromeida Technologies. All rights reserved.
