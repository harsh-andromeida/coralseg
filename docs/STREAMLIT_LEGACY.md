# 🪸 Andromeida Reef Vision Studio

Autonomous multi-model coral reef instance segmentation, taxonomy classification, and condition assessment powered by high-performance **ONNX Runtime**.

Zero PyTorch runtime footprint — sub-second inference, hardware-accelerated cross-platform execution, and ultra-lightweight desktop packaging.

---

## 🏛️ Model Architecture & Attribution

All models in Andromeida Reef Vision Studio are optimized and hosted as high-performance ONNX models in the public repository:  
🔗 **[harsh-awasthi/Andromeida-ReefVision-ONNX](https://huggingface.co/harsh-awasthi/Andromeida-ReefVision-ONNX)**

We gratefully acknowledge and credit the original research groups and creators of the foundational models:

| Component | Architecture | Original Model & Source | Original Authors / Research Group | License |
| :--- | :--- | :--- | :--- | :--- |
| **Instance Segmentation** | SAM ViT-B (Encoder + Decoder ONNX) | [reefsupport/CoralSCOP](https://huggingface.co/reefsupport/CoralSCOP) | **Reef Support** & **Meta AI Research** ([Segment Anything](https://github.com/facebookresearch/segment-anything)) | Apache 2.0 |
| **Taxonomy Classification** | BioCLIP ViT-B/16 + Text Projection Matrix | [ReefNet/finetuned-bioclip](https://huggingface.co/ReefNet/finetuned-bioclip) | **ReefNet** & **Imageomics Institute** ([BioCLIP: Foundation Model for the Tree of Life](https://imageomics.github.io/bioclip/)) | MIT |
| **Condition & Bleaching** | YOLO11n-cls ONNX | [NMFS-OSI/yolo11n-cls-noaa-esd-coral-bleaching](https://huggingface.co/NMFS-OSI/yolo11n-cls-noaa-esd-coral-bleaching) | **NOAA Fisheries** (NMFS Open Science Initiative / Ecosystem Sciences Division) | AGPL-3.0 |

---

## ✨ Features
- **Dense Instance Segmentation**: Automated coral reef delineation and boundary detection via CoralSCOP SAM ViT-B.
- **Taxonomic Classification**: 18-class coral taxa and growth form identification via fine-tuned BioCLIP embeddings.
- **Condition & Bleaching Assessment**: Real-time coral reef bleaching severity classification via NOAA YOLO11n.
- **Interactive Multi-Color Overlay & Export**: High-contrast color assignments per detected coral colony, alpha/border controls, and export to standard **COCO JSON** and PNG overlays.
- **Native Standalone Window**: Launches directly as a native desktop window (Cocoa WebKit on macOS, Edge WebView2 on Windows) without launching an external web browser.
- **Ultra-Lightweight Pure ONNX Engine**: Zero PyTorch and zero OpenCV dependencies. Boots instantly with minimal memory usage.
- **Cross-Platform Hardware Acceleration**:
  - **Apple Silicon (macOS)**: Native hardware acceleration via Apple CoreML Execution Provider on Apple M-Series GPUs and Apple Neural Engine (ANE), with multi-threaded SIMD/NEON fallback.
  - **NVIDIA CUDA**: Hardware accelerated inference via CUDA Execution Provider on NVIDIA GPUs.
  - **Windows (DirectML)**: Accelerated on ANY DirectX 12 GPU (NVIDIA, AMD Radeon, Intel Arc).
  - **Multi-Platform CPU**: Threaded SIMD/NEON vector acceleration across all architectures.

---

### 🚀 1-Click Automated Build Scripts

The repository includes zero-friction single-click build scripts for both macOS and Windows. They automatically detect/install Python if missing, configure `.venv`, install dependencies, compile the application with PyInstaller, package it, and open it immediately for installation:

| Platform | Single-Click Command / Action | Output Artifact | Auto-Action |
| :--- | :--- | :--- | :--- |
| **macOS** | `./build_mac.sh` | `dist/AndromeidaReefVision-macOS.dmg` | Auto-opens DMG for drag-and-drop install |
| **Windows** | Double-click `build_win.bat` *(or `.\build_win.ps1`)* | `dist\AndromeidaReefVision\AndromeidaReefVision.exe` | Auto-launches `.exe` & reveals in Explorer |

---

### 🍎 1. macOS (Apple Silicon M1–M4 & Intel)

#### Option A: Pre-built Package (For End-Users)
You can distribute and install the self-contained DMG installer: **`dist/AndromeidaReefVision-macOS.dmg`** (**~107 MB**).

1. **Download & Mount**:
   - Double-click `AndromeidaReefVision-macOS.dmg` to open the installer disk image.
2. **Drag & Drop to Install**:
   - Drag `AndromeidaReefVision.app` into the `Applications` folder shortcut:
     $$\text{[ AndromeidaReefVision.app ]} \xrightarrow{\quad\text{drag}\quad} \text{[ Applications ]}$$
3. **Launch the Application**:
   - Open **Launchpad** or navigate to `/Applications` and click **Andromeida Reef Vision**.
   - The app opens directly in its own **native desktop window** with the official coral mark in the macOS Dock.
4. **First Launch (Model Weights)**:
   - Click **"Download Foundation Models Now"** to stream the lightweight ONNX models into persistent local cache (`~/.cache/huggingface/`). Subsequent launches load instantly offline.

> [!TIP]
> **macOS Gatekeeper Verification Note**:
> Because the application is self-distributed without an Apple Developer certificate, macOS may display:  
> *"AndromeidaReefVision cannot be opened because the developer cannot be verified"*.
> - **Method 1 (GUI)**: Right-click (or Control-click) `AndromeidaReefVision.app` in `/Applications` $\rightarrow$ select **Open** $\rightarrow$ click **Open**.
> - **Method 2 (Terminal)**: Run the quarantine removal command:
>   ```bash
>   xattr -cr /Applications/AndromeidaReefVision.app
>   ```

---

#### Option B: 1-Click Build on macOS (For Developers)

Simply run:
```bash
./build_mac.sh
```

**What the script does automatically:**
1. Verifies Python 3.10+ (automatically installs Python 3.12 via Homebrew if not found).
2. Creates and activates virtual environment (`.venv`).
3. Installs/updates all dependencies (`requirements.txt`).
4. Compiles the native `.app` bundle with PyInstaller.
5. Packages into a Drag-and-Drop disk image (`dist/AndromeidaReefVision-macOS.dmg`).
6. Automatically mounts and opens the DMG in Finder so you can drag-and-drop install immediately!

---

### 🪟 2. Windows (10 & 11, 64-bit)

#### Option A: 1-Click Build on Windows (Double-Click or PowerShell)

Simply double-click:
```
build_win.bat
```
*(Or in PowerShell: `powershell -ExecutionPolicy Bypass -File build_win.ps1`)*

**What the script does automatically:**
1. Checks for Python (automatically installs Python 3.12 via `winget` or python.org silent installer if missing).
2. Sets up the virtual environment (`.venv`).
3. Installs all required dependencies (`requirements.txt`).
4. Compiles the standalone application using PyInstaller into `dist\AndromeidaReefVision\`.
5. Automatically launches `AndromeidaReefVision.exe` and highlights it in Windows Explorer for immediate access!

---

#### Option B: Portable Package (For End-Users)
On Windows, the application requires **zero installer wizards** and runs portably.

1. **Extract**:
   - Unzip `AndromeidaReefVision.zip` into any destination (e.g. `C:\Program Files\AndromeidaReefVision\` or `Desktop\AndromeidaReefVision\`).
2. **Launch**:
   - Double-click `AndromeidaReefVision.exe`.
   - The native desktop window opens automatically (powered by Windows built-in Microsoft Edge WebView2 runtime).
3. **First Launch (Model Weights)**:
   - Click **"Download Foundation Models Now"** to stream ONNX models into persistent user cache (`%USERPROFILE%\.cache\huggingface\`).

---

#### Manual Build on Windows (Optional)
If you prefer running commands step-by-step:
```cmd
git clone https://github.com/<your-org>/coralseg.git
cd coralseg

python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

:: Enable Hardware GPU Acceleration on ANY Windows GPU (NVIDIA, AMD Radeon, Intel Arc):
pip install onnxruntime-directml

:: Build application:
pyinstaller --clean -y coralseg.spec
```
*Output*: `dist\AndromeidaReefVision\AndromeidaReefVision.exe`

---

### 🐧 3. Linux (Ubuntu, Debian, Fedora, Arch)

1. **Install Dependencies**:
   ```bash
   sudo apt update && sudo apt install -y python3-venv python3-pip
   ```

2. **Clone & Setup**:
   ```bash
   git clone https://github.com/<your-org>/coralseg.git
   cd coralseg

   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run Application**:
   ```bash
   streamlit run app.py
   ```

---

### ⚡ 4. Cloud Notebooks (Google Colab & Kaggle)

In a code cell:
```python
# 1. Clone repository
!git clone https://github.com/<your-org>/coralseg.git
%cd coralseg

# 2. Install requirements
!pip install -r requirements.txt

# 3. Launch with Localtunnel tunnel
!python run_colab_kaggle.py
```
Or directly via terminal:
```bash
streamlit run app.py --server.port 8501 --server.headless true & npx localtunnel --port 8501
```

---

## 📊 Build Target & Size Breakdown

By migrating from PyTorch to **pure ONNX Runtime** and replacing OpenCV with **Pillow**:

| Platform / Target | Acceleration | Uncompressed Bundle | Distribution Package | Window Type |
| :--- | :--- | :--- | :--- | :--- |
| **macOS (Installer)** | Apple Silicon / Intel SIMD | **~207 MB** | **~111 MB** (`.dmg` installer) | Native Standalone Window |
| **Windows: Directory Bundle** | DirectML GPU / CPU | **~195 MB** | **~90 MB** (`.zip` archive) | Native Standalone Window |
| **Windows: Single-File (`.exe`)** | DirectML GPU / CPU | **~98 MB** | **~98 MB** (`.exe` standalone) | Native Standalone Window |

---

## 🚢 Offline & Air-Gapped Workstations

If deploying to a field vessel, research station, or laboratory without internet connectivity:

1. Download the 5 ONNX model files from [harsh-awasthi/Andromeida-ReefVision-ONNX](https://huggingface.co/harsh-awasthi/Andromeida-ReefVision-ONNX):
   - `sam_image_encoder.onnx`
   - `sam_mask_decoder.onnx`
   - `bioclip_visual.onnx`
   - `coral_taxonomy_embeddings.npy`
   - `bleaching_yolo11n.onnx`
2. Create an `onnx_models/` folder next to the executable:
   - **On macOS**: Place inside `dist/onnx_models/` or inside `/Applications/AndromeidaReefVision.app/Contents/Resources/onnx_models/`.
   - **On Windows**: Place inside `dist\AndromeidaReefVision\onnx_models\`.
3. The application detects local `onnx_models/` on launch and unlocks analysis with zero network access required.

---

## 🛠️ Troubleshooting & Advanced Options

- **Traditional Browser Mode**:
  If you prefer running in your standard web browser instead of the native desktop window (e.g. for development or debugging):
  ```bash
  ./dist/AndromeidaReefVision --browser
  ```
- **Custom Port or Headless Server**:
  Pass standard flags directly to the executable:
  ```bash
  # Custom port:
  ./dist/AndromeidaReefVision --browser --server.port 8502

  # Headless mode for remote servers:
  ./dist/AndromeidaReefVision --browser --server.headless true
  ```
- **macOS Quarantine Bypass**:
  ```bash
  xattr -cr /Applications/AndromeidaReefVision.app
  ```

