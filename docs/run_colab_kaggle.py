#!/usr/bin/env python3
"""
ReefVision Studio - Google Colab & Kaggle GPU Cloud Runner.
Configures NVIDIA CUDA hardware acceleration for ONNX Runtime and PyTorch,
ensures production static frontend is available, launches the FastAPI vision engine,
and exposes it securely via Cloudflare Tunnel.
"""

import subprocess
import time
import os
import sys
import re
import urllib.request
import platform
import stat
import shutil
import socket

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
DIST_DIR = os.path.join(ROOT_DIR, "dist")


def get_cloudflared_binary() -> str:
    """Download cloudflared binary if not present."""
    sys_cf = shutil.which("cloudflared")
    if sys_cf:
        return sys_cf

    local_bin = os.path.join(ROOT_DIR, "cloudflared")
    if os.path.exists(local_bin):
        return local_bin

    system = platform.system().lower()
    machine = platform.machine().lower()

    url = None
    if system == "linux":
        if "arm" in machine or "aarch64" in machine:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
        else:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    elif system == "darwin":
        if "arm" in machine:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-arm64"
        else:
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64"

    if url:
        try:
            print("📦 Downloading Cloudflare Tunnel binary...", flush=True)
            urllib.request.urlretrieve(url, local_bin)
            st = os.stat(local_bin)
            os.chmod(local_bin, st.st_mode | stat.S_IEXEC)
            return local_bin
        except Exception as e:
            print(f"⚠️ Failed to download cloudflared: {e}", flush=True)

    return ""


def get_cuda_library_paths() -> list:
    """Finds all nvidia and CUDA shared library directories in site-packages and system."""
    paths = []
    for p in sys.path:
        if "site-packages" in p or "dist-packages" in p:
            nvidia_dir = os.path.join(p, "nvidia")
            if os.path.isdir(nvidia_dir):
                for sub in os.listdir(nvidia_dir):
                    lib_dir = os.path.join(nvidia_dir, sub, "lib")
                    if os.path.isdir(lib_dir):
                        paths.append(lib_dir)
            torch_lib = os.path.join(p, "torch", "lib")
            if os.path.isdir(torch_lib):
                paths.append(torch_lib)

    for sys_p in (
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/opt/cuda/lib64",
    ):
        if os.path.isdir(sys_p):
            paths.append(sys_p)

    return paths


def free_port(port: int = 8000):
    """Ensure port 8000 is not blocked by previous crashed/zombie processes."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                print(f"  ⚠️ Port {port} is already in use. Freeing port...", flush=True)
                if shutil.which("fuser"):
                    subprocess.run(["fuser", "-k", f"{port}/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(1)
    except Exception:
        pass


def setup_colab_gpu_environment():
    """
    Ensures onnxruntime-gpu is installed and CUDA/cuDNN shared libraries are linked.
    Resolves the dominant Colab issue where CUDA is present in Torch but missing in ORT.
    """
    print("\n[1/4] Inspecting hardware acceleration and CUDA drivers...", flush=True)
    has_gpu = False
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            print(f"  ✅ NVIDIA GPU detected via PyTorch: {dev_name} (CUDA {torch.version.cuda})", flush=True)
            has_gpu = True
        else:
            print("  ℹ️ No CUDA GPU detected in PyTorch (running in CPU mode).", flush=True)
    except Exception as e:
        print(f"  ℹ️ PyTorch check: {e}", flush=True)

    if has_gpu:
        need_ort_gpu = False
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" not in providers:
                need_ort_gpu = True
            else:
                print(f"  ✅ ONNX Runtime CUDA Execution Provider is active!", flush=True)
        except ImportError:
            need_ort_gpu = True

        if need_ort_gpu:
            print("  ⚡ Installing onnxruntime-gpu for 50x-100x acceleration in Colab...", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"], stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu>=1.17.0"], check=False)
            print("  ✅ onnxruntime-gpu installation finished.", flush=True)

    # Ensure backend requirements are installed
    req_file = os.path.join(BACKEND_DIR, "requirements-gpu.txt" if has_gpu else "requirements.txt")
    if os.path.isfile(req_file):
        print(f"  📦 Checking dependencies from {os.path.basename(req_file)}...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"], check=False)

    # Ensure ffmpeg is present for web-standard H.264 video streaming and +faststart
    if not shutil.which("ffmpeg") and shutil.which("apt-get"):
        print("  🎥 Installing ffmpeg for universal browser video streaming...", flush=True)
        try:
            subprocess.run(["apt-get", "update", "-qq"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def ensure_production_frontend():
    """
    Ensures the production single-bundle frontend in dist/ is available.
    Eliminates Vite dev-server waterfalls over Cloudflare Tunnel (400+ sequential HTTP requests).
    Serving dist/ directly via FastAPI gives sub-second instant page loads.
    """
    index_html = os.path.join(DIST_DIR, "index.html")
    if os.path.isfile(index_html):
        print("\n[2/4] Production frontend bundle verified in dist/ (instant sub-second loading).", flush=True)
        return

    print("\n[2/4] Building production single-bundle frontend (dist/)...", flush=True)
    has_npm = shutil.which("npm") is not None
    if has_npm and os.path.isfile(os.path.join(ROOT_DIR, "package.json")):
        try:
            if not os.path.isdir(os.path.join(ROOT_DIR, "node_modules")):
                print("  📦 Installing npm dependencies (one-time setup)...", flush=True)
                subprocess.run(["npm", "install", "--no-audit", "--no-fund", "--quiet"], cwd=ROOT_DIR, check=True)
            print("  🔨 Building optimized bundle with 'npm run build'...", flush=True)
            subprocess.run(["npm", "run", "build"], cwd=ROOT_DIR, check=True)
            print("  ✅ Frontend build complete! Bundle ready in dist/.", flush=True)
        except Exception as e:
            print(f"  ⚠️ Warning: Frontend build failed ({e}). Backend will run API-only mode.", flush=True)
    else:
        print("  ℹ️ Node.js not detected. Backend will serve API and fallback page.", flush=True)


def main():
    print("=" * 65, flush=True)
    print("🪸  Andromeida Reef Vision Studio - Colab / Cloud Runner", flush=True)
    print("=" * 65, flush=True)

    # 1. GPU Acceleration setup
    setup_colab_gpu_environment()

    # 2. Ensure optimized production frontend bundle in dist/
    ensure_production_frontend()

    # 3. Clean port 8000 and start Unified FastAPI Engine on Port 8000
    free_port(8000)
    print("\n[3/4] Starting Unified ReefVision Engine on port 8000...", flush=True)
    backend_script = os.path.join(BACKEND_DIR, "run_server.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BACKEND_DIR}:{env.get('PYTHONPATH', '')}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Prepend CUDA and cuDNN shared libraries to LD_LIBRARY_PATH so dynamic linker maps them cleanly
    cuda_paths = get_cuda_library_paths()
    if cuda_paths:
        curr_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(cuda_paths) + (f":{curr_ld}" if curr_ld else "")
        print(f"  🔗 Linked {len(cuda_paths)} CUDA/cuDNN shared library directories into runtime path.", flush=True)

    server_proc = subprocess.Popen(
        [sys.executable, backend_script, "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT_DIR,
        env=env,
    )

    # Robust health-gate: wait up to 60s for /api/health to return HTTP 200 before exposing tunnel
    print("  ⏳ Waiting for FastAPI Vision Engine to initialize and pass health check...", flush=True)
    backend_ready = False
    start_time = time.time()
    timeout = 60

    while time.time() - start_time < timeout:
        ret = server_proc.poll()
        if ret is not None:
            print(f"❌ Error: FastAPI server terminated prematurely (exit code {ret}).", flush=True)
            return

        try:
            req = urllib.request.Request("http://127.0.0.1:8000/api/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    data = resp.read().decode()
                    print(f"  ✅ Backend online and healthy: {data[:100]}...", flush=True)
                    backend_ready = True
                    break
        except Exception:
            time.sleep(1.5)

    if not backend_ready:
        print(f"❌ Error: Backend did not respond on port 8000 within {timeout}s.", flush=True)
        server_proc.terminate()
        return

    # 4. Expose public URL via Cloudflare Tunnel on unified port 8000
    cf_bin = get_cloudflared_binary()
    if cf_bin:
        print("\n[4/4] Exposing port 8000 via Cloudflare Tunnel (HTTPS / WebSockets)...", flush=True)
        cf_proc = subprocess.Popen(
            [cf_bin, "tunnel", "--url", "http://127.0.0.1:8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            tunnel_url = None
            for line in iter(cf_proc.stdout.readline, ""):
                if not line:
                    break
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if match:
                    tunnel_url = match.group(0)
                    print("\n" + "=" * 65, flush=True)
                    print(f"🚀 REEFVISION STUDIO IS READY AT:", flush=True)
                    print(f"👉 {tunnel_url}", flush=True)
                    print("\n💡 INSTANT USAGE OPTIONS:", flush=True)
                    print(f"  A) Open in Browser: {tunnel_url}", flush=True)
                    print(f"     Loads in < 1 second using the pre-compiled production bundle.", flush=True)
                    print(f"  B) In Desktop App: Click 'Remote/Colab' & paste {tunnel_url}", flush=True)
                    print("=" * 65 + "\n", flush=True)
                    break

            if not tunnel_url:
                print("⚠️ Tunnel launched but could not parse public URL. Check cloudflared output.")

            # Monitor processes
            while True:
                if server_proc.poll() is not None:
                    print("\n❌ Backend server process stopped unexpectedly.", flush=True)
                    break
                if cf_proc.poll() is not None:
                    print("\n⚠️ Cloudflare tunnel disconnected.", flush=True)
                    break
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nStopping ReefVision servers...", flush=True)
        finally:
            server_proc.terminate()
            try:
                cf_proc.terminate()
            except Exception:
                pass
    else:
        print("❌ cloudflared binary not available. Backend is running on port 8000.")
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            server_proc.terminate()


if __name__ == "__main__":
    main()
