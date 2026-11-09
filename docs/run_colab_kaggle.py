#!/usr/bin/env python3
"""
ReefVision Studio - Google Colab & Kaggle GPU Cloud Runner.
Configures NVIDIA CUDA hardware acceleration for ONNX Runtime and PyTorch,
launches the FastAPI vision engine, and exposes it securely via Cloudflare Tunnel.
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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")


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
            print("  ⚡ Upgrading to onnxruntime-gpu for 50x-100x acceleration in Colab...", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"], stdout=subprocess.DEVNULL)
            subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu>=1.17.0"], check=False)
            print("  ✅ onnxruntime-gpu installation finished.", flush=True)

    # Ensure backend requirements are installed
    req_file = os.path.join(BACKEND_DIR, "requirements-gpu.txt" if has_gpu else "requirements.txt")
    if os.path.isfile(req_file):
        print(f"  📦 Checking dependencies from {os.path.basename(req_file)}...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file, "--quiet"], check=False)


def main():
    print("=" * 65, flush=True)
    print("🪸  Andromeida Reef Vision Studio - Colab / Cloud Runner", flush=True)
    print("=" * 65, flush=True)

    # 1. GPU Acceleration setup
    setup_colab_gpu_environment()

    # 2. Start FastAPI Vision Engine on Port 8000
    print("\n[2/4] Starting ReefVision FastAPI Engine on port 8000...", flush=True)
    backend_script = os.path.join(BACKEND_DIR, "run_server.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{BACKEND_DIR}:{env.get('PYTHONPATH', '')}"

    server_proc = subprocess.Popen(
        [sys.executable, backend_script, "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT_DIR,
        env=env,
    )
    time.sleep(3)

    if server_proc.poll() is not None:
        print("❌ Error: FastAPI server failed to start.", flush=True)
        return

    # Check health
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=3) as resp:
            data = resp.read().decode()
            print(f"  ✅ Backend online: {data[:120]}...", flush=True)
    except Exception as err:
        print(f"  ⚠️ Warning: Health check pending ({err})", flush=True)

    # 3. Check if frontend Vite dev server can be started
    has_node = shutil.which("npm") is not None
    frontend_proc = None
    target_tunnel_port = 8000

    if has_node and os.path.isfile(os.path.join(ROOT_DIR, "package.json")):
        print("\n[3/4] Node.js detected. Launching Vite frontend on port 1425...", flush=True)
        frontend_proc = subprocess.Popen(
            ["npm", "run", "dev:frontend"],
            cwd=ROOT_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        target_tunnel_port = 1425
    else:
        print("\n[3/4] Exposing FastAPI backend directly on port 8000...", flush=True)

    # 4. Expose public URL via Cloudflare Tunnel
    cf_bin = get_cloudflared_binary()
    if cf_bin:
        print(f"\n[4/4] Exposing port {target_tunnel_port} via Cloudflare Tunnel (HTTPS / WebSockets)...", flush=True)
        cf_proc = subprocess.Popen(
            [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{target_tunnel_port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        url_found = False
        try:
            for line in iter(cf_proc.stdout.readline, ""):
                if not line:
                    break
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if match:
                    url = match.group(0)
                    print("\n" + "=" * 65, flush=True)
                    if target_tunnel_port == 1425:
                        print(f"🚀 REEFVISION STUDIO IS READY IN YOUR BROWSER AT:", flush=True)
                        print(f"👉 {url}", flush=True)
                    else:
                        print(f"🚀 YOUR COLAB GPU BACKEND IS READY AT:", flush=True)
                        print(f"👉 {url}", flush=True)
                        print("\n💡 IN REEFVISION DESKTOP APP:", flush=True)
                        print(f"1. Click 'Remote/Colab' on the top status card.", flush=True)
                        print(f"2. Paste: {url}", flush=True)
                        print(f"3. Click 'Save & Connect' for instant GPU acceleration!", flush=True)
                    print("=" * 65 + "\n", flush=True)
                    url_found = True
                    break

            # Keep process alive
            cf_proc.wait()
        except KeyboardInterrupt:
            print("\nStopping ReefVision servers...", flush=True)
        finally:
            server_proc.terminate()
            if frontend_proc:
                try:
                    frontend_proc.terminate()
                except Exception:
                    pass
            try:
                cf_proc.terminate()
            except Exception:
                pass
    else:
        print(f"❌ cloudflared binary not available. Backend is running on port {target_tunnel_port}.")
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            server_proc.terminate()


if __name__ == "__main__":
    main()
