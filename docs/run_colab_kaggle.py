"""
Helper script to run the Streamlit CoralSCOP dashboard on Google Colab or Kaggle.
Exposes the local Streamlit port via Cloudflare Tunnel (reliable WebSocket support, no passwords)
with fallback to Localtunnel.
"""

import subprocess
import time
import os
import sys
import re
import urllib.request
import platform
import stat

def get_cloudflared_binary():
    """Download cloudflared binary if not present."""
    local_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared")
    if os.path.exists(local_bin):
        return local_bin

    # Check system PATH
    import shutil
    sys_cf = shutil.which("cloudflared")
    if sys_cf:
        return sys_cf

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
            print(f"📦 Downloading Cloudflare Tunnel binary...", flush=True)
            urllib.request.urlretrieve(url, local_bin)
            st = os.stat(local_bin)
            os.chmod(local_bin, st.st_mode | stat.S_IEXEC)
            return local_bin
        except Exception as e:
            print(f"⚠️ Failed to download cloudflared: {e}", flush=True)

    return None

def main():
    print("=" * 60, flush=True)
    print("🪸 CoralSCOP Streamlit Cloud / Colab / Kaggle Runner", flush=True)
    print("=" * 60, flush=True)

    # 1. Start Streamlit in background
    print("\n[1/2] Starting Streamlit server...", flush=True)
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    st_proc = subprocess.Popen(streamlit_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

    if st_proc.poll() is not None:
        print("❌ Error: Streamlit server failed to start.", flush=True)
        return

    # 2. Try Cloudflare Tunnel first (supports Streamlit WebSockets flawlessly, zero auth/passwords)
    cf_bin = get_cloudflared_binary()
    if cf_bin:
        print("\n[2/2] Exposing public URL via Cloudflare Tunnel (WebSockets enabled)...", flush=True)
        cf_proc = subprocess.Popen(
            [cf_bin, "tunnel", "--url", "http://127.0.0.1:8501"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        url_found = False
        try:
            for line in iter(cf_proc.stdout.readline, ''):
                if not line:
                    break
                match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                if match:
                    url = match.group(0)
                    print("\n" + "=" * 60, flush=True)
                    print(f"🚀 YOUR DASHBOARD IS READY AT:", flush=True)
                    print(f"👉 {url}", flush=True)
                    print("=" * 60 + "\n", flush=True)
                    url_found = True
                    break
            
            # Keep process alive
            cf_proc.wait()
        except KeyboardInterrupt:
            print("\nStopping servers...", flush=True)
        finally:
            st_proc.terminate()
            try:
                cf_proc.terminate()
            except Exception:
                pass
        return

    # Fallback to localtunnel
    print("\n[2/2] Exposing public URL via localtunnel...", flush=True)
    try:
        ipv4 = urllib.request.urlopen('https://ipv4.icanhazip.com', timeout=5).read().decode('utf8').strip()
        print(f"🔑 Localtunnel Password (your public IP): {ipv4}", flush=True)
        print("Click the URL below and paste this password/IP if prompted:\n", flush=True)
    except Exception:
        pass

    lt_proc = subprocess.Popen(
        ["npx", "-y", "localtunnel", "--port", "8501"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        for line in iter(lt_proc.stdout.readline, ''):
            if line:
                print(line.strip(), flush=True)
        lt_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...", flush=True)
    finally:
        st_proc.terminate()
        try:
            lt_proc.terminate()
        except Exception:
            pass

if __name__ == "__main__":
    main()
