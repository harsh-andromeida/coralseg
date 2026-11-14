"""
Run flow_prototype.py (dense optical-flow mask propagation) on a Modal GPU and fetch the masked video.

The GPU is chosen with BENCH_GPU at launch (decoration time), one `modal run` per GPU:

    BENCH_GPU=L4 modal run backend/modal_flow.py
    for g in T4 L4 A10G L40S A100-40GB H100; do BENCH_GPU=$g modal run backend/modal_flow.py & done; wait

Results are written to docs/benchmarks/raw/flow_<gpu>.json and flow_<gpu>_masked.mp4.
"""

import json
import os
import subprocess
from pathlib import Path

import modal

GPU = os.environ.get("BENCH_GPU", "T4")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VIDEO = ROOT / "demo_images" / "sample_reef_transect_25fps.mp4"

app = modal.App(f"reefvision-flow-{GPU.lower()}")
hf_cache = modal.Volume.from_name("reefvision-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(HERE / "requirements-gpu.txt"))
    # onnxruntime-gpu's CUDA provider needs CUDA runtime libs (curand, cufft, ...) that torch's wheel does not pull in
    .pip_install("onnxruntime-gpu[cuda,cudnn]")
    .env({"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        str(HERE), "/root/backend", ignore=["uploads", "exports", "__pycache__", "*.pyc"]
    )
    .add_local_file(str(VIDEO), "/root/demo_images/sample_reef_transect_25fps.mp4")
)


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache}, timeout=3000)
def flow(extra: str = "") -> dict:
    def sh(cmd: str) -> str:
        return subprocess.run(cmd, shell=True, text=True, capture_output=True).stdout.strip()

    env = {
        "nvidia_smi": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
        "vcpus": os.cpu_count(),
    }
    cmd = (
        "cd /root/backend && python flow_prototype.py "
        "--video /root/demo_images/sample_reef_transect_25fps.mp4 "
        f"--out-video /tmp/flow_masked.mp4 --out-json /tmp/flow.json {extra} > /tmp/run.log 2>&1"
    )
    rc = subprocess.run(cmd, shell=True).returncode
    log_tail = sh("tail -c 3000 /tmp/run.log")
    hf_cache.commit()
    if rc != 0 or not os.path.exists("/tmp/flow.json"):
        return {"ok": False, "env": env, "log_tail": log_tail}
    return {
        "ok": True,
        "env": env,
        "report": json.load(open("/tmp/flow.json")),
        "video": open("/tmp/flow_masked.mp4", "rb").read(),
    }


@app.local_entrypoint()
def main(extra: str = ""):
    out_dir = ROOT / "docs" / "benchmarks" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = flow.remote(extra)
    tag = f"flow_{GPU.lower()}"
    if not res["ok"]:
        print(f"[{GPU}] FAILED\n{res['log_tail']}")
        return
    (out_dir / f"{tag}_masked.mp4").write_bytes(res.pop("video"))
    (out_dir / f"{tag}.json").write_text(json.dumps(res, indent=2, default=str))
    r = res["report"]["final_pass"]
    print(f"[{GPU}] wall {r['wall_sec']}s | e2e {r['e2e_fps']} FPS | providers {res['report']['ort_providers']}")
