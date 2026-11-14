"""
Run bench_app_flow.py on a Modal GPU.

The GPU is chosen with BENCH_GPU at launch (decoration time), one `modal run` per GPU:

    BENCH_GPU=L4 modal run backend/modal_bench.py
    for g in T4 L4 A10G L40S A100-40GB H100; do BENCH_GPU=$g modal run backend/modal_bench.py & done; wait

Results are written to docs/benchmarks/raw/modal_<gpu>.{json,md}.
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

app = modal.App(f"reefvision-bench-{GPU.lower()}")
hf_cache = modal.Volume.from_name("reefvision-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements(str(HERE / "requirements-gpu.txt"))
    .env({"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        str(HERE), "/root/backend", ignore=["uploads", "exports", "__pycache__", "*.pyc"]
    )
    .add_local_file(str(VIDEO), "/root/demo_images/sample_reef_transect_25fps.mp4")
)


@app.function(image=image, gpu=GPU, volumes={"/hf": hf_cache}, timeout=3000)
def bench(runs: int = 3, extra: str = "") -> dict:
    def sh(cmd: str) -> str:
        return subprocess.run(cmd, shell=True, text=True, capture_output=True).stdout.strip()

    env = {
        "gpu_requested": GPU,
        "nvidia_smi": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"),
        "cpu": sh("lscpu | grep -E 'Model name' | head -1"),
        "vcpus": os.cpu_count(),
    }
    cmd = (
        "cd /root/backend && python bench_app_flow.py "
        "--video /root/demo_images/sample_reef_transect_25fps.mp4 "
        f"--runs {runs} --out-json /tmp/r.json --out-md /tmp/r.md {extra} > /tmp/run.log 2>&1"
    )
    rc = subprocess.run(cmd, shell=True).returncode
    log_tail = sh("tail -c 3000 /tmp/run.log")
    hf_cache.commit()
    if rc != 0 or not os.path.exists("/tmp/r.json"):
        return {"ok": False, "env": env, "log_tail": log_tail}
    return {
        "ok": True,
        "env": env,
        "report": json.load(open("/tmp/r.json")),
        "markdown": open("/tmp/r.md").read(),
    }


@app.local_entrypoint()
def main(runs: int = 3, extra: str = ""):
    out_dir = ROOT / "docs" / "benchmarks" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = bench.remote(runs, extra)
    tag = f"modal_{GPU.lower()}"
    (out_dir / f"{tag}.json").write_text(json.dumps(res, indent=2, default=str))
    if res["ok"]:
        (out_dir / f"{tag}.md").write_text(res["markdown"])
        w = res["report"]["warm_median"]
        print(
            f"[{GPU}] e2e {w['e2e_fps']} FPS | tracker {w['tracker_fps']} FPS | wall {w['wall_sec']}s "
            f"| providers {res['report']['ort_providers']}"
        )
    else:
        print(f"[{GPU}] FAILED\n{res['log_tail']}")
