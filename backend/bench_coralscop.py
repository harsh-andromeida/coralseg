"""
CoralSCOP keyframe segmentation latency: where the time goes and what each speed-up costs in mask agreement.
Runs on 3 keyframes of the demo clip (854x480 proxy, 16x16 grid like the app) for several configurations.
"""
import argparse, json, os, statistics, sys, time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core.segmentation as seg  # noqa: E402
from core.device import get_cuda_provider_options  # noqa: E402
from core.fp16_encoder import build_fp16_session  # noqa: E402
from core.models import HF_SAM_DECODER_FILE, HF_SAM_ENCODER_FILE, find_or_download_onnx_model, load_coralscop_model  # noqa: E402


class Timed:
    def __init__(self, sess):
        self.s, self.t, self.c = sess, 0.0, 0

    def run(self, *a, **k):
        t = time.perf_counter()
        r = self.s.run(*a, **k)
        self.t += time.perf_counter() - t
        self.c += 1
        return r

    def get_providers(self):
        return self.s.get_providers()


def frames(video, idxs, w=854, h=480):
    cap, out = cv2.VideoCapture(video), []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        out.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA))
    return out


def agreement(base, other, shape):
    """mean over base masks of the best IoU with any mask of `other` (both upsampled to `shape`)."""
    if not base:
        return None
    up = lambda m: cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    B = [up(m["mask"]) for m in base]
    O = [up(m["mask"]) for m in other]
    best = []
    for b in B:
        ious = [np.logical_and(b, o).sum() / max(1, np.logical_or(b, o).sum()) for o in O] or [0.0]
        best.append(max(ious))
    return round(float(np.mean(best)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(os.path.dirname(__file__), "..", "demo_images", "sample_reef_transect_25fps.mp4"))
    ap.add_argument("--out-json", default="coralscop_bench.json")
    ap.add_argument("--out-video", default=None)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    model = load_coralscop_model(device=a.device)
    enc32, dec = model["encoder"], model["decoder"]
    prov = enc32.get_providers()
    enc16 = dec16 = None
    try:
        enc_path = find_or_download_onnx_model(HF_SAM_ENCODER_FILE)
        so = ort.SessionOptions() if (ort := __import__("onnxruntime")) else None
        cuda = [("CUDAExecutionProvider", get_cuda_provider_options())] if "CUDAExecutionProvider" in prov else prov
        enc16 = build_fp16_session(enc_path, cuda, so)
        dec16 = build_fp16_session(find_or_download_onnx_model(HF_SAM_DECODER_FILE), cuda, so)
    except Exception as e:  # noqa: BLE001
        print("fp16 encoder unavailable:", e)

    imgs = frames(a.video, [0, 100, 200])
    shape = imgs[0].shape[:2]
    configs = [
        ("baseline (fp32 enc+dec, 32 pts/batch, full-res masks)", dict(enc=enc32, dec=dec, bs=None, scale=1.0, pps=16)),
        ("fp16 encoder only", dict(enc=enc16, dec=dec, bs=None, scale=1.0, pps=16)),
        ("fp16 encoder + decoder", dict(enc=enc16, dec=dec16, bs=None, scale=1.0, pps=16)),
        ("fp16 enc+dec, batch 128, 0.5x masks", dict(enc=enc16, dec=dec16, bs=128, scale=0.5, pps=16)),
        ("  ...same, 12x12 grid (144 pts)", dict(enc=enc16, dec=dec16, bs=128, scale=0.5, pps=12)),
        ("  ...same, 8x8 grid (64 pts)", dict(enc=enc16, dec=dec16, bs=128, scale=0.5, pps=8)),
        ("fp32 enc+dec, 12x12 grid (144 pts)", dict(enc=enc32, dec=dec, bs=None, scale=1.0, pps=12)),
        ("fp32 enc+dec, 8x8 grid (64 pts)", dict(enc=enc32, dec=dec, bs=None, scale=1.0, pps=8)),
    ]
    base_masks = None
    results = []
    for name, c in configs:
        if c["enc"] is None or c["dec"] is None:
            continue
        te, td = Timed(c["enc"]), Timed(c["dec"])
        bundle = {"encoder": te, "decoder": td}
        kw = dict(points_per_side=c["pps"], pred_iou_thresh=0.5, stability_score_thresh=0.5, min_mask_region_area=25,
                  decode_scale=c["scale"], decoder_batch_size=c["bs"])
        for im in imgs[:1]:
            seg.run_segmentation(bundle, im, **kw)  # warm-up
        tot, enc_t, dec_t, masks, agree = [], [], [], [], []
        for k, im in enumerate(imgs):
            for rep in range(3):
                te.t = td.t = 0.0
                t0 = time.perf_counter()
                m, _ = seg.run_segmentation(bundle, im, **kw)
                tot.append(time.perf_counter() - t0); enc_t.append(te.t); dec_t.append(td.t)
            masks.append(len(m))
            if base_masks is None or name.startswith("baseline"):
                if name.startswith("baseline"):
                    base_masks = (base_masks or []) + [m]
            agree.append(agreement(base_masks[k], m, shape) if base_masks and len(base_masks) > k else None)
        med = lambda x: round(1000 * statistics.median(x), 1)
        results.append({"config": name, "total_ms": med(tot), "encoder_ms": med(enc_t), "decoder_ms": med(dec_t),
                        "other_ms": round(med(tot) - med(enc_t) - med(dec_t), 1), "masks_per_keyframe": masks,
                        "mask_agreement_vs_baseline": agree})
        print(results[-1], flush=True)
    json.dump({"providers": prov, "results": results}, open(a.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
