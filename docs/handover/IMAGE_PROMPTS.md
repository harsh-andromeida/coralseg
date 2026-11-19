# Handover doc: image prompts (Coral Atlas theme)

Generate each image below and send it back with its file name. Everything else (logos, UI screenshots, benchmark charts) is fetched or captured by Claude.

## Shared style block (paste at the start of every prompt)

> Clean, modern technical flowchart for a professional software handover document. Flat vector style, pure white background (#FFFFFF), generous whitespace, no gradients, no 3D, no drop shadows heavier than a soft 4px blur, no clip-art, no emoji. Font: Plus Jakarta Sans (or a similar geometric sans), labels in dark ink #1B1F2A, bold for node titles, regular for small sub-labels. Nodes are rounded rectangles (16px radius) with a light tinted fill and a 2px border in the same hue, darker shade. Arrows are 2px solid #475166 lines with small filled arrowheads, orthogonal (right-angle) routing, never crossing. Decisions are rounded diamonds. Async/background work uses dashed borders and dashed arrows. Data stores are cylinders. Landscape 16:9, 2400×1350 px, all text horizontal and legible at 50% zoom. Spelling must be exact.

### Brand

The project is owned by **Andromeida**. The real wordmark (`assets/brand/andromeida_logo.svg`, navy #0E2656 → teal #129D89 gradient) is placed on the cover and footers by Claude, so **never ask the image model to draw the logo or the word "Andromeida"**. Where a prompt says "leave logo space", keep that area empty white.

### Palette (one hue per chapter)

| Chapter | Fill | Border / text accent |
|---|---|---|
| 01 Introduction / architecture | #FFF4E6 | #F76707 |
| 02 Core models | #FFE8CC | #E8590C |
| 03 Image flow | #E6FCF5 | #0CA678 |
| 04 Video flow | #E7F5FF | #1C7ED6 |
| 05 Tech stack | #F3F0FF | #7048E8 |
| 06 File structure | #FFF9DB | #F59F00 |
| Warnings / slow paths | #FFF5F5 | #E03131 |
| Neutral (user, files) | #F1F3F5 | #868E96 |

---

## 1. `fig-01-architecture.png` — System architecture (chapter 01)

> [style block] Title at top-left: "Reef Vision Studio — system architecture". Three stacked horizontal layers, left to right flow:
> Layer 1 (orange #F76707): "Tauri v2 desktop shell (Rust)" with sub-labels "spawns backend from .venv", "health-checks 127.0.0.1:8000", "native save dialog".
> Layer 2 (violet #7048E8): "React 19 + Vite UI" with three child nodes "Upload view", "Image analysis view", "Video analysis view"; small note "services/api.ts — every backend call".
> Layer 3 (green #0CA678): "FastAPI engine (local, port 8000)" with routers "system", "images", "batch", "video".
> Under the engine, an orange (#E8590C) box "ONNX Runtime" containing three model chips: "CoralSCOP (SAM encoder + decoder)", "BioCLIP visual encoder", "YOLO11n bleaching detector".
> On the right, a grey cloud node "Hugging Face Hub — harsh-awasthi/Andromeida-ReefVision-ONNX" with a dashed arrow labelled "first launch: download" into ONNX Runtime.
> At the bottom a cylinder "SQLite (video jobs, WAL)" and a folder icon "Exports: COCO, CSV, MP4".
> Arrows: UI ⇄ FastAPI labelled "HTTP / JSON"; Tauri → FastAPI labelled "starts + monitors".

## 2. `fig-02-first-launch.png` — First-launch model download (chapter 02)

> [style block] Vertical flowchart, orange accent #E8590C. Nodes: "App starts" → "Tauri launches FastAPI" → diamond "Models in local cache?" → No → "Download from Hugging Face (harsh-awasthi/Andromeida-ReefVision-ONNX)" with a list of 5 file chips: "sam_image_encoder.onnx", "sam_mask_decoder.onnx", "bioclip_visual.onnx", "coral_taxonomy_embeddings.npy", "bleaching_yolo11n.onnx" → "Progress shown on home screen" → "Models ready". Yes branch goes straight to "Models ready". Final node "Pick inference device" splits into three chips: "NVIDIA GPU (CUDA)", "Apple Silicon (CoreML)", "CPU fallback".

## 3. `fig-03-device-tiers.png` — Hardware tier resolution (chapter 02)

> [style block] Decision tree, orange accent. Start "Resolve device (core/device.py)". Diamond "CUDA available + onnxruntime-gpu loads?" → Yes → green node "Tier: CUDA — fastest, fp16 SAM". No → diamond "macOS on Apple Silicon?" → Yes → "Tier: CoreML — Neural Engine / GPU". No → "Tier: CPU — threaded, slower but works everywhere". Side note box in red tint: "If CUDA libraries are missing, ORT can silently fall back to CPU — the app verifies the active provider."

## 4. `fig-04-image-pipeline.png` — Image pipeline, single + bulk (chapter 03)

> [style block] Green accent #0CA678. Left: two entry nodes "Single image" and "Folder (batch queue)" merging into "Register image". Then a horizontal main row:
> "Segment — CoralSCOP automatic masks" (sub: "SAM encoder → grid prompts → decoder → NMS") →
> "Enrich" which forks into two parallel nodes: "BioCLIP — genus + growth form (zero-shot vs taxonomy embeddings)" and "YOLO11n — healthy vs bleached" →
> merge → "Overlay render (distinct colours)" →
> "Review & edit in canvas editor" →
> "Export".
> Under the "Review & edit" node a dashed loop: "Point / box prompt → SAM decoder → mask → polygon" back into the editor.
> Under "Folder" a small note: "BatchProcessor thread: pause · resume · cancel".

## 5. `fig-05-interactive-sam.png` — Interactive refinement (chapter 03)

> [style block] Green accent. Horizontal: "User clicks point or draws box" → "Cached image embedding (SAM_SESSION)" (cylinder, dashed note "encoder runs once per image") → "SAM mask decoder" → "Binary mask" → "Moore-neighbour contour trace" → "RDP polygon simplification" → "Editable polygon in canvas". A return arrow from the last node to the first labelled "add / remove points".

## 6. `fig-06-batch-lifecycle.png` — Batch job states (chapter 03)

> [style block] State diagram, green accent. States as rounded pills: "Queued" → "Running" ⇄ "Paused"; "Running" → "Completed"; "Running" → "Cancelled"; "Paused" → "Cancelled". Inside "Running": small sequence "segment → enrich → overlay" per image with a counter "image 12 / 40". Arrow from "Completed" to a folder node "Folder export (COCO dataset zip)".

## 7. `fig-07-exports-map.png` — Export options (chapters 03 and 04)

> [style block] Mind-map style, centre node "Exports". Left branch, green: "Images" → "COCO JSON (polygons + categories)", "COCO dataset zip (images + annotations + summary)", "CSV (per-colony rows: genus, growth form, bleaching, area)", "Overlay PNG (rendered masks)", "Raw model telemetry (JSON)". Right branch, blue: "Video" → "Annotated MP4", "Transect CSV (colonies along the dive)", "COCO video JSON (per-frame tracks)". Each leaf has a tiny grey caption on who uses it: "model training", "ecologists / spreadsheets", "review & reporting".

## 8. `fig-08-legacy-video.png` — Legacy video pipeline, restyled (chapter 04)

> [style block] Redraw of the original diagram, blue accent #1C7ED6 with the slow path in red tint. Top row: stick-figure "Actor" (grey) → "Video" → "25 FPS video feed" → "Frame X". "Frame X" → diamond "Overlap > threshold?". Formula in a neutral box top-right: "Overlap(t) = Area(Fₜ ∩ F_key) / Area(Fₜ)". Yes → parallelogram "SAM 2 track" → "Tracked segmentation". No → red-tinted parallelogram "Segmentation (1–2 s) + Bleach detection (0.5 s) + Taxonomy (10–25 s)" → "Detected segmentation", which loops back up to "Frame Y" and into the diamond. Both outputs merge into "Segmented video feed". Small red badge on "SAM 2 track": "~20 ms per colony per frame".

## 9. `fig-09-flow-pipeline.png` — New optical-flow pipeline (chapter 04)

> [style block] Blue accent. Main lane (solid, left to right): "Video 1080p" → "Decode (prefetch thread)" → "480p proxy" → "DIS optical flow (half-res, ~1 ms/frame)" → "Warp ONE label image (all colonies at once)" → "Polygons + bboxes (worker thread, 1 frame behind)" → cylinder "SQLite WAL (batched)" → green node "Job completed — UI usable".
> Upper lane (dashed, async): from "480p proxy" → "Ego-motion keyframe selector" → "CoralSCOP fp16 (background thread, guided prompts)" → dashed arrow into the warp node labelled "carry via recorded flows, one-to-one IoU match".
> Lower lane (dashed, async): from warp node labelled "new colony" → "BioCLIP genus + YOLO bleaching (once per colony)" → into SQLite.
> After "Job completed": dashed arrow to "Background export thread: MP4 + CSV + COCO (pending → ready)".
> Lane labels on the far left: "Real-time path", "Keyframes (async)", "Classification (async)".

## 10. `fig-10-why-flow.png` — Concept: track each colony vs warp one image (chapter 04)

> [style block] Side-by-side comparison panel. Left half, red tint, title "Before: SAM 2 tracks every colony": an underwater frame sketch with 6 coral outlines, each with its own small arrow and a stopwatch icon, caption "cost grows with colony count (27 colonies ≈ 350 ms/frame)". Right half, blue tint, title "After: warp one label image": same frame, one large flow-field arrow grid over the whole image, all outlines moved together, caption "cost is constant (~1 ms/frame), corals are static, the camera moves".

## 11. `fig-11-video-dataflow.png` — Video request/response flow (chapter 04)

> [style block] Sequence diagram, blue accent, four lifelines: "UI (VideoAnalysisView)", "FastAPI /api/video", "VideoProcessor thread", "SQLite store". Messages: "POST /upload" → "job id"; "POST /process"; VideoProcessor "per frame: FlowTracker" → "write frames"; UI loop box "poll /status + frame deltas every ~1 s"; "GET exports when export_status = ready".

## 12. `cover-hero.png` — Cover illustration (title page)

> Flat vector illustration, white background, wide 21:9 (2520×1080). A stylised coral reef seen from a diver's camera, made of simple geometric shapes in the six chapter colours (#F76707, #E8590C, #0CA678, #1C7ED6, #7048E8, #F59F00) on soft tints. Several coral colonies have thin outline polygons and small pill-shaped labels like a segmentation overlay ("Acropora · branching", "Porites · massive", "bleached"). A faint dotted grid suggests a camera frame. Calm, scientific, optimistic, no fish cartoons, no text other than the labels, no gradients except one very soft navy-to-teal (#0E2656 → #129D89, at 10–15% strength) water tint behind the reef that echoes the Andromeida brand. Leave logo space: an empty white band across the top 18% of the image.

---

# Part 2: Illustrations (to make the document more engaging)

## Illustration style block (paste at the start of every prompt in this part)

> Editorial flat-vector illustration for a professional technical handbook. White or very light tinted background, clean geometric shapes, subtle grain texture at 5%, soft long shadows at most, no gradients except where stated, no photorealism, no 3D render look, no emoji, no clip-art. Underwater subjects are drawn accurately (real coral shapes, not cartoon). Palette limited to the chapter hue named in the prompt plus ink #1B1F2A, warm grey #868E96 and white. No text, letters or numbers anywhere in the image unless the prompt quotes them exactly. Never draw any company logo or the word "Andromeida".

## 13–18. Chapter opener banners (one per chapter, 2400×800, 3:1)

Each sits at the top of its chapter; keep the left 40% calm (title text is overlaid there by Claude).

- **`ch01-intro.png`** — [illustration style] Hue #F76707 (orange) on #FFF4E6. A small autonomous underwater vehicle with a camera glides over a coral reef, a cone of light from its camera shows a faint dotted frame on the reef below. Subject placed in the right 60%.
- **`ch02-models.png`** — [illustration style] Hue #E8590C on #FFE8CC. Three abstract "lenses" floating above a single coral colony: one lens shows the colony as clean outline polygons, one shows a small branching tree-of-life diagram, one shows the colony half healthy-brown and half bleached-white. Right 60%.
- **`ch03-image-flow.png`** — [illustration style] Hue #0CA678 on #E6FCF5. A fanned stack of underwater photo prints sliding into a frame where corals get coloured outline masks; a few small point-markers and a dashed selection box on one colony. Right 60%.
- **`ch04-video-flow.png`** — [illustration style] Hue #1C7ED6 on #E7F5FF. A film strip curving across the reef; the coral outlines on successive frames move together, connected by faint flow arrows (a regular arrow field), suggesting motion tracking. Right 60%.
- **`ch05-tech-stack.png`** — [illustration style] Hue #7048E8 on #F3F0FF. Isometric stack of four translucent slabs floating above water: top slab a window frame (desktop app), second a component grid (UI), third a server rack glyph (API), bottom a neural-network node pattern (models). No logos. Right 60%.
- **`ch06-file-structure.png`** — [illustration style] Hue #F59F00 on #FFF9DB. A tidy isometric filing cabinet / folder tree whose folders open like drawers, with tiny coral icons inside some drawers. Right 60%.

## 19. `problem-manual-annotation.png` — Why the project exists (chapter 01, 1600×1000)

> [illustration style] Split composition, hue #E03131 left and #0CA678 right. Left: a tired researcher at a desk tracing coral outlines by hand on a monitor, a tall stack of photo folders and a clock showing many hours. Right: the same researcher relaxed, the monitor showing all corals outlined and labelled automatically, a small progress bar complete. A thin vertical divider between the halves.

## 20. `field-to-dataset.png` — The real-world journey (chapter 01, 2400×900)

> [illustration style] Horizontal journey, hues orange #F76707 → green #0CA678 → blue #1C7ED6. Four scenes connected by a dotted path: (1) diver or AUV filming a reef, (2) a laptop on a boat deck receiving footage, (3) the laptop screen showing segmented corals, (4) a neat stack of dataset cards and a spreadsheet glyph, with a small bar chart, representing reports for reef scientists.

## 21–23. Model "portrait" cards (chapter 02, each 1200×900)

- **`model-coralscop.png`** — [illustration style] Hue #E8590C. One coral colony photographed from above, overlaid with a sparse grid of small dots (prompt points) and crisp outline masks around each colony piece, a few dots highlighted where masks were born.
- **`model-bioclip.png`** — [illustration style] Hue #7048E8. A coral colony on the left connected by a dotted line to a branching tree-of-life diagram on the right, one branch highlighted, with small leaf nodes shaped like coral silhouettes.
- **`model-bleaching.png`** — [illustration style] Hue #F59F00. Two colonies of the same species side by side: left healthy (warm brown/olive tones), right bleached (white/pale), each inside a rounded detection box; the boxes use orange-amber for bleached and green #0CA678 for healthy.

## 24. `growth-forms-set.png` — Growth-form icon set (chapter 02, 2400×600)

> [illustration style] Six evenly spaced line-and-fill icons in one row, ink #1B1F2A outlines with #FFE8CC fills, consistent 3px stroke, each centred in a soft rounded square tile: branching coral, massive (boulder) coral, encrusting coral, foliose (leafy) coral, plate (table) coral, solitary (mushroom) coral. Nothing below the tiles (labels are added by Claude).

## 25. `bleaching-explainer.png` — What bleaching is (chapter 02, 1600×900)

> [illustration style] Hue #F59F00 with #0CA678. Three-stage left-to-right sequence of the same branching coral: healthy (coloured, tiny symbiotic algae dots inside), stressed (paler, thermometer icon above showing warm water), bleached (white skeleton showing, algae dots gone). Thin arrows between the stages.

## 26. `onnx-conversion.png` — PyTorch to ONNX concept (chapter 02 / 03, 1600×800)

> [illustration style] Hue #E8590C. Left: a bulky, tangled machine with many cables (the training-time model). Centre: a funnel or press. Right: a compact, sleek cartridge that slots into three sockets labelled with simple icons only: a GPU chip, an Apple-style chip silhouette without logo, and a generic CPU chip. Conveys "same model, lighter and portable".

## 27. `hardware-tiers.png` — Inference device choice (home screen section, 1600×700)

> [illustration style] Hue #E8590C. Three podium blocks of different heights: tallest with a GPU card, middle with a laptop showing a neural-engine chip glyph, shortest with a plain CPU chip. Speed lines behind the tallest; the CPU block sits on a wide, stable base to suggest "slower but works everywhere". No animals.

## 28. `laptop-frame.png` — Device mockup frame (cover and home screen, 2400×1500)

> Clean flat-vector modern laptop, front three-quarter view, on pure white, the screen area a solid flat #00FF00 rectangle (it will be replaced with a real screenshot), thin bezel, soft contact shadow only. No logo on the lid or bezel.

## 29. `old-vs-new-race.png` — Video speed metaphor (chapter 04, 1600×800)

> [illustration style] Hues #E03131 and #1C7ED6. Two lanes: top lane a small submarine dragging many tethered balloons (each balloon a coral outline), moving slowly; bottom lane a sleek submarine carrying a single transparent sheet with all coral outlines printed on it, moving fast with speed lines. No numbers.

## 30. `handover-team.png` — Closing page (2400×1000)

> [illustration style] Hue #0CA678 with #1C7ED6 accents. Two engineers on a research boat deck, one handing a glowing laptop to the other, reef visible through clear water below, calm sunrise sky in pale tints. Leave the left 40% empty for closing text.

---

## Not needed from you

- **Tech stack logos**: fetched by Claude into `docs/handover/assets/stack/`.
- **UI screenshots**: captured by Claude from the app running on Colab into `docs/handover/assets/screens/`.
- **Benchmark charts** (old vs new, per-GPU): drawn by Claude from `docs/VIDEO_PIPELINE_SUMMARY.md` numbers.
