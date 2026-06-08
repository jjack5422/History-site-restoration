# CVAT AI-Agent for craquelure + crack auto-annotation — Design

Date: 2026-06-08
Status: approved (design), pending spec review
Owner: zzz90

## 1. Goal

Use the existing fine-tuned heritage-deterioration experts as an **assisted-labeling tool inside
CVAT**, so the user reviews/edits model output instead of drawing from scratch. Scope for now:
two classes — **craquelure** and **crack**. The other three project labels
(loss / shrinkage / flaking) are out of scope and ignored by this function.

The model runs **locally** on the user's machine via a CVAT *AI agent*; only function metadata
(name + label definitions) is uploaded to CVAT. Target server is cloud **app.cvat.ai**.

## 2. Background / current assets

- Models (final experts, trained on all 32 labeled tiles):
  - craquelure: `crack_detection_sam2/runs/expert_craq_v3_final_small/last.pt`
    — SAM2 Hiera-small dense-seg head (`model_seg.SAM2SemSeg`), 2-class {background, craquelure}.
  - crack: `crack_detection_unet/runs/expert_crack_v3_final_resnet50/last.pt`
    — ResUNet (`segmentation_models_pytorch` Unet, resnet50), 2-class {background, crack}.
- Reusable inference code (no logic duplication):
  - `crack_detection_sam2/predict_full.py`: `load_model_from_ckpt(ckpt, device) -> (model, payload)`
    and `predict_full(model, img, device, tile, stride, batch_size, ...) -> (C,H,W) softmax probs`
    (sliding-window, Gaussian-blended).
  - `crack_detection_unet/src/predict_full.py`: analogous `load_model_from_ckpt` + `predict_full`.
- Both `predict_full.py` modules define the SAME symbol names and module globals
  (`predict_full`, `load_model_from_ckpt`, `NUM_CLASSES`, `CLASS_NAMES`); they must be imported as
  **separate module objects** so globals stay independent.

## 3. CVAT AI-Agents API (verified against docs.cvat.ai, 2026-06)

- A **native function** is a Python object exposing:
  - `spec` property → `cvat_sdk.auto_annotation.DetectionFunctionSpec(labels=[...])`, with
    `cvataa.label_spec("<name>", <local_id>, type="mask")` entries.
  - `detect(context, image) -> list[LabeledShapeRequest]`. `context` carries `frame_name` and
    `conf_threshold` (float | None); `image` is a `PIL.Image.Image`.
  - Mask results built with `cvataa.mask(label_id=<local_id>, points=encode_mask(bool_mask, bbox))`
    where `encode_mask` is from `cvat_sdk.masks` and `bbox = [x1, y1, x2, y2]`.
- **Label matching is by name**: the function's local ids map to the CVAT project labels with the
  same name; ids need not match. So the CVAT project must contain labels literally named
  `craquelure` and `crack`, both of **mask** type.
- A module may expose a `create(...)` factory; CLI `-p name=type:value` params are passed to it.
- Workflow: `function create-native` registers the function (returns an integer function-id);
  `function run-agent <id>` runs a local process that services requests. `create-native` is
  Cloud/Enterprise-2.25+ only (app.cvat.ai qualifies).
- Auth: `CVAT_ACCESS_TOKEN` (Personal Access Token) + `--server-host https://app.cvat.ai`.

## 4. Architecture

```
app.cvat.ai (cloud UI)
   │  annotation request (frame)                 user's machine
   │  ───────────────────────────────▶  cvat-cli function run-agent <id>
   │                                              │  (cvat_agent_env)
   │                                              │  models loaded once at create():
   │                                              │   - SAM2 craq dense-seg
   │                                              │   - ResUNet crack
   │                                              ▼
   │                                     craq_crack_func.detect(image)
   │                                       craq  prob (sliding-window 512/256)
   │                                       crack prob (sliding-window 512/256)
   │                                       priority craq_over_crack
   │                                       connected components per class
   │  ◀───────  mask shapes (craquelure, crack)   cvataa.mask(... encode_mask ...)
```

Single agent, single env, two labels. One auto-annotate action yields both classes.

## 5. Components

### 5.1 `crack_detection_sam2/cvat_agent/craq_crack_func.py`
Self-contained native-function module.

- **Imports** the two inference modules by file path via `importlib.util.spec_from_file_location`
  under distinct names (`_sam2_pf`, `_unet_pf`) to avoid global collisions.
- `create(craq_ckpt=str, crack_ckpt=str, variant=str, tile=int, stride=int,
  thresh_craq=float, thresh_crack=float, min_blob=int, priority=str, device=str)`:
  - loads both models once (`_sam2_pf.load_model_from_ckpt`, `_unet_pf.load_model_from_ckpt`),
  - returns the function object holding the models + params.
- `spec` →
  `DetectionFunctionSpec(labels=[label_spec("craquelure", 0, type="mask"),
  label_spec("crack", 1, type="mask")])`.
- `detect(context, image)`:
  1. `img = np.asarray(image.convert("RGB"))`.
  2. `pc = _sam2_pf.predict_full(craq_model, img, device, tile, stride)`; `craq = pc[1] > thr_craq`
     (thr from `context.conf_threshold` if set else `thresh_craq`).
  3. `pk = _unet_pf.predict_full(crack_model, img, device, tile, stride)`;
     `crack = pk[1] > thr_crack`.
  4. if `priority == "craq_over_crack"`: `crack &= ~craq`.
  5. for each class mask: `scipy.ndimage.label` → for each component with area ≥ `min_blob`,
     compute `bbox=[x1,y1,x2,y2]`, append `cvataa.mask(label_id, encode_mask(component, bbox))`.
  6. return the combined list (may be empty).

Default params: `tile=512, stride=256, thresh_craq=0.5, thresh_crack=0.5, min_blob=64,
priority="craq_over_crack", variant="small", device="cuda"`.

### 5.2 Environment `cvat_agent_env`
A fresh venv holding both model stacks + CVAT SDK in one interpreter:
- torch 2.11.0+cu128 / torchvision 0.26.0+cu128 (the sam2 build; loads both checkpoints),
- editable `SAM-2` (`/home/zzz90/research/segment-anything-2`),
- `segmentation_models_pytorch`, `timm`, `scipy`, `pillow`, `numpy`,
- `cvat-sdk`, `cvat-cli`.
Risk: smp + SAM-2 coexistence. Mitigation: env built and smoke-validated (§7) before any CVAT
wiring. Fallback if irreconcilable: split into two functions/agents (one per env) — not expected.

### 5.3 Registration / run scripts
`crack_detection_sam2/cvat_agent/register.sh` and `run_agent.sh` wrapping the CLI with the ckpt
`-p` params and `--server-host https://app.cvat.ai`. Token read from `CVAT_ACCESS_TOKEN` (never
committed).

## 6. Data flow / contracts

- Input per request: one `PIL.Image.Image` frame (whatever resolution the CVAT task images are;
  the existing tiles are 1024×1024 but the function is resolution-agnostic via sliding window).
- Output: list of `LabeledShapeRequest` mask shapes, `label_id ∈ {0=craquelure, 1=crack}`.
- No overlap between the two classes (priority rule). Empty result is valid (no detections).

## 7. Testing

1. **Env smoke (offline):** in `cvat_agent_env`, import both `predict_full` modules, load both
   checkpoints, run one `selected_slices` tile through each `predict_full` — assert output shape
   `(2,H,W)` and finite. Confirms smp + SAM-2 coexist and GPU fits both (8 GB).
2. **Function smoke (offline):** instantiate `create(...)`, call `detect(ctx, tile)` on a tile known
   to contain craquelure; assert ≥1 mask shape returned and that `encode_mask` output decodes back
   to a mask of the frame size.
3. **Live smoke (CVAT):** throwaway app.cvat.ai task with 1–2 images and labels
   {craquelure, crack} (mask); `create-native` then `run-agent`; trigger automatic annotation in the
   UI; confirm both-class masks appear for review.

## 8. Constraints / non-goals

- Only craquelure + crack. loss/shrinkage/flaking not produced (manual).
- GPU memory: both models resident on the 8 GB RTX 5060; if OOM, reduce inference `batch_size`
  (model load is small; risk low).
- Quality bound by the experts themselves (craq val IoU ~0.43-0.49 non-fold2; crack much weaker per
  prior results) — this is an *accelerator*, not ground truth; user reviews every frame.
- Token/credentials never committed; agent process must stay running during annotation.

## 9. Open items to confirm before implementation

- CVAT project label names are exactly `craquelure` and `crack`, both mask type.
- A Personal Access Token exists for `CVAT_ACCESS_TOKEN`.
