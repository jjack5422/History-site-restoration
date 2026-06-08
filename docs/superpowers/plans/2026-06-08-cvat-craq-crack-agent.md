# CVAT craq+crack AI-Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a local CVAT AI agent that auto-annotates `craquelure` and `crack` mask shapes on app.cvat.ai frames using the existing fine-tuned SAM2 dense-seg and ResUNet experts.

**Architecture:** A single native function (`craq_crack_func.py`) loads both checkpoints once, runs sliding-window inference per frame, applies craq-over-crack priority, and returns per-connected-component CVAT mask shapes. It runs in a dedicated venv (`cvat_agent_env`) holding both model stacks plus the CVAT SDK. Register once with `cvat-cli function create-native`, then keep `cvat-cli function run-agent` running while annotating.

**Tech Stack:** Python, PyTorch 2.11+cu128, SAM-2 (editable), segmentation_models_pytorch, scipy, cvat-sdk, cvat-cli.

Spec: `docs/superpowers/specs/2026-06-08-cvat-craq-crack-agent-design.md`

---

## Setup (before Task 1)

- [ ] Create a feature branch off `master`:

```bash
cd /home/zzz90/research
git checkout -b feature/cvat-craq-crack-agent
```

- [ ] Create the package dir and a `.gitignore` guarding secrets:

```bash
mkdir -p /home/zzz90/research/crack_detection_sam2/cvat_agent
printf '*.env\n.token\n__pycache__/\n' > /home/zzz90/research/crack_detection_sam2/cvat_agent/.gitignore
```

---

## Task 1: Build `cvat_agent_env` and prove both models load + run

**Files:**
- Create: `crack_detection_sam2/cvat_agent/requirements.txt`
- Create: `crack_detection_sam2/cvat_agent/_env_smoke.py`

- [ ] **Step 1: Write the env requirements file**

Create `crack_detection_sam2/cvat_agent/requirements.txt`:

```
# Torch matched to the working sam2 build (cu128). Install torch separately (see Step 2).
segmentation-models-pytorch
timm
scipy
pillow
numpy
cvat-sdk
cvat-cli
```

- [ ] **Step 2: Create the venv and install deps**

Run:

```bash
cd /home/zzz90/research
python3 -m venv cvat_agent_env
cvat_agent_env/bin/pip install --upgrade pip
# torch/vision: same versions as sam2_env (cu128)
cvat_agent_env/bin/pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
cvat_agent_env/bin/pip install -r crack_detection_sam2/cvat_agent/requirements.txt
# editable installs so model + shared code import cleanly
cvat_agent_env/bin/pip install -e /home/zzz90/research/segment-anything-2
cvat_agent_env/bin/pip install -e /home/zzz90/research/_lib/crackseg_common
```

Expected: installs complete without an unresolved dependency-conflict error. (pip may print
non-fatal "dependency resolver" warnings; those are OK. A hard `ERROR: Cannot install ...
because these package versions have conflicting dependencies` is a real failure — see Step 5.)

- [ ] **Step 3: Write the env smoke script**

Create `crack_detection_sam2/cvat_agent/_env_smoke.py`:

```python
"""Prove cvat_agent_env can load both experts and run one tile each. No CVAT involved."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SAM2_ROOT = Path("/home/zzz90/research/crack_detection_sam2")
UNET_SRC = Path("/home/zzz90/research/crack_detection_unet/src")
CRAQ_CKPT = SAM2_ROOT / "runs/expert_craq_v3_final_small/last.pt"
CRACK_CKPT = Path("/home/zzz90/research/crack_detection_unet/runs/expert_crack_v3_final_resnet50/last.pt")
TILE_DIR = Path("/home/zzz90/research/_data/selected_slices/batch_1")

for p in (str(SAM2_ROOT), str(UNET_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    sam2_pf = load_module("_sam2_pf", SAM2_ROOT / "predict_full.py")
    unet_pf = load_module("_unet_pf", UNET_SRC / "predict_full.py")

    craq_model, _ = sam2_pf.load_model_from_ckpt(str(CRAQ_CKPT), "cuda")
    crack_model, _ = unet_pf.load_model_from_ckpt(str(CRACK_CKPT), "cuda")

    tile = sorted(TILE_DIR.glob("*.jpg"))[0]
    img = np.asarray(Image.open(tile).convert("RGB"))

    pc = sam2_pf.predict_full(craq_model, img, "cuda", tile=512, stride=256)
    pk = unet_pf.predict_full(crack_model, img, "cuda", tile=512, stride=256)
    assert pc.shape[0] == 2 and pc.shape[1:] == img.shape[:2], pc.shape
    assert pk.shape[0] == 2 and pk.shape[1:] == img.shape[:2], pk.shape
    assert np.isfinite(pc).all() and np.isfinite(pk).all()
    print("ENV SMOKE OK:", tile.name, "craq", pc.shape, "crack", pk.shape)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the env smoke**

Run:

```bash
cd /home/zzz90/research && cvat_agent_env/bin/python crack_detection_sam2/cvat_agent/_env_smoke.py
```

Expected: prints `ENV SMOKE OK: <tile>.jpg craq (2, 1024, 1024) crack (2, 1024, 1024)` with no
traceback and no CUDA OOM.

- [ ] **Step 5: (Only if Step 2 or 4 failed on smp/sam2 conflict) record and fall back**

If torch cannot satisfy both SAM-2 and smp, or both models OOM together on 8 GB: stop and report.
The spec's documented fallback is two separate functions/agents (one per env). Do not silently
work around it. Otherwise skip this step.

- [ ] **Step 6: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/cvat_agent/requirements.txt crack_detection_sam2/cvat_agent/_env_smoke.py crack_detection_sam2/cvat_agent/.gitignore
git commit -m "feat(cvat-agent): dedicated env + both-model load smoke"
```

---

## Task 2: Native function module + offline detect test

**Files:**
- Create: `crack_detection_sam2/cvat_agent/craq_crack_func.py`
- Test: `crack_detection_sam2/cvat_agent/test_craq_crack_func.py`

- [ ] **Step 1: Write the failing test**

Create `crack_detection_sam2/cvat_agent/test_craq_crack_func.py`:

```python
"""Offline test: the native function loads and returns valid CVAT mask shapes on a real tile."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import craq_crack_func as F  # noqa: E402
from cvat_sdk.masks import decode_mask  # noqa: E402


class _Ctx:
    conf_threshold = None


def _a_craq_tile():
    # pick a tile that the prior pre-label run found craquelure in (binary_craq > 0)
    binc = Path("/home/zzz90/research/crack_detection_sam2/runs/2026-06-08-prelabel-selected/merged/binary_craq")
    for p in sorted(binc.glob("*.png")):
        if np.asarray(Image.open(p)).any():
            for root in ("batch_1", "batch_2", "batch_3"):
                jpg = Path("/home/zzz90/research/_data/selected_slices") / root / f"{p.stem}.jpg"
                if jpg.exists():
                    return jpg
    raise AssertionError("no craq-positive tile found")


def test_detect_returns_valid_mask_shapes():
    fn = F.create()  # defaults: both final ckpts, tile512/stride256, thr0.5
    spec_names = {l.name for l in fn.spec.labels}
    assert spec_names == {"craquelure", "crack"}

    jpg = _a_craq_tile()
    image = Image.open(jpg).convert("RGB")
    shapes = fn.detect(_Ctx(), image)

    assert isinstance(shapes, list)
    assert len(shapes) >= 1, "expected at least one mask on a craq-positive tile"
    label_ids = {s.label_id for s in shapes}
    assert label_ids <= {0, 1}
    for s in shapes:
        assert s.type == "mask"
        m = decode_mask(s.points, image_width=image.width, image_height=image.height)
        assert m.shape == (image.height, image.width) and m.any()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd /home/zzz90/research/crack_detection_sam2/cvat_agent && /home/zzz90/research/cvat_agent_env/bin/python -m pytest test_craq_crack_func.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'craq_crack_func'` (module not written yet).

- [ ] **Step 3: Write the native function module**

Create `crack_detection_sam2/cvat_agent/craq_crack_func.py`:

```python
"""CVAT native function: auto-annotate craquelure (SAM2 dense-seg) + crack (ResUNet) masks.

Loaded by cvat-cli (create-native / run-agent) via the create() factory. Runs both experts with
sliding-window inference, applies craq-over-crack priority, and returns one CVAT mask shape per
connected component.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import cvat_sdk.auto_annotation as cvataa
from cvat_sdk.masks import encode_mask
from scipy.ndimage import label as cc_label, find_objects

SAM2_ROOT = Path("/home/zzz90/research/crack_detection_sam2")
UNET_SRC = Path("/home/zzz90/research/crack_detection_unet/src")
DEFAULT_CRAQ_CKPT = str(SAM2_ROOT / "runs/expert_craq_v3_final_small/last.pt")
DEFAULT_CRACK_CKPT = "/home/zzz90/research/crack_detection_unet/runs/expert_crack_v3_final_resnet50/last.pt"

for _p in (str(SAM2_ROOT), str(UNET_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _CraqCrackFunction:
    def __init__(self, craq_model, crack_model, sam2_pf, unet_pf,
                 device, tile, stride, thresh_craq, thresh_crack, min_blob, priority):
        self._craq_model = craq_model
        self._crack_model = crack_model
        self._sam2_pf = sam2_pf
        self._unet_pf = unet_pf
        self._device = device
        self._tile = tile
        self._stride = stride
        self._thresh_craq = thresh_craq
        self._thresh_crack = thresh_crack
        self._min_blob = min_blob
        self._priority = priority

    @property
    def spec(self) -> cvataa.DetectionFunctionSpec:
        return cvataa.DetectionFunctionSpec(labels=[
            cvataa.label_spec("craquelure", 0, type="mask"),
            cvataa.label_spec("crack", 1, type="mask"),
        ])

    def detect(self, context, image: PIL.Image.Image):
        img = np.asarray(image.convert("RGB"))
        override = getattr(context, "conf_threshold", None)
        thr_craq = override if override is not None else self._thresh_craq
        thr_crack = override if override is not None else self._thresh_crack

        pc = self._sam2_pf.predict_full(self._craq_model, img, self._device,
                                        tile=self._tile, stride=self._stride)
        pk = self._unet_pf.predict_full(self._crack_model, img, self._device,
                                        tile=self._tile, stride=self._stride)
        craq = pc[1] > thr_craq
        crack = pk[1] > thr_crack
        if self._priority == "craq_over_crack":
            crack = crack & (~craq)

        shapes = []
        shapes += self._masks_for(craq, label_id=0)
        shapes += self._masks_for(crack, label_id=1)
        return shapes

    def _masks_for(self, mask, label_id):
        out = []
        lab, n = cc_label(mask)
        if n == 0:
            return out
        for i, sl in enumerate(find_objects(lab), start=1):
            if sl is None:
                continue
            sub = (lab[sl] == i)
            if int(sub.sum()) < self._min_blob:
                continue
            y1, x1 = sl[0].start, sl[1].start
            y2, x2 = sl[0].stop, sl[1].stop  # slice stop == exclusive upper, matches encode_mask
            comp = np.zeros(mask.shape, dtype=bool)
            comp[sl] = sub
            out.append(cvataa.mask(label_id, encode_mask(comp, [x1, y1, x2, y2])))
        return out


def create(craq_ckpt: str = DEFAULT_CRAQ_CKPT,
           crack_ckpt: str = DEFAULT_CRACK_CKPT,
           device: str = "cuda",
           tile: int = 512,
           stride: int = 256,
           thresh_craq: float = 0.5,
           thresh_crack: float = 0.5,
           min_blob: int = 64,
           priority: str = "craq_over_crack") -> _CraqCrackFunction:
    sam2_pf = _load_module("_sam2_pf", SAM2_ROOT / "predict_full.py")
    unet_pf = _load_module("_unet_pf", UNET_SRC / "predict_full.py")
    craq_model, _ = sam2_pf.load_model_from_ckpt(craq_ckpt, device)
    crack_model, _ = unet_pf.load_model_from_ckpt(crack_ckpt, device)
    return _CraqCrackFunction(craq_model, crack_model, sam2_pf, unet_pf,
                              device, tile, stride, thresh_craq, thresh_crack,
                              int(min_blob), priority)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
cd /home/zzz90/research/crack_detection_sam2/cvat_agent && /home/zzz90/research/cvat_agent_env/bin/python -m pytest test_craq_crack_func.py -v
```

Expected: PASS (`test_detect_returns_valid_mask_shapes`). If pytest is missing:
`/home/zzz90/research/cvat_agent_env/bin/pip install pytest` then rerun.

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/cvat_agent/craq_crack_func.py crack_detection_sam2/cvat_agent/test_craq_crack_func.py
git commit -m "feat(cvat-agent): native function for craq+crack masks + offline test"
```

---

## Task 3: CLI wrapper scripts (register / run agent)

**Files:**
- Create: `crack_detection_sam2/cvat_agent/register.sh`
- Create: `crack_detection_sam2/cvat_agent/run_agent.sh`
- Create: `crack_detection_sam2/cvat_agent/README.md`

- [ ] **Step 1: Write register.sh**

Create `crack_detection_sam2/cvat_agent/register.sh`:

```bash
#!/usr/bin/env bash
# One-time: register the native function on app.cvat.ai. Prints the function-id to reuse in run_agent.sh.
# Requires CVAT_ACCESS_TOKEN exported (Personal Access Token from app.cvat.ai).
set -euo pipefail
: "${CVAT_ACCESS_TOKEN:?export CVAT_ACCESS_TOKEN with your app.cvat.ai PAT}"
CLI=/home/zzz90/research/cvat_agent_env/bin/cvat-cli
FUNC=/home/zzz90/research/crack_detection_sam2/cvat_agent/craq_crack_func.py

"$CLI" --server-host https://app.cvat.ai \
  function create-native "Heritage craq+crack" \
  --function-file "$FUNC"
```

- [ ] **Step 2: Write run_agent.sh**

Create `crack_detection_sam2/cvat_agent/run_agent.sh`:

```bash
#!/usr/bin/env bash
# Keep running while annotating. Usage: ./run_agent.sh <function-id>
set -euo pipefail
: "${CVAT_ACCESS_TOKEN:?export CVAT_ACCESS_TOKEN with your app.cvat.ai PAT}"
FUNC_ID="${1:?usage: run_agent.sh <function-id from register.sh>}"
CLI=/home/zzz90/research/cvat_agent_env/bin/cvat-cli
FUNC=/home/zzz90/research/crack_detection_sam2/cvat_agent/craq_crack_func.py

"$CLI" --server-host https://app.cvat.ai \
  function run-agent "$FUNC_ID" \
  --function-file "$FUNC"
```

- [ ] **Step 3: Write README.md (operator runbook)**

Create `crack_detection_sam2/cvat_agent/README.md`:

```markdown
# Heritage craq+crack CVAT AI agent

Local auto-annotation for `craquelure` + `crack` mask labels on app.cvat.ai. Model runs locally;
only function metadata is uploaded.

## Prereqs
- venv `cvat_agent_env` built (see requirements.txt + spec).
- A Personal Access Token from app.cvat.ai.
- CVAT project/task has labels named exactly `craquelure` and `crack`, both **mask** type.

## Use
```bash
export CVAT_ACCESS_TOKEN=<your PAT>     # do not commit
bash register.sh                         # one-time; note the printed function-id
bash run_agent.sh <function-id>          # keep running while you annotate
```
Then in the CVAT task: Actions / Automatic annotation -> pick "Heritage craq+crack" -> run -> review.

## Params (override via -p name=type:value on the cvat-cli line)
craq_ckpt, crack_ckpt, device, tile(512), stride(256), thresh_craq(0.5), thresh_crack(0.5),
min_blob(64), priority(craq_over_crack).

## Notes
- craquelure quality > crack quality; expect more manual fixing on crack.
- Agent must stay running; stop with Ctrl-C.
```

- [ ] **Step 4: Make scripts executable and verify the CLI is callable**

Run:

```bash
chmod +x /home/zzz90/research/crack_detection_sam2/cvat_agent/register.sh /home/zzz90/research/crack_detection_sam2/cvat_agent/run_agent.sh
/home/zzz90/research/cvat_agent_env/bin/cvat-cli function --help
```

Expected: cvat-cli prints function subcommand help including `create-native` and `run-agent`.

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/cvat_agent/register.sh crack_detection_sam2/cvat_agent/run_agent.sh crack_detection_sam2/cvat_agent/README.md
git commit -m "feat(cvat-agent): register/run wrapper scripts + runbook"
```

---

## Task 4: Live CVAT integration smoke (manual, with user)

This task needs the user's app.cvat.ai account and PAT; it is a guided manual verification, not
automated. Do not fabricate success — report exactly what CVAT shows.

- [ ] **Step 1: User exports the token**

Ask the user to run (in their shell): `export CVAT_ACCESS_TOKEN=<their PAT>` in the same terminal
that will run the agent.

- [ ] **Step 2: Confirm labels**

In app.cvat.ai, verify the target project/task has labels `craquelure` and `crack`, both mask type.
If missing, user adds them.

- [ ] **Step 3: Register the function**

Run `bash crack_detection_sam2/cvat_agent/register.sh` and record the printed function-id.
Expected: a function-id integer; the function appears under the user's CVAT functions list.

- [ ] **Step 4: Start the agent**

Run `bash crack_detection_sam2/cvat_agent/run_agent.sh <function-id>` (leave it running).
Expected: log lines showing the agent connected and waiting for requests; both checkpoints loaded.

- [ ] **Step 5: Trigger auto-annotation on a 1-2 image throwaway task**

In the CVAT UI: Actions -> Automatic annotation -> select "Heritage craq+crack" -> map labels ->
run. Expected: job completes; craquelure and/or crack mask shapes appear on the frames for review.

- [ ] **Step 6: Record the outcome**

Write a short result note (function-id, what appeared, any errors) into the run manifest in Task 5.

---

## Task 5: Run manifest + finish branch

**Files:**
- Create: `crack_detection_sam2/runs/2026-06-08-cvat-agent/manifest.md`

- [ ] **Step 1: Write the manifest**

Create `crack_detection_sam2/runs/2026-06-08-cvat-agent/manifest.md` capturing: goal, the exact
env-build commands (Task 1 Step 2), the function-file path, registration command + resulting
function-id, the live-smoke outcome from Task 4 Step 6, and the spec/plan paths. Mark conclusion
符合/不符合/待查 against the spec's testing section.

- [ ] **Step 2: Add an EXPERIMENTS.md line**

Append to `crack_detection_sam2/EXPERIMENTS.md` under "其他產出（非訓練 run）":
a one-line entry for `2026-06-08-cvat-agent` (CVAT craq+crack assisted-labeling agent, env +
function + scripts, run path).

- [ ] **Step 3: Commit and surface branch for merge decision**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/runs/2026-06-08-cvat-agent/manifest.md crack_detection_sam2/EXPERIMENTS.md
git commit -m "docs(cvat-agent): run manifest + experiments index"
```

Then invoke `superpowers:finishing-a-development-branch` to decide merge/PR/cleanup.

---

## Notes for the implementer
- Run everything with the `cvat_agent_env` interpreter, not sam2_env/unet_env.
- Never echo or commit the PAT. `.gitignore` already guards `*.env`/`.token`.
- If GPU OOM when both models are resident, lower inference batch by passing a smaller `batch_size`
  through the predict_full calls (add a `batch_size` create() param mirroring tile/stride) — only if
  observed, per YAGNI.
