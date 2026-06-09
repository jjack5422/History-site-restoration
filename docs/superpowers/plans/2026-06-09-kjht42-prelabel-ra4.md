# KJHT-42 retrain + R-A4 prelabel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain binary craquelure + crack experts on the user's first-42 KJTHT labeled tiles, then batch pre-label the 16 unlabeled `KJTHT-SC-R-A4-3` tiles and package them as a CVAT-importable segmentation-mask zip (craquelure and crack kept as separate classes).

**Architecture:** Pure orchestration of the existing heritage pipeline — no new model code. One small tested helper (`make_nofold_split.py`) generates an all-tiles "train on everything" split. The rest are command-runs with a verification gate after each phase: build binary datasets → train two experts (two envs) → two-pass inference + merge → package.

**Tech Stack:** existing `build_binary_datasets.py`, `train.py` (SAM2, `sam2_env`), `crack_detection_unet/src/train.py` (ResUNet, `unet_env`), `predict_full.py` (both envs), `merge_pre_label.py`, `package_cvat_segmask.py`. No system `python` — use `/home/zzz90/research/sam2_env/bin/python` and `/home/zzz90/research/unet_env/bin/python`.

**Spec:** `docs/superpowers/specs/2026-06-09-kjht42-prelabel-ra4-design.md`

**Conventions:**
- Run dirs: experts → `crack_detection_sam2/runs/2026-06-09-kjht42-experts/{craq,crack}` (crack actually trains in `crack_detection_unet`, output there); prelabel + package → `crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4/`.
- New datasets `_data/kjht42_craq`, `_data/kjht42_crack` — never touch existing `_data/labeled32_*_v3`.
- Commit messages end with the trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` (omitted from `-m` templates below for brevity).
- These phases are long GPU runs; inline execution with checkpoints is the natural fit.

---

## Task 1: `make_nofold_split.py` helper (train-on-everything split)

**Files:**
- Create: `crack_detection_sam2/scripts/make_nofold_split.py`
- Test: `crack_detection_sam2/tests/test_make_nofold_split.py`

- [ ] **Step 1: Write the failing test** — create `crack_detection_sam2/tests/test_make_nofold_split.py`:

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from make_nofold_split import build_nofold


def test_build_nofold_all_tiles_train_and_val():
    idx = {"items": [{"tile": "a.png"}, {"tile": "b.png"}, {"tile": "c.png"}]}
    out = build_nofold(idx)
    assert len(out["folds"]) == 1
    f = out["folds"][0]
    assert f["fold"] == 0
    assert f["train"] == ["a.png", "b.png", "c.png"]
    assert f["val"] == f["train"]          # final-expert style: eval on all, keep last.pt
    print("OK test_make_nofold_split")


if __name__ == "__main__":
    test_build_nofold_all_tiles_train_and_val()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python tests/test_make_nofold_split.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'make_nofold_split'`.

- [ ] **Step 3: Implement** — create `crack_detection_sam2/scripts/make_nofold_split.py`:

```python
"""Generate a no-fold 'train on everything' split for a tiles dataset.

Reads <tiles_root>/tile_index.json and writes <tiles_root>/nofold_all_train.json with a single
fold whose train and val both contain every tile (final-expert style: train on all, keep last.pt).
Both train.py and crack_detection_unet/src/train.py read folds[fold]["train"]/["val"] tile lists.
"""
from __future__ import annotations

import argparse
import json
import os


def build_nofold(tile_index):
    tiles = [it["tile"] for it in tile_index["items"]]
    return {
        "n_splits": 1,
        "group_by": "stem",
        "folds": [{"fold": 0, "train": tiles, "val": tiles}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_root", required=True)
    args = ap.parse_args()
    with open(os.path.join(args.tiles_root, "tile_index.json")) as f:
        idx = json.load(f)
    out = build_nofold(idx)
    path = os.path.join(args.tiles_root, "nofold_all_train.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}: {len(out['folds'][0]['train'])} tiles (train==val)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python tests/test_make_nofold_split.py`
Expected: `OK test_make_nofold_split`.

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/scripts/make_nofold_split.py crack_detection_sam2/tests/test_make_nofold_split.py
git commit -m "feat(kjht42): make_nofold_split helper (train-on-everything split)"
```

---

## Task 2: Stage inputs + build binary datasets + generate splits (Phase 0 + A)

**Files:**
- Create (data, not committed): `_data/_kjht42_seg/`, `_data/_kjht42_ra4_unlabeled/`, `_data/kjht42_craq/`, `_data/kjht42_crack/`
- Create (committed): `crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4/` stem lists

- [ ] **Step 1: Stage the 42 training masks and the 16 unlabeled R-A4 images**

Run:
```bash
cd /home/zzz90/research
RUN=crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4
mkdir -p "$RUN" _data/_kjht42_seg _data/_kjht42_ra4_unlabeled

# 42 training stems = first 42 lines of default.txt
head -n 42 _data/0-41test/ImageSets/Segmentation/default.txt > "$RUN/train42_stems.txt"
while read -r s; do cp "_data/0-41test/SegmentationClass/$s.png" _data/_kjht42_seg/; done < "$RUN/train42_stems.txt"

# labeled R-A4 stems (the 8 with masks) and all R-A4 tiles present as raw images
ls _data/0-41test/SegmentationClass | grep '^KJTHT-SC-R-A4-3_' | sed 's/\.png$//' | sort > "$RUN/ra4_labeled_stems.txt"
ls _data/image_1024_slices       | grep '^KJTHT-SC-R-A4-3_' | sed 's/\.jpg$//' | sort > "$RUN/ra4_all_stems.txt"
# unlabeled = all minus labeled
comm -23 "$RUN/ra4_all_stems.txt" "$RUN/ra4_labeled_stems.txt" > "$RUN/ra4_unlabeled_stems.txt"
while read -r s; do cp "_data/image_1024_slices/$s.jpg" _data/_kjht42_ra4_unlabeled/; done < "$RUN/ra4_unlabeled_stems.txt"
```

- [ ] **Step 2: Verify staging**

Run:
```bash
cd /home/zzz90/research
echo "staged masks:   $(ls _data/_kjht42_seg | wc -l)  (expect 42)"
echo "unlabeled R-A4: $(ls _data/_kjht42_ra4_unlabeled | wc -l)  (expect 16)"
cat crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4/ra4_unlabeled_stems.txt
```
Expected: 42 staged masks; 16 unlabeled R-A4 jpgs; the 16 list is `R1_C01..C08`, `R2_C01,C02,C07,C08`, `R3_C01,C02,C07,C08`. If counts differ, STOP and report.

- [ ] **Step 3: Build binary craq + crack datasets**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
/home/zzz90/research/sam2_env/bin/python scripts/build_binary_datasets.py \
  --seg_dir /home/zzz90/research/_data/_kjht42_seg \
  --image_dir /home/zzz90/research/_data/image_1024_slices \
  --out_root_template /home/zzz90/research/_data/kjht42_{class} \
  --classes crack craquelure --tile_size 512 --stride 256 --seed 42
```
Expected: creates `/home/zzz90/research/_data/kjht42_craq/tiles_512/` and `/home/zzz90/research/_data/kjht42_crack/tiles_512/`, each with `images/`, `masks/`, `tile_index.json`, `group_split_stem.json`.

- [ ] **Step 4: Generate train-on-everything splits**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
/home/zzz90/research/sam2_env/bin/python scripts/make_nofold_split.py --tiles_root /home/zzz90/research/_data/kjht42_craq/tiles_512
/home/zzz90/research/sam2_env/bin/python scripts/make_nofold_split.py --tiles_root /home/zzz90/research/_data/kjht42_crack/tiles_512
```
Expected: each prints `wrote .../nofold_all_train.json: N tiles (train==val)` with N > 0.

- [ ] **Step 5: Verify datasets and commit the run stem lists**

Run:
```bash
cd /home/zzz90/research
for c in craq crack; do
  /home/zzz90/research/sam2_env/bin/python -c "import json; d=json.load(open('_data/kjht42_$c/tiles_512/tile_index.json')); n=len(d['items']); fg=sum(1 for it in d['items'] if it.get('has_fg')); print('$c tiles:',n,'fg tiles:',fg)"
done
git add crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4/*.txt
git commit -m "exp(kjht42): stage 42-tile train set + 16 unlabeled R-A4 list"
```
Expected: both datasets have tiles > 0; `craq` has fg tiles > 0 (craquelure is present in the 42). If `craq` fg == 0, STOP and report (palette/class extraction bug). `crack` fg may be small (sparse cracks) — that is expected, not a failure.

---

## Task 3: Train the craquelure expert (SAM2-small, `sam2_env`)

**Files:** Create `crack_detection_sam2/runs/2026-06-09-kjht42-experts/craq/` (checkpoints + logs)

- [ ] **Step 1: Run craq training (all 42 tiles, 50 epochs)**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
/home/zzz90/research/sam2_env/bin/python train.py \
  --tiles_root /home/zzz90/research/_data/kjht42_craq/tiles_512 \
  --split /home/zzz90/research/_data/kjht42_craq/tiles_512/nofold_all_train.json \
  --fold 0 --variant small --image_size 512 --batch_size 4 --epochs 50 \
  --warmup_epochs 2 --base_lr 3e-4 --encoder_lr_mult 0.1 --class_weight_mode median_freq \
  --class_names background,craquelure \
  --output_dir runs/2026-06-09-kjht42-experts/craq
```
Expected: trains 50 epochs; `[val]` IoU rises; writes `runs/2026-06-09-kjht42-experts/craq/last.pt` and `log.json`.

- [ ] **Step 2: Verify checkpoint and loss trend**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
ls -la runs/2026-06-09-kjht42-experts/craq/last.pt
/home/zzz90/research/sam2_env/bin/python -c "import json; h=json.load(open('runs/2026-06-09-kjht42-experts/craq/log.json'))['history']; print('ep1 loss', h[0]['train']['loss'] if isinstance(h[0]['train'],dict) else h[0]['train'], '-> last', h[-1].get('val'))"
```
Expected: `last.pt` exists; training loss decreased from epoch 1 to last. If loss is flat or `last.pt` missing, STOP and report.

- [ ] **Step 3: Commit the run (checkpoints + logs)**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/runs/2026-06-09-kjht42-experts/craq
git commit -m "exp(kjht42): craquelure expert (SAM2-small) trained on 42 KJTHT tiles"
```

---

## Task 4: Train the crack expert (ResUNet resnet50, `unet_env`)

**Files:** Create `crack_detection_unet/runs/2026-06-09-kjht42-experts/crack/` (checkpoints + logs)

- [ ] **Step 1: Run crack training (all 42 tiles, 50 epochs)**

Run:
```bash
cd /home/zzz90/research/crack_detection_unet
/home/zzz90/research/unet_env/bin/python src/train.py \
  --tiles_root /home/zzz90/research/_data/kjht42_crack/tiles_512 \
  --split /home/zzz90/research/_data/kjht42_crack/tiles_512/nofold_all_train.json \
  --fold 0 --encoder resnet50 --encoder_weights imagenet --image_size 512 \
  --batch_size 8 --epochs 50 --warmup_epochs 2 --base_lr 3e-4 --encoder_lr_mult 0.1 \
  --class_weight_mode median_freq --class_names background,crack \
  --output_dir runs/2026-06-09-kjht42-experts/crack
```
Expected: trains 50 epochs; writes `crack_detection_unet/runs/2026-06-09-kjht42-experts/crack/last.pt` and `log.json`.

- [ ] **Step 2: Verify checkpoint**

Run:
```bash
ls -la /home/zzz90/research/crack_detection_unet/runs/2026-06-09-kjht42-experts/crack/last.pt
```
Expected: `last.pt` exists. (crack val IoU may be low — sparse crack labels in this set, expected per spec.) If `last.pt` missing or training errored, STOP and report.

- [ ] **Step 3: Commit the run**

```bash
cd /home/zzz90/research
git add crack_detection_unet/runs/2026-06-09-kjht42-experts/crack
git commit -m "exp(kjht42): crack expert (ResUNet resnet50) trained on 42 KJTHT tiles"
```

---

## Task 5: Pre-label the 16 unlabeled R-A4 tiles (two-pass inference + merge)

**Files:** Create `crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4/{craq_raw,crack_raw,merged}/`

- [ ] **Step 1: craquelure inference (SAM2, `sam2_env`)**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
RUN=runs/2026-06-09-kjht42-prelabel-ra4
IMG=/home/zzz90/research/_data/_kjht42_ra4_unlabeled
/home/zzz90/research/sam2_env/bin/python predict_full.py \
  --ckpt runs/2026-06-09-kjht42-experts/craq/last.pt \
  --image_dir "$IMG" --out_dir "$RUN/craq_raw" \
  --tile 512 --stride 256 --batch_size 4 --save_prob
```
Expected: writes `$RUN/craq_raw/prob/<stem>.npy` for the 16 tiles.

- [ ] **Step 2: crack inference (ResUNet, `unet_env`)**

Run:
```bash
cd /home/zzz90/research/crack_detection_unet
IMG=/home/zzz90/research/_data/_kjht42_ra4_unlabeled
OUT=/home/zzz90/research/crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4/crack_raw
/home/zzz90/research/unet_env/bin/python src/predict_full.py \
  --ckpt runs/2026-06-09-kjht42-experts/crack/last.pt \
  --image_dir "$IMG" --out_dir "$OUT" \
  --tile 512 --stride 256 --batch_size 4 --save_prob
```
Expected: writes `crack_raw/prob/<stem>.npy` for the 16 tiles.

- [ ] **Step 3: Merge to VOC palette (craquelure + crack kept distinct, craq overrides overlap)**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
RUN=runs/2026-06-09-kjht42-prelabel-ra4
IMG=/home/zzz90/research/_data/_kjht42_ra4_unlabeled
/home/zzz90/research/sam2_env/bin/python scripts/merge_pre_label.py \
  --craq_prob_dir "$RUN/craq_raw/prob" --crack_prob_dir "$RUN/crack_raw/prob" \
  --image_dir "$IMG" --out_dir "$RUN/merged" \
  --craq_thresh 0.5 --crack_thresh 0.5 --priority craq_over_crack
```
Expected: writes `$RUN/merged/voc_palette/<stem>.png` (16), plus `binary_craq/`, `binary_crack/`, `overlay/`.

- [ ] **Step 4: Verify pre-label output**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4
echo "voc masks: $(ls merged/voc_palette | wc -l)  (expect 16)"
/home/zzz90/research/sam2_env/bin/python -c "
import numpy as np, glob
from PIL import Image
craq=(102,255,102); crack=(255,24,3)
nq=nc=0
for f in glob.glob('merged/voc_palette/*.png'):
    a=np.array(Image.open(f).convert('RGB'))
    nq+= (np.all(a==craq,axis=-1)).any(); nc+=(np.all(a==crack,axis=-1)).any()
print('tiles with craquelure:',nq,'| tiles with crack:',nc)"
```
Expected: 16 voc masks; craquelure present on several tiles (sanity that inference ran). crack may be on few/zero (sparse) — not a failure. If 0 voc masks or all-empty, STOP and report.

- [ ] **Step 5: Commit overlays + binary masks (skip large prob .npy)**

```bash
cd /home/zzz90/research
RUN=crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4
printf '%s\n' 'craq_raw/' 'crack_raw/' > "$RUN/.gitignore"   # raw prob .npy are large, do not commit
git add "$RUN/.gitignore" "$RUN/merged/voc_palette" "$RUN/merged/binary_craq" "$RUN/merged/binary_crack" "$RUN/merged/overlay"
git commit -m "exp(kjht42): pre-label 16 unlabeled R-A4 tiles (craq+crack VOC palette)"
```

---

## Task 6: Package CVAT import zip + run manifest (Phase D)

**Files:** Create `crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4/cvat_import/` (+ `.zip`), `manifest.md`

- [ ] **Step 1: Package to CVAT "Segmentation mask 1.1" layout**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
RUN=runs/2026-06-09-kjht42-prelabel-ra4
/home/zzz90/research/sam2_env/bin/python scripts/package_cvat_segmask.py \
  --voc_dir "$RUN/merged/voc_palette" \
  --labelmap /home/zzz90/research/_data/0-41test/labelmap.txt \
  --out_dir "$RUN/cvat_import" --zip
```
Expected: creates `$RUN/cvat_import/` with `labelmap.txt`, `ImageSets/Segmentation/default.txt`, `SegmentationClass/` (16), `SegmentationObject/` (16), and `$RUN/cvat_import.zip`.

- [ ] **Step 2: Verify the zip layout**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4
unzip -l cvat_import.zip | grep -E 'labelmap|SegmentationClass/.*png|default.txt' | head
echo "class pngs in zip: $(unzip -l cvat_import.zip | grep -c 'SegmentationClass/.*png')  (expect 16)"
```
Expected: zip contains `labelmap.txt`, `ImageSets/Segmentation/default.txt`, and 16 `SegmentationClass/*.png`. If not, STOP and report.

- [ ] **Step 3: Write the run manifest**

Create `crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4/manifest.md` documenting: purpose, the exact commands from Tasks 2-6, data paths (`_data/_kjht42_seg`, `_data/kjht42_{craq,crack}`, `_data/_kjht42_ra4_unlabeled`), the two expert checkpoints + their final val metrics (from each `log.json`), the 16 R-A4 stems, envs (`sam2_env` / `unet_env`), git SHA, and the craq-over-crack merge policy. Mirror the format of `runs/2026-06-08-prelabel-image/manifest.md`.

- [ ] **Step 4: Commit package + manifest**

```bash
cd /home/zzz90/research
RUN=crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4
git add "$RUN/cvat_import" "$RUN/cvat_import.zip" "$RUN/manifest.md"
git commit -m "exp(kjht42): CVAT import zip for 16 R-A4 pre-labels + manifest"
```

- [ ] **Step 5: Hand off**

Report the path to `cvat_import.zip`. The user imports it into the CVAT task via **Import annotations → Segmentation mask 1.1**, then reviews/corrects the 16 R-A4 tiles (craquelure reliable; crack needs more manual fixing per spec).

---

## Self-Review notes (reconciled against spec)

- Spec coverage: dataset build + new dirs (Task 2), train-on-all split (Task 1+2), craq expert (Task 3), crack expert (Task 4), two-pass prelabel + merge keeping classes distinct (Task 5), CVAT package + manifest (Task 6). Verification gates match spec section 6.
- The only new code is `make_nofold_split.py` (TDD, Task 1); everything else reuses existing scripts with verified arg names.
- Paths/names consistent: `_data/kjht42_{craq,crack}` (from `short={"craquelure":"craq"}`), `nofold_all_train.json`, expert ckpts at `runs/2026-06-09-kjht42-experts/{craq,crack}/last.pt`, prelabel at `runs/2026-06-09-kjht42-prelabel-ra4/`.
- crack quality is flagged as a known risk, not a failure condition, in the Task 4/5 verifications.
