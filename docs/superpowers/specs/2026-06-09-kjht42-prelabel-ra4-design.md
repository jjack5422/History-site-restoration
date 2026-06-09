# Retrain craq/crack on 42 KJTHT tiles, prelabel 16 unlabeled R-A4 tiles — Design

Date: 2026-06-09
Status: approved (design), pending spec review
Owner: zzz90

## 1. Goal

Train binary **craquelure** and **crack** experts on the user's 42-image labeled batch, then run
batch pre-label inference on the 16 unlabeled tiles of the `KJTHT-SC-R-A4-3` panel, and package the
result as a CVAT-importable segmentation-mask zip for the user to review/correct. Only craq + crack
(the other deterioration classes stay manual). This reuses the existing heritage dense-seg + pre-
label pipeline end to end; no new model code.

## 2. Inputs (verified)

- Labels: `/home/zzz90/research/_data/0-41test/` — CVAT Pascal-VOC export, 6-class palette
  (`labelmap.txt`: background, crack=255,24,3, craquelure=102,255,102, flaking, loss, shrinkage).
  `SegmentationClass/<stem>.png` are RGB-palette masks; 184 slots, 100 non-empty.
- **Training set = first 42 entries** of `0-41test/ImageSets/Segmentation/default.txt` (index 0-47
  is the whole KJTHT site; the user chose the strict first 42 = index 0-41). Of these 42, 41 have
  non-empty masks and 1 is empty (`KJTHT-SC-L-A4-4_R2_C05`); 2 are labeled R-A4 tiles
  (`KJTHT-SC-R-A4-3_R2_C03`, `R2_C04`).
- RGB tiles (1024x1024): `/home/zzz90/research/_data/image_1024_slices/<stem>.jpg` (has every KJTHT
  tile, labeled and unlabeled).
- **Pre-label target = 16 unlabeled R-A4 tiles**: of the 24 `KJTHT-SC-R-A4-3` tiles in
  `image_1024_slices`, 8 are labeled (`R2_C03-C06`, `R3_C03-C06`); the 16 unlabeled are
  `R1_C01..C08` (8), `R2_C01,C02,C07,C08` (4), `R3_C01,C02,C07,C08` (4).

## 3. Reused pipeline components (no new model code)

- `crack_detection_sam2/scripts/build_binary_datasets.py` — VOC palette masks + 1024 RGB →
  per-class binary masks + `tiles_512/` (tile 512 / stride 256) + `tile_index.json` +
  `group_split_stem.json` (4-fold LOSO). Args: `--seg_dir --image_dir --out_root_template
  --classes crack craquelure --tile_size 512 --stride 256 --bg_keep_ratio 0.15 --seed 42`.
  `{class}` in the template is replaced by the short name (craquelure→craq, crack→crack).
- craq trainer: `crack_detection_sam2/train.py` (env `sam2_env`) — SAM2 Hiera-small dense-seg head.
  Reads `--split` whose `folds[fold]` has `train`/`val` tile-name lists.
- crack trainer: `crack_detection_unet/src/train.py` (env `unet_env`) — ResUNet resnet50.
- `crack_detection_sam2/predict_full.py` (craq, `sam2_env`) and
  `crack_detection_unet/predict_full.py` (crack, `unet_env`) — sliding-window inference,
  `--save_prob`. Two passes because the envs are mutually exclusive.
- `crack_detection_sam2/scripts/merge_pre_label.py` — craq+crack prob → VOC palette class mask.
- `crack_detection_sam2/scripts/package_cvat_segmask.py` — VOC palette → CVAT "Segmentation mask
  1.1" import layout + `.zip`.

## 4. Architecture / phases

```
[0] stage     42-stem list (default.txt[0:42]) ; copy 42 masks -> _data/_kjht42_seg/
              list 16 unlabeled R-A4 stems ; copy 16 jpgs -> _data/_kjht42_ra4_unlabeled/
[A] dataset   build_binary_datasets.py  --seg_dir _data/_kjht42_seg
                                         --image_dir _data/image_1024_slices
                                         --out_root_template _data/kjht42_{class}
              -> _data/kjht42_craq/tiles_512 , _data/kjht42_crack/tiles_512
              + generate nofold_all_train.json per dataset (all tiles in train & val)
[B] train     craq:  sam2_env  train.py  --tiles_root kjht42_craq/tiles_512
                     --split .../nofold_all_train.json --epochs 50 --class_names background,craquelure
                     --output_dir runs/2026-06-09-kjht42-experts/craq
              crack: unet_env  src/train.py  --tiles_root kjht42_crack/tiles_512
                     --split .../nofold_all_train.json --epochs 50 --class_names background,crack
                     --output_dir runs/2026-06-09-kjht42-experts/crack
[C] prelabel  predict_full (craq, sam2_env) + (crack, unet_env) on _kjht42_ra4_unlabeled
              --tile 512 --stride 256 --save_prob ; merge_pre_label.py -> merged/voc_palette
[D] package   package_cvat_segmask.py --voc_dir merged/voc_palette
                     --labelmap _data/0-41test/labelmap.txt --out_dir <run>/cvat_import --zip
              -> <run>/cvat_import.zip  (16 R-A4 tiles, craq+crack, ready to import)
```

Output artifacts live under `crack_detection_sam2/runs/2026-06-09-kjht42-prelabel-ra4/` (prelabel +
package) and `.../runs/2026-06-09-kjht42-experts/` (the two checkpoints), per the experiment-
tracking convention (manifest + commands + git SHA).

## 5. Key decisions

- **New dataset dirs `_data/kjht42_{craq,crack}`** — do NOT reuse/overwrite the existing
  `_data/labeled32_{craq,crack}_v3`.
- **Train on all 42, no held-out fold** (final-expert style, like `expert_*_v3_final`): use a
  generated `nofold_all_train.json` with every tile in both `train` and `val`; keep `last.pt`.
  No fold evaluation — this is a pre-label accelerator the user reviews in CVAT.
- **Hyperparameters** mirror the prior final experts: craq epochs 50, base_lr 3e-4, batch 4,
  freeze_trunk, median_freq; crack epochs 50, base_lr 3e-4, batch 8, resnet50/imagenet,
  median_freq. AMP on.
- **Thresholds** for merge: craq 0.5 / crack 0.5 (pipeline default).
- **Two classes stay distinct in the output**: `merge_pre_label.py` writes a VOC palette mask that
  keeps craquelure (green 102,255,102) and crack (red 255,24,3) as **separate** classes — it does
  NOT collapse them into one. Overlap is resolved by `--priority craq_over_crack` (craquelure
  overrides crack where both fire). It also emits per-class `binary_craq/` and `binary_crack/`
  masks. So in CVAT the import shows two distinct labels.

## 6. Verification (cheap, between phases)

1. After [A]: `kjht42_craq/tiles_512/tile_index.json` and `kjht42_crack/.../tile_index.json` each
   list > 0 tiles; craq dataset has tiles with foreground (craquelure present in the 42).
2. After [A] split-gen: each `nofold_all_train.json` `folds[0]["train"]` length == number of tiles
   in that dataset, and `["val"]` == `["train"]`.
3. After [B]: `runs/2026-06-09-kjht42-experts/{craq,crack}/last.pt` exist; training loss decreased.
4. After [C]: 16 VOC palette masks produced (one per unlabeled R-A4 tile); craq channel has
   non-zero pixels on at least some tiles (sanity that inference ran and the panel has craquelure).
5. After [D]: `cvat_import.zip` exists with `SegmentationClass/` containing the 16 stems and a
   `labelmap.txt`; it unzips to the CVAT "Segmentation mask 1.1" layout.

## 7. Constraints / risks / non-goals

- **crack quality risk**: the 42 KJTHT tiles are craquelure-dominant; crack pixels are sparse, so
  the crack expert will be weaker than craq (consistent with prior results). craq pre-label is the
  reliable part; crack pre-label needs more manual fixing. Acceptable — it is an accelerator.
- The single empty training tile contributes only background; `bg_keep_ratio=0.15` drops most pure-
  background tiles, so it has negligible effect.
- Two-env split (sam2_env / unet_env) is mandatory for both training and inference; they cannot
  share an interpreter.
- Non-goals: no flaking/loss/shrinkage, no multi-class model, no held-out evaluation, no changes to
  the existing `labeled32_*_v3` datasets or the prior experts.

## 8. Open items (resolved)

- "42" = default.txt index 0-41 (user confirmed strict first 42 over the full 48-tile KJTHT block).
- Split format: `train.py` and `crack_detection_unet/src/train.py` both read `folds[fold]["train"]`
  / `["val"]` tile-name lists (`load_split`), so the generated nofold file uses those keys.
- Envs: `/home/zzz90/research/sam2_env` (craq) and `/home/zzz90/research/unet_env` (crack); no
  system `python`.
