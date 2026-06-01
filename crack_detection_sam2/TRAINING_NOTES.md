# crack_detection_sam2 訓練流程整理

本文整理 `/home/zzz90/research/crack_detection_sam2`：以 SAM2.1 (Segment Anything 2) Hiera image encoder 為 backbone、外接輕量 FPN seg head 的多類別語意分割流程。資料來源為實際讀取以下檔案：

- `train.py`、`model_seg.py`、`model.py`、`losses.py`、`metrics.py`
- `dataset.py`、`augment.py`、`data_utils.py`
- `scripts/tile_pairs.py`、`scripts/make_group_split.py`、`scripts/remap_to_2class.py`、`scripts/plot_training.py`
- `predict_full.py`、`configs/default.yaml`
- 各 run 的 `args.json` / `log.json` / stdout log（`outputs/stem_fold0_small*` 系列）
- `data/tiles_512*/tile_index.json`、各自的 `group_split_stem.json`
- `merged_4class_mask_semantic{,_v2}/`、`merged_2class_crack_craquelure_v2/`

執行環境：`/home/zzz90/research/sam2_env/bin/python`（virtualenv，python 3.12）。

---

## 1. 任務與類別

支援三套 class config，CLI 用 `--class_names` 切換：

| 類別 set                | CLASS_NAMES                                              | 對應 mask 來源                              |
|-------------------------|----------------------------------------------------------|---------------------------------------------|
| 5-class (預設)          | background, crack, loss, shrinkage, craquelure           | `merged_4class_mask_semantic{,_v2}/masks`   |
| 3-class (crack+craquelure) | background, crack, craquelure                          | `merged_2class_crack_craquelure_v2/masks`   |

5-class palette（`predict_full.py`）：
```
0 background  (0,0,0)
1 crack       (255,0,0)
2 loss        (0,255,255)
3 shrinkage   (255,255,0)
4 craquelure  (255,0,255)
```

mIoU / mDice 等 macro 指標預設「不含 background」(`metrics.py` `ConfusionMeter.compute(..., ignore_index=0)`)。

### Class priority（多類重疊規則）

mask 產生階段（在 `merged_4class_mask_semantic_v2/overlap_summary.json` 紀錄）使用的覆寫順序：
```
crack -> loss -> shrinkage -> craquelure   (後者 overwrite 前者)
```
所以 craquelure 在重疊時優先於 crack。3-class 版本經 `scripts/remap_to_2class.py` 由 v2 source mask 重映射而來：

```
{0:0, 1:1, 2:0, 3:0, 4:2}    # loss/shrinkage 併入 background
```

→ 重映射後維持「craquelure 蓋過 crack」的規則。

`predict_full.py` 也提供 inference 時的 priority post-processing：`--craq_dilate N` 將 craquelure 區膨脹 N 像素，內部被預測為 crack 的像素改判 craquelure。

---

## 2. 模型架構 (`model_seg.py`)

### Backbone
- 來自 SAM2.1 官方 image encoder（Hiera ViT + FpnNeck），透過 `model.py:build_sam2_model` → `sam2.build_sam.build_sam2` 載入。
- 四個 variant 對應 checkpoint（`checkpoints/`）：

| variant | config                                        | checkpoint                       |
|---------|-----------------------------------------------|----------------------------------|
| tiny    | `configs/sam2.1/sam2.1_hiera_t.yaml`          | `sam2.1_hiera_tiny.pt`           |
| small   | `configs/sam2.1/sam2.1_hiera_s.yaml`          | `sam2.1_hiera_small.pt`（實跑） |
| base    | `configs/sam2.1/sam2.1_hiera_b+.yaml`         | `sam2.1_hiera_base_plus.pt`      |
| large   | `configs/sam2.1/sam2.1_hiera_l.yaml`          | `sam2.1_hiera_large.pt`          |

- 載入後立即 `del sam2`，只保留 `image_encoder`，丟掉 mask decoder / memory 模組省顯存。

### Seg head — `FPNSegHead`
- 取 `image_encoder.neck.backbone_fpn` 中 `n_levels = len(backbone_channel_list) - scalp` 個 level（strides 4 / 8 / 16，每層 256ch），1x1 conv 投影到 `hidden=128`。
- 對齊到最高解析度（stride 4）後 concat → `Conv3x3 + BN + ReLU + Dropout2d(0.1)` → `Conv1x1` 輸出 `num_classes` logits。
- forward 結尾 bilinear 上採樣回原輸入 H×W。
- `num_classes` 來自 `dataset.NUM_CLASSES`，受 `set_class_names()` 動態調整。

### Pretrain / Freeze
- **是 pretrain**：完全沿用 SAM2.1 官方 image encoder 權重（在 SA-1B + 影片資料上訓練），head 為新 init conv。
- 預設 `freeze_trunk=True`、`freeze_neck=False`：Hiera trunk 凍結、FpnNeck + head 可訓。
- CLI 開關 `--no_freeze_trunk`、`--freeze_neck`。
- `param_groups()` 切兩個 group：head 用 `base_lr`，可訓的 encoder 部分用 `base_lr * encoder_lr_mult`。
- 實際 small variant：總 ≈ 34.9M / trainable ≈ 0.91M（log 第一行）。

> SAM2 pretrain 解析度為 1024，512 輸入時 pos embed 會 bilinear interpolate；若品質不夠可改 1024。

---

## 3. 資料流程

### 3.1 Source mask 版本
| 目錄                                       | 類別 | 說明                                                                 |
|--------------------------------------------|------|----------------------------------------------------------------------|
| `merged_4class_mask_semantic/`             | 5    | 第一版                                                               |
| `merged_4class_mask_semantic_v2/`          | 5    | v2：craquelure mask 已做 connected-component area filter (min_area=100) |
| `merged_2class_crack_craquelure_v2/`       | 3    | 由上者重映射而來，loss/shrinkage→bg                                  |

### 3.2 切片 (`scripts/tile_pairs.py`)
共同設定：`size=512`、`stride=256`、`bg_std_threshold=5.0`、`bg_keep_ratio=0.15`、`seed=42`。對純背景 tile：std<5 直接丟，否則隨機 15% 保留。

| 切片輸出                          | num_classes | total | kept_fg | kept_bg | tiles index |
|-----------------------------------|-------------|-------|---------|---------|-------------|
| `data/tiles_512/`                 | 5           | 608   | 530     | 11      | 541         |
| `data/tiles_512_v2/`              | 5           | 608   | 504     | 15      | 519         |
| `data/tiles_512_2class_v2/`       | 3           | 608   | 423     | 28      | 451         |

各版本 class pixel counts (含 train+val)：

```
v1 5-class : bg=134,662,120  crack=858,494   loss=2,439,574  shrinkage=1,212,590  craquelure=2,647,126
v2 5-class : bg=130,134,201  crack=860,020   loss=2,448,600  shrinkage=1,212,591  craquelure=1,397,324
v2 3-class : bg=115,969,600  crack=860,020                                         craquelure=1,397,324
```

### 3.3 Group split (`scripts/make_group_split.py`)
4 folds、seed=42，`group_by=stem`（每張原圖一 group），避免同 stem 的 tile 同時出現在 train/val。

| 切片                          | groups | fold0 train / val |
|-------------------------------|--------|-------------------|
| `tiles_512`                   | 28     | 420 / 121         |
| `tiles_512_v2`                | 27     | 374 / 145         |
| `tiles_512_2class_v2`         | 27     | 321 / 130         |

### 3.4 Dataset / Augmentation
`dataset.py` `TileSegDataset`：讀 RGB image + 單通道 label mask。

`augment.py`（Albumentations）：

訓練：
```
RandomResizedCrop 512, scale=(0.6,1.0), ratio=(0.75,1.33), p=1
HorizontalFlip p=0.5
VerticalFlip   p=0.5
RandomRotate90 p=0.5
Affine translate=±5%, scale=(0.9,1.1), rotate=±15°, p=0.5
ElasticTransform alpha=40 sigma=6 p=0.3
RandomBrightnessContrast ±0.2 p=0.5
GaussNoise p=0.2
MotionBlur blur_limit=5 p=0.2
Normalize ImageNet mean/std
ToTensorV2
```

驗證：
```
LongestMaxSize 512
PadIfNeeded 512x512 fill=0 fill_mask=0
Normalize ImageNet
ToTensorV2
```

> Mask 全程 `mask_interpolation=0`（nearest）避免 label id 被插出小數。

---

## 4. Loss

`losses.py`：`CEDiceLoss = ce_weight·CE + dice_weight·Dice`

- **CE**：可帶 per-class weight。
- **Dice**：multi-class soft Dice (macro 平均)，`ignore_index_in_dice=0` 略過 background，與 mIoU 一致。
- **class weight**（`dataset.py:compute_class_weights`）：
  - `median_freq`（預設）：`w_c = median(freq) / freq_c`（不存在類別權重 0）
  - `inv_sqrt`：`w_c = sqrt(total / (C · count_c))`
  - `none`：不加權

實際各 run fold0 train 的 median_freq class weights：

| run                            | weights                                              |
|--------------------------------|------------------------------------------------------|
| stem_fold0_small (v1, 5-class) | `[0.0192, 3.0439, 1.0, 1.7548, 0.9736]`              |
| stem_fold0_small_v2 (5-class)  | `[0.0105, 1.3594, 0.6236, 1.0, 1.0927]`              |
| stem_fold0_small_2class_v2 (3) | (median_freq, 由 `[33.4M, 0.135M, 0.496M]` 推得，crack 權重最高) |

---

## 5. Optimization

`train.py:main`：

- **Optimizer**：AdamW，兩個 param group（head, encoder），`weight_decay=1e-4`
- **LR schedule**：linear warmup + cosine decay (`cosine_with_warmup`)
  - warmup: `step / max(1, warmup_steps)` (0→1)
  - decay : `0.5 * (1 + cos(π·progress))` (1→0)
  - 各 group lr = `base_lr_g * scale`
- **AMP**：預設 fp16 (`torch.amp.autocast("cuda", dtype=torch.float16)` + `GradScaler`)；`--no_amp` 關掉。
- **Gradient clip**：`clip_grad_norm_(.., max_norm=5.0)`，僅對 `requires_grad` 參數。
- **Optim step 流程**：`zero_grad(set_to_none=True)` → forward(autocast) → `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → grad clip → `scaler.step` → `scaler.update`。

---

## 6. CLI 旗標

```
python train.py
  --tiles_root          (default data/tiles_512)
  --split               (default data/tiles_512/group_split_stem.json)
  --fold N              (default 0)
  --variant {tiny,small,base,large}   (default small)
  --image_size N        (default 512)
  --batch_size N        (default 4)
  --num_workers N       (default 4)
  --epochs N            (default 60)
  --warmup_epochs N     (default 2)
  --base_lr F           (default 3e-4)
  --encoder_lr_mult F   (default 0.1)
  --weight_decay F      (default 1e-4)
  --ce_weight F         (default 0.5)
  --dice_weight F       (default 0.5)
  --class_weight_mode {median_freq, inv_sqrt, none}   (default median_freq)
  --freeze_trunk / --no_freeze_trunk     (default freeze)
  --freeze_neck                          (default off)
  --no_amp                               (default AMP on)
  --seed N              (default 42)
  --output_dir DIR      (default outputs/run)
  --log_interval N      (default 20)
  --class_names CSV     (e.g. "background,crack,craquelure"; 未指定則用 dataset 預設 5 類)
```

---

## 7. 已完成的訓練 run

三次 run 都用 fold=0, variant=small, image_size=512, batch_size=4, base_lr=3e-4, encoder_lr_mult=0.1, weight_decay=1e-4, ce_weight=dice_weight=0.5, class_weight_mode=median_freq, freeze_trunk=True, freeze_neck=False, AMP on, seed=42, warmup_epochs=2, epochs=80, log_interval=50。

差異只有資料 / 類別。

| run                                  | tiles_root                  | classes (含 bg) | iters/ep | best mIoU @ ep |
|--------------------------------------|-----------------------------|-----------------|----------|----------------|
| `outputs/stem_fold0_small`           | `data/tiles_512`            | 5               | 105      | **0.1755 @ ep63** |
| `outputs/stem_fold0_small_v2`        | `data/tiles_512_v2`         | 5               | 93       | **0.2503 @ ep59** |
| `outputs/stem_fold0_small_2class_v2` | `data/tiles_512_2class_v2`  | 3               | 80       | **0.1517 @ ep46** |

> total_steps = epochs × iters/ep；warmup_steps = warmup_epochs × iters/ep。

### 7.1 stem_fold0_small (v1 5-class) — best mIoU=0.1755 @ ep63
```
crack       IoU=0.0601  Dice=0.1134
loss        IoU=0.2394  Dice=0.3863
shrinkage   IoU=0.2130  Dice=0.3512
craquelure  IoU=0.1897  Dice=0.3189
macro       IoU=0.1755  Dice=0.2924
pixel_acc   = 0.8079
```
crack 最弱，整體偏 high recall / low precision。

### 7.2 stem_fold0_small_v2 (v2 5-class) — best mIoU=0.2503 @ ep59
```
crack       IoU=0.0461  Dice=0.0881
loss        IoU=0.4503  Dice=0.6210
shrinkage   IoU=0.3189  Dice=0.4836
craquelure  IoU=0.1860  Dice=0.3136
macro       IoU=0.2503  Dice=0.3766
pixel_acc   = 0.8628
```
v2 mask 把 craquelure 雜點清掉後 loss / shrinkage 大幅提升，整體 mIoU +0.075。crack 仍 ~0.046。

### 7.3 stem_fold0_small_2class_v2 (v2, bg/crack/craquelure) — best mIoU=0.1517 @ ep46
```
crack       IoU=0.0638  Dice=0.1199
craquelure  IoU=0.2396  Dice=0.3866
macro       IoU=0.1517  Dice=0.2532
pixel_acc   = 0.9113
```
雖然 macro mIoU 較 v2 5-class 低（少了 loss / shrinkage 兩個容易類），但 craquelure / crack 各自指標都比 v2 5-class 好（craquelure 0.186 → 0.240，crack 0.046 → 0.064）。

> 三個 run 的 `training_curves.png` 都已用 `scripts/plot_training.py` 從各自的 `log.json` 產生。

---

## 8. 訓練流程細節

`train_one_epoch()`：
1. 每 step 依 `global_step` 算 lr scale，套用所有 param group。
2. `optimizer.zero_grad(set_to_none=True)`。
3. forward → `CEDiceLoss` → `scaler.scale(loss).backward()` → unscale → grad clip → `scaler.step` → `scaler.update`。
4. running loss / ce / dice 累積，依 `log_interval` 印出。

`evaluate()`（每 epoch 結束）：
- `ConfusionMeter` 累計 confusion matrix。
- `compute(ignore_index=0)`：per-class IoU/Dice/Precision/Recall + macro mIoU/mDice + pixel accuracy + 缺席類別清單 + confusion matrix。
- 額外算 `val_ce_loss`（不含 class weight、不含 dice）。

每 epoch 結束會：
- append history → `log.json`
- 寫 `last.pt`（含 model + optimizer state）
- 若 val mIoU 創新高則寫 `best.pt`（不含 optimizer）

---

## 9. 推論 (`predict_full.py`)

- 讀 `best.pt`：從 ckpt 內 `args.class_names` 動態切換 `CLASS_NAMES`、`NUM_CLASSES`、`CLASS_RGB`，無需 CLI 重複指定；同時依 ckpt 內 variant / freeze 設定重建 `SAM2SemSeg`，`load_state_dict(strict=False)`。
- **Sliding window**：`tile=512`、`stride=384`，**Gaussian window** (sigma = tile·0.125) 加權平均各 tile 的 softmax 機率，最後 argmax。
- **TTA**：`--tta_flip` 啟用 horizontal + vertical flip，logits 平均除 3。
- **AMP**：預設 fp16，`--no_amp` 關閉。
- **Craquelure 優先 post-processing**：`--craq_dilate N`（預設 0=關閉）。先 argmax 出 label，再以 N px 結構元素膨脹 craquelure mask，內部被判 crack 的像素改判 craquelure。對齊 mask 標註的 priority。`scipy.ndimage.binary_dilation` 為主，否則 fallback 到純 numpy 方形膨脹。
- **輸出**：
  - `label/`：單通道 png（值 = class id）
  - `color/`：palette 染色
  - `overlay/`：與原圖 alpha=0.5 疊圖
  - `prob/`：可選，每像素 softmax probabilities npy
- 若給 `--mask_dir`，會比對同名 GT 算 per-image / overall metrics 並寫 `per_image_metrics.csv` + `overall_metrics.json`。

---

## 10. 訓練曲線 (`scripts/plot_training.py`)

從 `output_dir/log.json` 讀 history，輸出 `output_dir/training_curves.png`，2x2 子圖：

1. train total / CE / Dice + val CE
2. val pixel accuracy
3. val macro mIoU / mDice
4. per-class IoU

標題自動標出 best mIoU 與對應 epoch。可在訓練中途隨時跑（會用到目前為止的 history）。

---

## 11. 重現指令

### 11.1 v1 5-class（原始 baseline）
```
# tile
python scripts/tile_pairs.py \
  --image_dir merged_4class_mask_semantic/images \
  --mask_dir  merged_4class_mask_semantic/masks \
  --out_dir   data/tiles_512 \
  --size 512 --stride 256 --bg_keep_ratio 0.15

# split
python scripts/make_group_split.py \
  --tiles_root data/tiles_512 --group_by stem --n_splits 4 --seed 42 \
  --out data/tiles_512/group_split_stem.json

# train
python train.py \
  --tiles_root data/tiles_512 \
  --split data/tiles_512/group_split_stem.json \
  --fold 0 --variant small --epochs 80 --batch_size 4 \
  --output_dir outputs/stem_fold0_small
```

### 11.2 v2 5-class
```
python scripts/tile_pairs.py \
  --image_dir merged_4class_mask_semantic_v2/images \
  --mask_dir  merged_4class_mask_semantic_v2/masks \
  --out_dir   data/tiles_512_v2 \
  --size 512 --stride 256 --bg_keep_ratio 0.15

python scripts/make_group_split.py \
  --tiles_root data/tiles_512_v2 --group_by stem --n_splits 4 --seed 42 \
  --out data/tiles_512_v2/group_split_stem.json

python train.py \
  --tiles_root data/tiles_512_v2 \
  --split data/tiles_512_v2/group_split_stem.json \
  --fold 0 --variant small --epochs 80 --batch_size 4 \
  --output_dir outputs/stem_fold0_small_v2
```

### 11.3 v2 3-class（bg / crack / craquelure，craquelure 重疊優先）
```
# 重映射
python scripts/remap_to_2class.py \
  --src_dir merged_4class_mask_semantic_v2 \
  --out_dir merged_2class_crack_craquelure_v2

# tile (num_classes=3)
python scripts/tile_pairs.py \
  --image_dir merged_2class_crack_craquelure_v2/images \
  --mask_dir  merged_2class_crack_craquelure_v2/masks \
  --out_dir   data/tiles_512_2class_v2 \
  --size 512 --stride 256 --num_classes 3 --bg_keep_ratio 0.15

python scripts/make_group_split.py \
  --tiles_root data/tiles_512_2class_v2 --group_by stem --n_splits 4 --seed 42 \
  --out data/tiles_512_2class_v2/group_split_stem.json

python train.py \
  --tiles_root data/tiles_512_2class_v2 \
  --split data/tiles_512_2class_v2/group_split_stem.json \
  --fold 0 --variant small --epochs 80 --batch_size 4 \
  --class_names "background,crack,craquelure" \
  --output_dir outputs/stem_fold0_small_2class_v2
```

### 11.4 推論（含 craquelure 優先 post-processing）
```
python predict_full.py \
  --ckpt outputs/stem_fold0_small_2class_v2/best.pt \
  --image /home/zzz90/research/crack_detection/data/image/KJTHT-SC-M-2RB1-4.jpg \
  --out_dir outputs/predict_KJTHT-SC-M-2RB1-4_2class_v2_craqprio8 \
  --tile 512 --stride 384 --tta_flip \
  --craq_dilate 8
```

### 11.5 畫訓練曲線
```
python scripts/plot_training.py --log_dir outputs/stem_fold0_small_2class_v2
# → outputs/stem_fold0_small_2class_v2/training_curves.png
```

---

## 12. 過去觀察與後續方向

- 三次 run 一致現象：crack 類 IoU < 0.07，明顯被「細結構 + class imbalance」雙重壓制；macro precision 低、recall 高，模型偏 over-prediction 前景。
- v1 → v2 mask 清過 craquelure 雜點後整體 mIoU 由 0.176 → 0.250；切到 3 類後 craquelure / crack 個別都比 v2 5-class 還高，但 macro 因失去 loss / shrinkage 兩個易類所以較低。
- 想再進一步可試：
  - 提高 dice_weight、降低 class_weight 的 crack 倍率，緩解 over-prediction。
  - `--no_freeze_trunk` 或 `--encoder_lr_mult 0.05` 解凍部分 Hiera blocks 微調。
  - `image_size=1024`（與 SAM2 pretrain 一致），不過顯存與 batch size 需調整。
  - inference 端調 `--craq_dilate`（典型 4–16）依視覺結果取捨。
