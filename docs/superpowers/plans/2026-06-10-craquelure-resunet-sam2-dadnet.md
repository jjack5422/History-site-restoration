# Craquelure 雙模型實驗 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 在 craquelure 二值分割上建立並訓練兩條路線——(A) ResUNet 粗 expert → SAM2 以 mask/點 prompt 精修;(B) 從零復現 DADNet——並用同一 holdout 比較 IoU/F1。

**Architecture:** 共用 Stage 0 craquelure 資料(前 95 張標註, 512 tile 給 A、224 chunk 給 B, n_splits=5 取 fold0 當 ~80/20 holdout)。A 兩階段離線傳 prompt(unet_env 產 ResUNet 機率圖 → sam2_env 訓 SAM2)。B 獨立 venv 自實作 DADNet。

**Tech Stack:** smp.Unet(resnet50) / unet_env;SAM2 PromptedSAM2Seg / sam2_env;DADNet(ConvNeXt-T + Neighborhood/Axial attention)/ dadnet_env;共用 `_lib/crackseg_common`。

參考 spec: `docs/superpowers/specs/2026-06-10-craquelure-resunet-sam2-dadnet-design.md`。

---

## Phase 0: 共用 craquelure 資料準備

### Task 0.1: 篩出前 95 張標註並建二值 tile 資料集 (512) + chunk 資料集 (224)

**Files:**
- Create: `_data/craq_0-94_v1/_seg95/` (95 個 SegmentationClass PNG 複本)
- Create: `_data/craq_0-94_v1/tiles_512/{images,masks,tile_index.json,group_split_stem.json}` (build_binary_datasets 產)
- Create: `_data/craq_0-94_v1/chunks_224/...` (同上, tile_size 224)
- Reuse: `crack_detection_sam2/scripts/build_binary_datasets.py`

- [ ] **Step 1: 建 95 張過濾後的 seg 目錄**

```bash
cd /home/zzz90/research
mkdir -p _data/craq_0-94_v1/_seg95
sam2_env/bin/python - <<'PY'
from pathlib import Path
import shutil
root = Path("_data/0-94test")
stems = [l.strip() for l in (root/"ImageSets/Segmentation/default.txt").read_text().splitlines() if l.strip()][:95]
dst = Path("_data/craq_0-94_v1/_seg95"); dst.mkdir(parents=True, exist_ok=True)
for s in stems:
    shutil.copy2(root/"SegmentationClass"/f"{s}.png", dst/f"{s}.png")
print("copied", len(list(dst.glob('*.png'))), "masks")
PY
```
Expected: `copied 95 masks`

- [ ] **Step 2: 建 512 tile 資料集 (craquelure, n_splits=5 → fold0 = 20% holdout)**

```bash
cd /home/zzz90/research/crack_detection_sam2
../unet_env/bin/python scripts/build_binary_datasets.py \
  --seg_dir /home/zzz90/research/_data/craq_0-94_v1/_seg95 \
  --image_dir /home/zzz90/research/_data/selected_slices/batch_1 \
  --out_root_template /home/zzz90/research/_data/craq_0-94_v1 \
  --classes craquelure --tile_size 512 --stride 256 --n_splits 5 --seed 42
```
Expected: 印出 summary(`kept_foreground`>0)+ fold0..4 統計;產生 `_data/craq_0-94_v1/tiles_512/`。

注意:`--out_root_template` 無 `{class}` 佔位 → out_root 即 `_data/craq_0-94_v1`(單類)。

- [ ] **Step 3: 建 224 chunk 資料集 (給 DADNet)**

```bash
cd /home/zzz90/research/crack_detection_sam2
../unet_env/bin/python scripts/build_binary_datasets.py \
  --seg_dir /home/zzz90/research/_data/craq_0-94_v1/_seg95 \
  --image_dir /home/zzz90/research/_data/selected_slices/batch_1 \
  --out_root_template /home/zzz90/research/_data/craq_0-94_v1 \
  --classes craquelure --tile_size 224 --stride 112 --n_splits 5 --seed 42
```
Expected: 產生 `_data/craq_0-94_v1/tiles_224/`(腳本以 `tiles_{tile_size}` 命名)。

- [ ] **Step 4: 驗證**

```bash
cd /home/zzz90/research
ls _data/craq_0-94_v1/tiles_512/images | wc -l
sam2_env/bin/python -c "import json;d=json.load(open('_data/craq_0-94_v1/tiles_512/group_split_stem.json'));f=d['folds'][0];print('val tiles',len(f['val']),'train tiles',len(f['train']),'val_groups',f['val_groups'])"
```
Expected: tile 數 > 0;fold0 val/train 比例約 20/80。

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add -A _data/craq_0-94_v1/tiles_512/tile_index.json _data/craq_0-94_v1/tiles_512/group_split_stem.json _data/craq_0-94_v1/tiles_224/tile_index.json _data/craq_0-94_v1/tiles_224/group_split_stem.json 2>/dev/null
git commit -m "data(craq): build craquelure binary tiles(512)+chunks(224) from first-95 of 0-94test" || echo "nothing to commit (tiles gitignored)"
```

---

## Phase A: ResUNet craquelure expert → SAM2 prompt 精修

### Task A1: 訓練 ResUNet craquelure expert

**Files:**
- Reuse: `crack_detection_unet/src/train.py`
- Output: `crack_detection_unet/runs/2026-06-10-craq-resunet50/`

- [ ] **Step 1: 啟動訓練 (背景)**

```bash
cd /home/zzz90/research/crack_detection_unet
../unet_env/bin/python src/train.py \
  --tiles_root /home/zzz90/research/_data/craq_0-94_v1/tiles_512 \
  --split /home/zzz90/research/_data/craq_0-94_v1/tiles_512/group_split_stem.json \
  --fold 0 --encoder resnet50 --class_names background,craquelure \
  --epochs 80 --batch_size 8 --base_lr 3e-4 \
  --output_dir runs/2026-06-10-craq-resunet50 2>&1 | tee runs/2026-06-10-craq-resunet50.log
```
Expected: 每 epoch 印 `[val] mIoU=...`;`best.pt` 隨 mIoU 改善寫出。

- [ ] **Step 2: 驗證收斂**

```bash
cd /home/zzz90/research/crack_detection_unet
../unet_env/bin/python -c "import json;h=json.load(open('runs/2026-06-10-craq-resunet50/log.json'))['history'];print('epochs',len(h));print('best val miou',max(e['val']['miou'] for e in h if e['val']['miou']==e['val']['miou']))"
```
Expected: best val mIoU 為有限值(>0)。

- [ ] **Step 3: Commit log**

```bash
cd /home/zzz90/research
git add crack_detection_unet/runs/2026-06-10-craq-resunet50/log.json crack_detection_unet/runs/2026-06-10-craq-resunet50/args.json
git commit -m "exp(craq): train ResUNet-50 craquelure expert (fold0 holdout)"
```

### Task A2: 離線產生 ResUNet craquelure 機率圖 (per-tile prompt)

**Files:**
- Reuse: `crack_detection_unet/src/predict_full.py`
- Output: `_data/craq_0-94_v1/tiles_512/resunet_prob/prob/{tile}.npy` (2,512,512)

- [ ] **Step 1: 對全部 512 tile 推論存 prob**

```bash
cd /home/zzz90/research/crack_detection_unet
../unet_env/bin/python src/predict_full.py \
  --ckpt runs/2026-06-10-craq-resunet50/best.pt \
  --image_dir /home/zzz90/research/_data/craq_0-94_v1/tiles_512/images \
  --tile 512 --stride 512 --save_prob \
  --out_dir /home/zzz90/research/_data/craq_0-94_v1/tiles_512/resunet_prob
```
Expected: `resunet_prob/prob/*.npy` 數量 = tile 數;每個 shape (2,512,512)。

- [ ] **Step 2: 驗證**

```bash
cd /home/zzz90/research
sam2_env/bin/python -c "import numpy as np,glob;fs=glob.glob('_data/craq_0-94_v1/tiles_512/resunet_prob/prob/*.npy');a=np.load(fs[0]);print('n',len(fs),'shape',a.shape,'craq_ch_range',float(a[1].min()),float(a[1].max()))"
```
Expected: shape (2,512,512);channel 1 (craquelure) 介於 0..1。

### Task A3: SAM2 prompt 精修訓練器 + 點取樣 (新程式, TDD)

**Files:**
- Create: `crack_detection_sam2/craq_prompt_sampling.py` (從 ResUNet mask 取點)
- Create: `crack_detection_sam2/train_craq_promptrefine.py` (訓練器, 兩模式)
- Create: `crack_detection_sam2/tests/test_craq_prompt_sampling.py`
- Reuse: `model_prompted_sam2.py::PromptedSAM2Seg`(forward(x, point_coords, point_labels, prev_mask=None))、`train_prompt.py` 的 loss/evaluate 模式、`_lib/crackseg_common`。

- [ ] **Step 1: 先寫點取樣的失敗測試**

```python
# tests/test_craq_prompt_sampling.py
import numpy as np, torch
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from craq_prompt_sampling import sample_points_from_prob

def test_returns_pos_and_neg_in_bounds():
    prob = np.zeros((512,512), np.float32); prob[100:150,100:150] = 0.9  # 一塊高信心 craq
    coords, labels = sample_points_from_prob(prob, n_pos=3, n_neg=3, thr=0.5, size=512, seed=0)
    assert coords.shape == (6,2) and labels.shape == (6,)
    assert ((coords>=0)&(coords<512)).all()
    assert (labels==1).sum()==3 and (labels==0).sum()==3
    # 正點落在高機率區
    for (y,x),l in zip(coords.astype(int), labels):
        if l==1: assert prob[y,x] >= 0.5
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `cd crack_detection_sam2 && ../sam2_env/bin/python -m pytest tests/test_craq_prompt_sampling.py -v`
Expected: FAIL(module/function 不存在)。

- [ ] **Step 3: 實作 `craq_prompt_sampling.py`**

```python
"""從 ResUNet craquelure 機率圖取正/負點 prompt(座標為 [x?] 注意:回傳 (y,x) 影像座標)。"""
import numpy as np

def sample_points_from_prob(prob, n_pos=3, n_neg=3, thr=0.5, size=512, seed=0):
    """prob: (H,W) craquelure 前景機率。回傳 coords (N,2) 影像座標(row=y,col=x)、labels (N,) 1=正 0=負。
    不足時以隨機點補(正點落在 argmax 區、負點落在低機率區)。"""
    rng = np.random.default_rng(seed)
    H, W = prob.shape
    pos_yx = np.argwhere(prob >= thr)
    neg_yx = np.argwhere(prob < thr)
    def pick(pool, k, fallback_val):
        if len(pool) >= k:
            idx = rng.choice(len(pool), size=k, replace=False); return pool[idx]
        # 不足:用整圖隨機點補
        extra = rng.integers(0, [H, W], size=(k - len(pool), 2))
        return np.concatenate([pool, extra], axis=0) if len(pool) else extra
    pos = pick(pos_yx, n_pos, 1).astype(np.float32)
    neg = pick(neg_yx, n_neg, 0).astype(np.float32)
    coords = np.concatenate([pos, neg], axis=0)
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)
    return coords, labels
```

- [ ] **Step 4: 跑測試確認通過**

Run: `cd crack_detection_sam2 && ../sam2_env/bin/python -m pytest tests/test_craq_prompt_sampling.py -v`
Expected: PASS。

- [ ] **Step 5: 確認 PromptedSAM2Seg forward 介面 + 座標慣例**

Run: `cd crack_detection_sam2 && ../sam2_env/bin/python -c "import inspect,model_prompted_sam2 as m;print(inspect.signature(m.PromptedSAM2Seg.forward));print(inspect.signature(m.PromptedSAM2Seg.decode))"`
Expected: 印出 `forward(self, x, point_coords, point_labels, prev_mask=None)`。
注意:SAM2 prompt_encoder 點座標為 **(x,y)** 順序;取樣回傳 (y,x) → 餵入前需 `coords[:, ::-1]` 轉成 (x,y)。在 train 腳本轉換。mask 模式時 point_coords 傳單一 padding 點 label=-1。

- [ ] **Step 6: 實作 `train_craq_promptrefine.py`**(骨架;細節依 train_prompt.py 既有 loss/評估)

```python
"""SAM2 以 ResUNet craquelure 機率圖為 prompt 精修。--prompt_mode {mask,points}。
Dataset 回傳 (image_tile, gt_mask, resunet_prob)。重用 crackseg_common 與 train_prompt 的 loss/metric 模式。"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, "/home/zzz90/research/_lib")
from model_prompted_sam2 import PromptedSAM2Seg
from craq_prompt_sampling import sample_points_from_prob
from crackseg_common.augment import IMAGENET_MEAN, IMAGENET_STD
from PIL import Image

class PromptTileDS(Dataset):
    def __init__(self, tiles_root, prob_dir, names):
        self.timg = Path(tiles_root)/"images"; self.tmsk = Path(tiles_root)/"masks"
        self.prob = Path(prob_dir)/"prob"; self.names = names
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        n = self.names[i]
        img = np.array(Image.open(self.timg/n).convert("RGB"))
        msk = (np.array(Image.open(self.tmsk/n)) > 0).astype(np.int64)
        prob = np.load(self.prob/(Path(n).stem+".npy"))[1]  # craquelure ch
        x = torch.from_numpy(img).float().div_(255).permute(2,0,1)
        m = torch.tensor(IMAGENET_MEAN).view(3,1,1); s = torch.tensor(IMAGENET_STD).view(3,1,1)
        return {"image": (x-m)/s, "mask": torch.from_numpy(msk),
                "prob": torch.from_numpy(prob).float(), "name": n}

def build_prompts(batch, mode, device, size=512):
    probs = batch["prob"]  # (B,H,W)
    B = probs.shape[0]
    if mode == "mask":
        pm = F.interpolate(probs.unsqueeze(1), size=(128,128), mode="bilinear", align_corners=False).to(device)
        # padding point label=-1
        coords = torch.zeros(B,1,2, device=device); labels = -torch.ones(B,1, dtype=torch.long, device=device)
        return coords, labels, pm
    else:  # points
        cs, ls = [], []
        for b in range(B):
            c, l = sample_points_from_prob(probs[b].numpy(), n_pos=3, n_neg=3, size=size, seed=b)
            c = c[:, ::-1].copy()  # (y,x)->(x,y)
            cs.append(torch.from_numpy(c).float()); ls.append(torch.from_numpy(l))
        return torch.stack(cs).to(device), torch.stack(ls).to(device), None

# main(): argparse(--tiles_root,--split,--fold,--prob_dir,--prompt_mode,--epochs,--output_dir...)
# 載 split fold0 train/val names(過濾存在於 images 的 tile);建 model = PromptedSAM2Seg("small", 512);
# loss: dice+bce(可直接照抄 train_prompt.py 的 BinaryCEDiceLoss);optimizer: AdamW(model.param_groups(base_lr));
# 迴圈: logits = model(img, *build_prompts(...)); loss; eval 用 (logits>0) vs gt 算 IoU/F1;存 best/last/metrics.json。
```
(實作時把 `train_prompt.py` 的 `BinaryCEDiceLoss` 與 `evaluate` 直接搬入或 import。)

- [ ] **Step 7: smoke (1 epoch, 限步數) 驗證可前向+反傳**

```bash
cd /home/zzz90/research/crack_detection_sam2
../sam2_env/bin/python train_craq_promptrefine.py \
  --tiles_root /home/zzz90/research/_data/craq_0-94_v1/tiles_512 \
  --split /home/zzz90/research/_data/craq_0-94_v1/tiles_512/group_split_stem.json --fold 0 \
  --prob_dir /home/zzz90/research/_data/craq_0-94_v1/tiles_512/resunet_prob \
  --prompt_mode mask --epochs 1 --batch_size 2 --output_dir runs/_smoke_mask
```
Expected: 跑完 1 epoch 無 shape 錯誤,印出 val IoU/F1。

- [ ] **Step 8: 正式訓練兩模式**

```bash
cd /home/zzz90/research/crack_detection_sam2
for MODE in mask points; do
../sam2_env/bin/python train_craq_promptrefine.py \
  --tiles_root /home/zzz90/research/_data/craq_0-94_v1/tiles_512 \
  --split /home/zzz90/research/_data/craq_0-94_v1/tiles_512/group_split_stem.json --fold 0 \
  --prob_dir /home/zzz90/research/_data/craq_0-94_v1/tiles_512/resunet_prob \
  --prompt_mode $MODE --epochs 60 --batch_size 4 \
  --output_dir runs/2026-06-10-craq-sam2prompt-$MODE 2>&1 | tee runs/2026-06-10-craq-sam2prompt-$MODE.log
done
```
Expected: 兩 run 各產 best.pt + metrics.json。

- [ ] **Step 9: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/craq_prompt_sampling.py crack_detection_sam2/train_craq_promptrefine.py crack_detection_sam2/tests/test_craq_prompt_sampling.py
git add crack_detection_sam2/runs/2026-06-10-craq-sam2prompt-*/metrics.json crack_detection_sam2/runs/2026-06-10-craq-sam2prompt-*/log.json 2>/dev/null
git commit -m "feat(craq): SAM2 prompt-refine trainer (mask+points) over ResUNet craquelure prob"
```

---

## Phase B: DADNet 復現

### Task B1: 建 dadnet_env 並驗證 natten/timm

**Files:** Create venv `dadnet_env`

- [ ] **Step 1: 建環境**

```bash
cd /home/zzz90/research
python3 -m venv dadnet_env
dadnet_env/bin/pip install --upgrade pip
dadnet_env/bin/pip install torch torchvision timm albumentations numpy pillow tqdm
dadnet_env/bin/pip install natten || echo "NATTEN_INSTALL_FAILED -> 用純 PyTorch 退路"
```

- [ ] **Step 2: smoke 驗證**

```bash
cd /home/zzz90/research
dadnet_env/bin/python - <<'PY'
import torch, timm
m = timm.create_model("convnext_tiny", features_only=True, pretrained=True)
x = torch.randn(1,3,224,224); fs = m(x)
print("convnext_tiny feature shapes:", [f.shape for f in fs])
try:
    import natten; print("natten OK", natten.__version__)
except Exception as e:
    print("natten missing -> fallback unfold NA:", e)
PY
```
Expected: 印出 4 層 feature shape;natten 可用或標記退路。

### Task B2: 實作 DADNet (新程式, TDD 形狀測試)

**Files:**
- Create: `crack_detection_dadnet/dadnet_model.py` (NeighborhoodAttention, AxialAttention, BiAxialBlock, DualAttention, DADNet)
- Create: `crack_detection_dadnet/tests/test_dadnet_shapes.py`

- [ ] **Step 1: 形狀失敗測試**

```python
# tests/test_dadnet_shapes.py
import torch, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dadnet_model import DADNet
def test_forward_shape():
    m = DADNet(num_classes=2, k=7, dilation=7).eval()
    y = m(torch.randn(2,3,224,224))
    assert y.shape == (2,2,224,224), y.shape
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `cd crack_detection_dadnet && ../dadnet_env/bin/python -m pytest tests/test_dadnet_shapes.py -v`
Expected: FAIL。

- [ ] **Step 3: 實作 `dadnet_model.py`**(ConvNeXt-T encoder + U-Net decoder;skip 經 DualAttention)

關鍵:
- `NeighborhoodAttention(dim, k=7)`:有 natten 用 `natten.functional.na2d`;否則純 PyTorch `F.unfold` 版(k×k 鄰域 QK^T softmax · V)。
- `AxialAttention(dim)`:對 H 軸與 W 軸各做一次 MHSA(把另一軸併入 batch),`BA(X)=X+row+col`。
- `BiAxialBlock(dim, dilation=7)`:RFB 式多分支(1×1、3×3、3×3 dilation d)+ AxialAttention 融合。
- `DualAttention(dim)`:`x + NA(x) + BAB(x)` 接到 skip。
- `DADNet`:timm `convnext_tiny` features_only → 4 stage features;decoder 逐層 upsample + concat(skip 過 DualAttention)→ head 出 num_classes。

- [ ] **Step 4: 跑測試確認通過**

Run: `cd crack_detection_dadnet && ../dadnet_env/bin/python -m pytest tests/test_dadnet_shapes.py -v`
Expected: PASS (輸出 (2,2,224,224))。

### Task B3: 訓練 DADNet craquelure

**Files:**
- Create: `crack_detection_dadnet/train_dadnet.py` (讀 tiles_224 + group_split fold0;Adam lr1e-4 batch16 CE;存 best/metrics)
- Output: `crack_detection_dadnet/runs/2026-06-10-craq-dadnet/`

- [ ] **Step 1: smoke (1 epoch)**

```bash
cd /home/zzz90/research/crack_detection_dadnet
../dadnet_env/bin/python train_dadnet.py \
  --tiles_root /home/zzz90/research/_data/craq_0-94_v1/tiles_224 \
  --split /home/zzz90/research/_data/craq_0-94_v1/tiles_224/group_split_stem.json --fold 0 \
  --epochs 1 --batch_size 8 --output_dir runs/_smoke
```
Expected: 1 epoch 無錯,印 val IoU/F1。

- [ ] **Step 2: 正式訓練**

```bash
cd /home/zzz90/research/crack_detection_dadnet
../dadnet_env/bin/python train_dadnet.py \
  --tiles_root /home/zzz90/research/_data/craq_0-94_v1/tiles_224 \
  --split /home/zzz90/research/_data/craq_0-94_v1/tiles_224/group_split_stem.json --fold 0 \
  --epochs 100 --batch_size 16 --lr 1e-4 \
  --output_dir runs/2026-06-10-craq-dadnet 2>&1 | tee runs/2026-06-10-craq-dadnet.log
```
Expected: best.pt + metrics.json;val craquelure IoU/F1。

- [ ] **Step 3: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_dadnet/*.py crack_detection_dadnet/tests/*.py crack_detection_dadnet/runs/2026-06-10-craq-dadnet/metrics.json crack_detection_dadnet/runs/2026-06-10-craq-dadnet/log.json
git commit -m "feat(craq): DADNet (ConvNeXt-T + neighborhood/axial attention) craquelure expert"
```

---

## Phase C: 對照

### Task C1: 彙整 craquelure 對照表

- [ ] **Step 1: 收集四個 val IoU/F1**(ResUNet / SAM2+mask / SAM2+points / DADNet),寫進 `crack_detection_sam2/EXPERIMENTS.md` 新一節「craquelure 多模型對照」。
- [ ] **Step 2: Commit。**

---

## Self-Review notes
- 涵蓋 spec 全部章節:Stage 0(Task 0.1)、A1/A2/A3(Task A1-A3)、B(B1-B3)、評估(C1)。
- 已查證重用:build_binary_datasets(n_splits=5+fold0=holdout)、train.py、predict_full.py(--save_prob per-tile)、PromptedSAM2Seg.forward(prev_mask/points)。
- 風險點已標:natten 安裝退路、SAM2 點座標 (x,y) 轉換、mask 模式 padding 點。
