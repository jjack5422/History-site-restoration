# YOLO-prompted SAM2 decoder fine-tune（craquelure）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 訓練 SAM2 的 prompt_encoder+mask_decoder(encoder 凍結),用 GT 中軸點當 prompt 學 craquelure 分割,並以 oracle(GT 點)與 YOLO-point 兩模式 4-fold 評估,驗證「訓練過的 SAM2」能否讓 prompt 精修勝過 SepSAM YOLO-only(0.541)與 frozen-SAM2,並接近 dense-seg(0.634)。

**Architecture:** 改寫 `model_prompt_seg.py:SAM2PromptSeg` 成吃外部點 prompt 的 `PromptedSAM2Seg`;GT mask 中軸取點;BCE+Dice;4-fold(同 labeled32_craq_v3 split)。

**Tech Stack:** Python 3.12 / venv `/home/zzz90/research/sam2_env`、PyTorch、SAM2、skimage.medial_axis、Albumentations(既有 dataset/augment)。

> **環境前提:**
> - python: `/home/zzz90/research/sam2_env/bin/python`(簡稱 `$PY`)。cwd: `/home/zzz90/research/crack_detection_sam2`。
> - 非 git repo → commit 選用。
> - **已確認介面:** `model.build_sam2_model(variant, device, mode="train")`;`dataset.TileSegDataset` `__getitem__` 回 `{"image","mask"}`;`train_prompt.py:BinaryCEDiceLoss(ce_weight,dice_weight,pos_weight)`,`criterion(logits[B,1,H,W], target_idx[B,H,W]) -> (loss, parts)`;`dataset.compute_class_weights/load_tile_index/set_class_names`。`SAM2PromptSeg` 既有 forward 用 prompt_encoder(points=(coords,labels))+mask_decoder(multimask_output=False)。
> - 資料: `data/labeled32_craq_v3/tiles_512`(images/masks 0/1)+ `group_split_stem.json`(4 fold,`folds[k]` 有 `train`/`val` = png 檔名清單)。
> - 對照: dense-seg+CLAHE **0.634**、SepSAM YOLO-only **0.541**、frozen-SAM2 SAM-only ~0.10(皆排除 fold2)。fold2(A4-4 彩繪)預期崩。

---

## File Structure（皆在 `crack_detection_sam2/`）
- `model_prompted_sam2.py`(新)— `PromptedSAM2Seg`(外部點 prompt)+ `param_groups`。
- `gt_points.py`(新)— `gt_points(mask_bin, n_points)` 中軸取點(訓練/eval 共用)。
- `train_promptsam2_craq.py`(新)— 4-fold 訓練(GT 點、BCE+Dice、decoder+prompt_encoder 可訓)。
- `cache_yolo_points.py`(新,**在 SepSAM2 專案、用 SepSAM2 env 跑**)— 用 fold-k craq YOLO agent 對 val tile 取點,存 `craqfold{k}/yolo_points.json`。
- `eval_promptsam2_craq.py`(新)— oracle + YOLO-point 雙模式 4-fold eval + 對照表。
- `tests/test_gt_points.py`、`tests/test_prompted_sam2_forward.py`(新)。

---

## Task 1: gt_points 中軸取點 helper

**Files:** Create `gt_points.py`, `tests/test_gt_points.py`

- [ ] **Step 1: 失敗測試** — `tests/test_gt_points.py`:
```python
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gt_points import gt_points


def test_empty_mask_returns_empty():
    m = np.zeros((64, 64), np.uint8)
    pts, labs = gt_points(m, 10)
    assert pts.shape == (0, 2) and labs.shape == (0,)


def test_points_lie_on_foreground_and_positive():
    m = np.zeros((64, 64), np.uint8)
    m[30:34, 5:60] = 1  # 一條水平帶
    pts, labs = gt_points(m, 8)
    assert 1 <= pts.shape[0] <= 8
    assert (labs == 1).all()
    for x, y in pts.astype(int):
        assert m[y, x] == 1  # (x,y) 順序,落在前景


if __name__ == "__main__":
    test_empty_mask_returns_empty()
    test_points_lie_on_foreground_and_positive()
    print("OK test_gt_points")
```

- [ ] **Step 2: 跑→失敗** `$PY tests/test_gt_points.py` → ModuleNotFoundError: gt_points

- [ ] **Step 3: 實作 `gt_points.py`:**
```python
"""gt_points.py — 從二值 mask 沿中軸均勻取正點,給 SAM2 當 point prompt。"""
import numpy as np
from skimage.morphology import medial_axis


def gt_points(mask_bin, n_points):
    """Args: mask_bin HxW(非零=前景), n_points 目標點數。
    Returns: pts (k,2) float32 每列 (x,y) 像素座標; labels (k,) int64 全 1。空 mask→空。"""
    mask_bin = np.asarray(mask_bin).astype(bool)
    if mask_bin.sum() == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.int64)
    skel = medial_axis(mask_bin)
    ys, xs = np.where(skel)
    if xs.size == 0:                      # 極細區骨架可能空 → 退回用前景像素
        ys, xs = np.where(mask_bin)
    k = int(min(max(n_points, 1), xs.size))
    sel = np.linspace(0, xs.size - 1, num=k).astype(int)
    pts = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float32)
    labels = np.ones(k, dtype=np.int64)
    return pts, labels
```

- [ ] **Step 4: 跑→通過** `$PY tests/test_gt_points.py` → `OK test_gt_points`
- [ ] **Step 5(選用) commit**

---

## Task 2: PromptedSAM2Seg 模型

**Files:** Create `model_prompted_sam2.py`, `tests/test_prompted_sam2_forward.py`

- [ ] **Step 1: 失敗測試(forward 形狀 + 可微)** — `tests/test_prompted_sam2_forward.py`:
```python
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_prompted_sam2 import PromptedSAM2Seg


def test_forward_shape_and_grad():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = PromptedSAM2Seg(variant="small", image_size=512, device=dev).to(dev)
    x = torch.randn(2, 3, 512, 512, device=dev)
    coords = torch.tensor([[[100., 100.], [200., 200.]],
                           [[150., 150.], [0., 0.]]], device=dev)   # 第2張第2點為 pad
    labels = torch.tensor([[1, 1], [1, -1]], device=dev)            # -1 = padding
    y = m(x, coords, labels)
    assert y.shape == (2, 1, 512, 512), y.shape
    # decoder 參數應可訓、encoder 凍結
    tr = [n for n, p in m.named_parameters() if p.requires_grad]
    assert any("sam_mask_decoder" in n for n in tr)
    assert not any("image_encoder" in n for n in tr)


if __name__ == "__main__":
    test_forward_shape_and_grad()
    print("OK test_prompted_sam2_forward")
```

- [ ] **Step 2: 跑→失敗** `$PY tests/test_prompted_sam2_forward.py` → ModuleNotFoundError

- [ ] **Step 3: 實作 `model_prompted_sam2.py`**(以 `model_prompt_seg.py:SAM2PromptSeg` 為本,prompt 改外部輸入):
```python
"""model_prompted_sam2.py — SAM2(image_encoder 凍結)+ prompt_encoder + mask_decoder,
prompt 點為 forward 的外部輸入(非 learnable)。輸出 [B,1,H,W] binary logits。"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import build_sam2_model


class PromptedSAM2Seg(nn.Module):
    def __init__(self, variant="small", image_size=512,
                 freeze_image_encoder=True, freeze_prompt_encoder=False,
                 device: Optional[str] = None):
        super().__init__()
        sam2 = build_sam2_model(variant=variant, device=device, mode="train")
        embed = image_size // sam2.backbone_stride
        sam2.image_size = image_size
        sam2.sam_image_embedding_size = embed
        sam2.sam_prompt_encoder.input_image_size = (image_size, image_size)
        sam2.sam_prompt_encoder.image_embedding_size = (embed, embed)
        sam2.sam_prompt_encoder.mask_input_size = (4 * embed, 4 * embed)
        self.image_encoder = sam2.image_encoder
        self.sam_prompt_encoder = sam2.sam_prompt_encoder
        self.sam_mask_decoder = sam2.sam_mask_decoder
        self.use_high_res = sam2.use_high_res_features_in_sam
        del sam2
        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        if freeze_prompt_encoder:
            for p in self.sam_prompt_encoder.parameters():
                p.requires_grad = False

    def forward(self, x, point_coords, point_labels):
        B, _, H, W = x.shape
        bb = self.image_encoder(x)
        fpn = bb["backbone_fpn"]
        high_res = ([self.sam_mask_decoder.conv_s0(fpn[0]),
                     self.sam_mask_decoder.conv_s1(fpn[1])] if self.use_high_res else None)
        feat = fpn[-1]
        sparse, dense = self.sam_prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=None)
        low, _, _, _ = self.sam_mask_decoder(
            image_embeddings=feat,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=False, high_res_features=high_res)
        return F.interpolate(low.float(), size=(H, W), mode="bilinear", align_corners=False)

    def param_groups(self, base_lr, encoder_lr_mult=0.01):
        dec, enc = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (dec if ("sam_mask_decoder" in n or "sam_prompt_encoder" in n) else enc).append(p)
        groups = [{"params": dec, "lr": base_lr, "name": "decoder"}]
        if enc:
            groups.append({"params": enc, "lr": base_lr * encoder_lr_mult, "name": "encoder"})
        return [g for g in groups if g["params"]]
```

- [ ] **Step 4: 跑→通過(需 GPU + SAM2 small ckpt)** `$PY tests/test_prompted_sam2_forward.py` → `OK test_prompted_sam2_forward`
  - 若報 ckpt 缺失/SAM2 載入錯,報 BLOCKED 附 traceback(不要亂改)。
- [ ] **Step 5(選用) commit**

---

## Task 3: 訓練 4-fold（GT 點）

**Files:** Create `train_promptsam2_craq.py`

- [ ] **Step 1: 實作 `train_promptsam2_craq.py`**(以 `train_prompt.py` 為骨架,改用 PromptedSAM2Seg + GT 點 prompt):
```python
"""train_promptsam2_craq.py — 4-fold 訓練 PromptedSAM2Seg(craquelure),GT 中軸點當 prompt。"""
import argparse, json, math, os, time
import numpy as np, torch
from torch.utils.data import DataLoader
from augment import train_transforms, val_transforms
import dataset as _dataset
from dataset import TileSegDataset, compute_class_weights, load_tile_index, set_class_names
from train_prompt import BinaryCEDiceLoss
from model_prompted_sam2 import PromptedSAM2Seg
from gt_points import gt_points

TILES = "data/labeled32_craq_v3/tiles_512"
SPLIT = "data/labeled32_craq_v3/tiles_512/group_split_stem.json"


def build_prompts(masks, n_points, image_size, device):
    """masks [B,H,W] tensor(0/1)→ (coords [B,maxk,2], labels [B,maxk]); pad label=-1。
    空樣本退回影像中心單一正點。"""
    per = []
    arr = masks.detach().cpu().numpy()
    for i in range(arr.shape[0]):
        pts, labs = gt_points(arr[i] > 0, n_points)
        if pts.shape[0] == 0:
            pts = np.array([[image_size / 2, image_size / 2]], np.float32)
            labs = np.ones(1, np.int64)
        per.append((pts, labs))
    maxk = max(p[0].shape[0] for p in per)
    B = len(per)
    coords = np.zeros((B, maxk, 2), np.float32)
    labels = -np.ones((B, maxk), np.int64)
    for i, (pts, labs) in enumerate(per):
        k = pts.shape[0]; coords[i, :k] = pts; labels[i, :k] = labs
    return (torch.from_numpy(coords).to(device), torch.from_numpy(labels).to(device))


def cosine_with_warmup(step, total, warm):
    if step < warm:
        return step / max(1, warm)
    prog = (step - warm) / max(1, total - warm)
    return 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, loader, n_points, image_size, device):
    model.eval(); tp = fp = fn = 0
    for batch in loader:
        img = batch["image"].to(device); msk = batch["mask"].to(device)
        coords, labels = build_prompts(msk, n_points, image_size, device)
        logits = model(img, coords, labels)
        pred = (logits.squeeze(1) > 0).long(); gt = (msk > 0).long()
        tp += ((pred == 1) & (gt == 1)).sum().item()
        fp += ((pred == 1) & (gt == 0)).sum().item()
        fn += ((pred == 0) & (gt == 1)).sum().item()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"iou": tp / max(tp + fp + fn, 1), "f1": 2 * p * r / max(p + r, 1e-8),
            "precision": p, "recall": r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--variant", default="small")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--n_points", type=int, default=10)  # ~512//50
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--base_lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_prefix", default="outputs/promptsam2_craq")
    args = ap.parse_args()
    set_class_names(["background", "craquelure"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)

    tile_index = load_tile_index(TILES)
    by_name = {it["tile"]: it for it in tile_index["items"]}
    payload = json.load(open(SPLIT))
    msk_dir = os.path.join(TILES, "masks")

    def has_fg(tile_name):  # 只訓前景 tile
        import cv2
        stem = os.path.splitext(tile_name)[0]
        m = cv2.imread(os.path.join(msk_dir, stem + ".png"), 0)
        return m is not None and (m > 0).any()

    for k in args.folds:
        fd = payload["folds"][k]
        tr_items = [by_name[n] for n in fd["train"] if n in by_name and has_fg(n)]
        va_items = [by_name[n] for n in fd["val"] if n in by_name]
        print(f"fold{k}: train_fg={len(tr_items)} val={len(va_items)}", flush=True)
        tr_ds = TileSegDataset(TILES, tr_items, transforms=train_transforms(image_size=args.image_size))
        va_ds = TileSegDataset(TILES, va_items, transforms=val_transforms(image_size=args.image_size))
        tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        cw, counts = compute_class_weights(tr_items, TILES, num_classes=_dataset.NUM_CLASSES, mode="median_freq")
        pos_w = min(float(counts[0]) / max(float(counts[1]), 1), 100.0) if len(counts) >= 2 else None
        model = PromptedSAM2Seg(variant=args.variant, image_size=args.image_size, device=device).to(device)
        groups = model.param_groups(args.base_lr); base_lrs = [g["lr"] for g in groups]
        opt = torch.optim.AdamW(groups, lr=args.base_lr, weight_decay=1e-4)
        crit = BinaryCEDiceLoss(0.5, 0.5, pos_weight=pos_w).to(device)
        scaler = torch.amp.GradScaler("cuda")
        total = max(1, args.epochs * len(tr)); warm = max(1, args.warmup_epochs * len(tr))
        out = f"{args.out_prefix}_fold{k}"; os.makedirs(out, exist_ok=True)
        best = -1; log = {"history": []}
        for ep in range(args.epochs):
            model.train(); t0 = time.time(); run = 0; nb = 0
            for it, batch in enumerate(tr):
                gs = ep * len(tr) + it; sc = cosine_with_warmup(gs, total, warm)
                for g, lr in zip(opt.param_groups, base_lrs): g["lr"] = lr * sc
                img = batch["image"].to(device); msk = batch["mask"].to(device)
                coords, labels = build_prompts(msk, args.n_points, args.image_size, device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(img, coords, labels)
                    loss, _ = crit(logits, msk)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                scaler.step(opt); scaler.update()
                run += float(loss.detach()); nb += 1
            ev = evaluate(model, va, args.n_points, args.image_size, device)
            print(f"fold{k} ep{ep+1}/{args.epochs} loss={run/max(nb,1):.4f} "
                  f"valF1={ev['f1']:.4f} IoU={ev['iou']:.4f} {(time.time()-t0):.1f}s", flush=True)
            log["history"].append({"epoch": ep + 1, "val": ev})
            json.dump(log, open(os.path.join(out, "log.json"), "w"))
            if ev["iou"] > best:
                best = ev["iou"]
                torch.save({"model": model.state_dict(), "epoch": ep + 1, "val": ev,
                            "args": vars(args)}, os.path.join(out, "best.pt"))
        print(f"fold{k} done best_iou={best:.4f}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 單 fold smoke(2 epochs)確認能跑、loss 下降**
Run: `$PY train_promptsam2_craq.py --folds 0 --epochs 2 --batch_size 4`
Expected: 印出 `fold0: train_fg=... val=...`、兩個 epoch 的 loss/valF1 行,無錯;產生 `outputs/promptsam2_craq_fold0/best.pt`。若 OOM,報告(可降 batch_size=2)。

- [ ] **Step 3: 背景跑完整 4-fold(coordinator)**
Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
nohup /home/zzz90/research/sam2_env/bin/python train_promptsam2_craq.py \
  --folds 0 1 2 3 --epochs 80 --batch_size 4 > outputs/promptsam2_craq.log 2>&1 &
```
Expected: `outputs/promptsam2_craq_fold{0..3}/best.pt`。
- [ ] **Step 4(選用) commit**

---

## Task 4: 快取 YOLO 點（SepSAM2 env）

**Files:** Create `/home/zzz90/research/crack_detection_SepSAM2/sepsam/scripts/cache_yolo_points.py`

- [ ] **Step 1: 實作 cache_yolo_points.py**(用 SepSAM2 craq agent 對 fold val 取點存 json):
```python
"""cache_yolo_points.py — 用 fold-k craquelure YOLO agent 對該 fold val tile 取中軸點,
存 <tiles>/craqfold{k}/yolo_points.json: {stem: [[x,y],...]}(供 crack_detection_sam2 的 eval 載入)。"""
import argparse, glob, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import Agent
from src.geometry import mask_to_points_and_width

TILES = "/home/zzz90/research/crack_detection_sam2/data/labeled32_craq_v3/tiles_512"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--points_divisor", type=int, default=50)
    args = ap.parse_args()
    from ultralytics.nn.tasks import C2f_SEA  # noqa
    for k in args.folds:
        ck = f"runs/segment/runs/sepsam_agent_craq_cv_fold{k}/weights/best.pt"
        agent = Agent(ck, device="cuda")
        vi = os.path.join(TILES, f"craqfold{k}", "val_images")
        out = {}
        for p in sorted(glob.glob(os.path.join(vi, "*.png"))):
            st = os.path.splitext(os.path.basename(p))[0]
            bgr = cv2.imread(p); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask, _ = agent.predict(rgb, conf=args.conf, iou=0.5)
            n = max(rgb.shape[:2]) // args.points_divisor
            pts, _ = mask_to_points_and_width(mask > 0, n)
            out[st] = pts.tolist()
        dst = os.path.join(TILES, f"craqfold{k}", "yolo_points.json")
        json.dump(out, open(dst, "w"))
        print(f"fold{k}: {len(out)} tiles -> {dst}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑(SepSAM2 env)**
Run:
```bash
cd /home/zzz90/research/crack_detection_SepSAM2/sepsam
/home/zzz90/research/SepSAM2_env/bin/python scripts/cache_yolo_points.py --folds 0 1 2 3 --conf 0.05
```
Expected: 4 行,各 fold 產生 `craqfold{k}/yolo_points.json`。
- [ ] **Step 3(選用) commit**

---

## Task 5: 雙模式 4-fold 評估

**Files:** Create `eval_promptsam2_craq.py`(在 crack_detection_sam2)

- [ ] **Step 1: 實作 `eval_promptsam2_craq.py`:**
```python
"""eval_promptsam2_craq.py — oracle(GT 點)+ YOLO-point(快取)兩模式,逐 fold 評 PromptedSAM2Seg。"""
import argparse, glob, json, os
import cv2, numpy as np, torch
from model_prompted_sam2 import PromptedSAM2Seg
from gt_points import gt_points

TILES = "data/labeled32_craq_v3/tiles_512"
DENSE = {0: 0.634, 1: 0.614, 2: 0.040, 3: 0.655}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def prf(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    f1 = 2 * pr * rc / max(pr + rc, 1e-8); iou = tp / max(tp + fp + fn, 1)
    if (tp + fp + fn) == 0:  # GT 與 pred 皆空
        return 1.0, 1.0, 1.0, 1.0
    return pr, rc, f1, iou


def load_img(path, image_size):
    bgr = cv2.imread(path); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (image_size, image_size))
    t = ((r / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    return torch.from_numpy(t[None].astype(np.float32)), rgb.shape[:2]


@torch.no_grad()
def predict(model, img_t, pts, image_size, device, orig_hw):
    h, w = orig_hw
    if len(pts) == 0:
        return np.zeros((h, w), np.uint8)
    sx, sy = image_size / w, image_size / h
    pc = np.array([[x * sx, y * sy] for x, y in pts], np.float32)[None]
    pl = np.ones((1, pc.shape[1]), np.int64)
    logits = model(img_t.to(device), torch.from_numpy(pc).to(device), torch.from_numpy(pl).to(device))
    m = (logits.squeeze().cpu().numpy() > 0).astype(np.uint8)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def eval_fold(k, mode, image_size, n_points, device):
    model = PromptedSAM2Seg(variant="small", image_size=image_size, device=device).to(device)
    ck = torch.load(f"outputs/promptsam2_craq_fold{k}/best.pt", map_location=device)
    model.load_state_dict(ck["model"]); model.eval()
    vi = os.path.join(TILES, f"craqfold{k}", "val_images")
    vm = os.path.join(TILES, f"craqfold{k}", "val_masks")
    yp = json.load(open(os.path.join(TILES, f"craqfold{k}", "yolo_points.json"))) if mode == "yolo" else None
    recs = []
    for p in sorted(glob.glob(os.path.join(vi, "*.png"))):
        st = os.path.splitext(os.path.basename(p))[0]
        gt = cv2.imread(os.path.join(vm, st + ".png"), 0)
        img_t, hw = load_img(p, image_size)
        if mode == "oracle":
            pts, _ = gt_points(gt > 0, n_points); pts = pts.tolist()
        else:
            pts = yp.get(st, [])
        pred = predict(model, img_t, pts, image_size, device, hw)
        recs.append(prf(pred, gt > 0))
    a = np.array(recs, float).mean(0)
    return {"P": a[0], "R": a[1], "F1": a[2], "IoU": a[3], "n": len(recs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--n_points", type=int, default=10)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for mode in ["oracle", "yolo"]:
        print(f"\n==== mode={mode} ====")
        rows = []
        for k in args.folds:
            r = eval_fold(k, mode, args.image_size, args.n_points, device)
            rows.append((k, r))
            print(f"fold{k} n={r['n']} F1={r['F1']:.3f} P={r['P']:.3f} R={r['R']:.3f} IoU={r['IoU']:.3f} | dense-seg={DENSE[k]}")
        no2 = [r for kk, r in rows if kk != 2]
        m4 = sum(r["F1"] for _, r in rows) / len(rows)
        mno2 = sum(r["F1"] for r in no2) / max(len(no2), 1)
        print(f"mean F1 4-fold={m4:.3f}  排除fold2={mno2:.3f}  (dense-seg 0.634 / SepSAM-YOLO 0.541)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑評估**
Run: `$PY eval_promptsam2_craq.py --folds 0 1 2 3`
Expected: oracle 與 yolo 兩段,各 fold F1 + 排除-fold2 平均。

- [ ] **Step 3: 寫總結 `outputs/promptsam2_craq_summary.md`,判定:**
  1. **oracle** 排除-fold2 F1 vs dense-seg 0.634(decoder 學習上限);
  2. **yolo-point** 排除-fold2 F1 vs SepSAM YOLO-only 0.541 與 frozen-SAM2(訓練 SAM2 是否讓 refine 有用);
  3. oracle 與 yolo-point 的差 = YOLO recall 天花板代價。
- [ ] **Step 4(選用) commit**

---

## Self-Review 註記
- Spec §3 架構→Task 2;§4 GT 點/空-mask→Task 1+Task 3(`has_fg` 過濾 + 空樣本中心點 fallback);§5 訓練→Task 3;§6 雙模式 eval→Task 4(YOLO 點快取)+Task 5;§7 元件全列。
- 一致性:`PromptedSAM2Seg(variant,image_size,device)` / `gt_points(mask,n)` / `craqfold{k}/yolo_points.json` / `outputs/promptsam2_craq_fold{k}/best.pt` 跨 Task 一致。
- 跨 env:Task 4 在 SepSAM2 env 產 YOLO 點 json,Task 5 在 sam2_env 載入 → 不在同進程混 env。
- 空-mask:訓練 `has_fg` 過濾 + aug 後空樣本給中心正點;eval 無點→全空 pred,`prf` 對「皆空」回 1.0。
- 已知坑:SAM2 prompt encoder 點座標為 [0,image_size] 像素、label -1=pad(測試已涵蓋);eval 對非 512 原圖做縮放並把點座標等比例縮放。
