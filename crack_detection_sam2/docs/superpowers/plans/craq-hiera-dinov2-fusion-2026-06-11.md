# Craquelure SAM2 Refine + DINOv2 語意注入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在現有 craquelure SAM2 prompt-refine pipeline 上注入凍結 DINOv2 語意特徵到 mask decoder 的 `feat`,驗證能否在守住 recall 下提升 precision/IoU。

**Architecture:** 不改 SAM2 大體架構。SAM2 image encoder (Hiera, 凍結) 產出 `feat=fpn[-1]` (32×32×256) 與 `high_res`;新增 `FeatFusionAdapter`(殘差 + zero-init)把離線 cache 的凍結 DINOv2 token map (384ch) 融進 `feat` → `feat'`,`high_res` 不動;ResUNet logit 照舊餵 prompt encoder。DINOv2 特徵離線 cache(比照 `resunet_prob`),訓練時不跑 DINOv2,避 8GB OOM。

**Tech Stack:** Python, PyTorch 2.11 (sam2_env), timm 1.0.27 (`vit_small_patch14_reg4_dinov2.lvd142m`, dim 384, 37×37 patches), SAM2 (`model_prompted_sam2.PromptedSAM2Seg`), pytest。所有 python 指令用 `/home/zzz90/research/sam2_env/bin/python`。

**Spec:** `crack_detection_sam2/docs/superpowers/specs/craq-hiera-dinov2-fusion-design-2026-06-11.md`

---

## File Structure

- Create `crack_detection_sam2/scripts/cache_dinov2_feats.py` — 離線把每個 tile 的 DINOv2 token map 存成 `[384,37,37]` fp16 `.npy`。
- Create `crack_detection_sam2/model_fused_sam2.py` — `FeatFusionAdapter` + `FusedPromptedSAM2Seg(PromptedSAM2Seg)`。
- Modify `crack_detection_sam2/train_craq_promptrefine.py` — 加 `--dino_feat_dir`/`--dino_dim`;dataset 選讀 dino;給定時用 fused model。未給定時行為與現狀逐位元相同(C0 公平)。
- Create `crack_detection_sam2/tests/test_fused_sam2_forward.py` — adapter zero-init identity + fused forward shape。
- Create `crack_detection_sam2/scripts/dump_fp_map.py` — Gate 0b:baseline best.pt 在 val 的 FP overlay。
- Create `crack_detection_sam2/scripts/eval_fused_sweep.py` — threshold sweep 後處理評估(baseline 與 fused 共用)。
- Create `crack_detection_sam2/scripts/run_craq_fused.sh` — cache + C0/E1 跨 5 fold 的執行腳本。

固定路徑常數(整份 plan 用)：
- TILES = `/home/zzz90/research/_data/craq_0-94_v1/tiles_512`
- SPLIT = `/home/zzz90/research/_data/craq_0-94_v1/tiles_512/group_split_stem.json`(5 folds, LOSO-by-stem)
- PROB = `/home/zzz90/research/_data/craq_0-94_v1/tiles_512/resunet_prob`
- DINO = `/home/zzz90/research/_data/craq_0-94_v1/tiles_512/dinov2_feat`
- PY = `/home/zzz90/research/sam2_env/bin/python`

---

## Task 1: DINOv2 feature cache 腳本 (Gate 0a)

**Files:**
- Create: `crack_detection_sam2/scripts/cache_dinov2_feats.py`

- [ ] **Step 1: 寫 cache 腳本**

```python
"""離線把 tiles_512 每張影像的 DINOv2 (reg4, S/14) patch token map cache 成 [C,37,37] fp16 .npy。
與 resunet_prob 平行;訓練時只讀此 cache,不跑 DINOv2。"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import timm
import torch
from PIL import Image

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="tiles_512/images 目錄")
    ap.add_argument("--out", required=True, help="輸出 dinov2_feat 目錄")
    ap.add_argument("--model", default="vit_small_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=518)  # 37*14
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = timm.create_model(args.model, pretrained=True, num_classes=0).eval().to(dev)
    npref = m.num_prefix_tokens
    g = args.size // 14
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(Path(args.images).glob("*.png")) + sorted(Path(args.images).glob("*.jpg"))
    print(f"model={args.model} prefix={npref} grid={g}x{g} n_imgs={len(imgs)}")
    done = 0
    for f in imgs:
        dst = out / (f.stem + ".npy")
        if dst.exists():
            continue
        im = Image.open(f).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
        x = torch.from_numpy(np.array(im)).float().div_(255).permute(2, 0, 1)
        x = ((x - MEAN) / STD).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.amp.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
            t = m.forward_features(x)            # [1, npref + g*g, C]
        t = t[:, npref:, :].float()             # [1, g*g, C]
        fm = t.reshape(1, g, g, -1).permute(0, 3, 1, 2).squeeze(0).cpu().numpy()  # [C,g,g]
        np.save(dst, fm.astype(np.float16))
        done += 1
        if done % 50 == 0:
            print(f"  {done} cached", flush=True)
    print(f"done. cached {done}, total existing {len(list(out.glob('*.npy')))}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Gate 0a — 在 sam2_env 跑 5 張 smoke 確認 DINOv2 載得起來且輸出形狀正確**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
PY=/home/zzz90/research/sam2_env/bin/python
mkdir -p /tmp/dino_smoke
$PY - <<'EOF'
import subprocess, glob, shutil, os
src="/home/zzz90/research/_data/craq_0-94_v1/tiles_512/images"
os.makedirs("/tmp/dino_smoke_imgs", exist_ok=True)
for f in sorted(glob.glob(src+"/*"))[:5]:
    shutil.copy(f, "/tmp/dino_smoke_imgs/")
print("copied 5")
EOF
$PY scripts/cache_dinov2_feats.py --images /tmp/dino_smoke_imgs --out /tmp/dino_smoke
$PY -c "import numpy as np, glob; a=np.load(glob.glob('/tmp/dino_smoke/*.npy')[0]); print('shape', a.shape, 'dtype', a.dtype)"
```
Expected: 印出 `model=... prefix=5 grid=37x37`,最後 `shape (384, 37, 37) dtype float16`。若 DINOv2 權重下載失敗或 import 出錯 → Gate 0a 未過,停下回報。

- [ ] **Step 3: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/scripts/cache_dinov2_feats.py
git commit -m "feat(craq): DINOv2 (timm reg4 S/14) feature cache script

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: FeatFusionAdapter (殘差 + zero-init)

**Files:**
- Create: `crack_detection_sam2/model_fused_sam2.py`
- Test: `crack_detection_sam2/tests/test_fused_sam2_forward.py`

- [ ] **Step 1: 寫 failing test(adapter zero-init = identity)**

```python
# tests/test_fused_sam2_forward.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_fused_sam2 import FeatFusionAdapter


def test_adapter_zero_init_is_identity():
    a = FeatFusionAdapter(sam_dim=256, dino_dim=384).eval()
    feat = torch.randn(2, 256, 32, 32)
    dino = torch.randn(2, 384, 37, 37)
    out = a(feat, dino)
    assert out.shape == feat.shape
    # zero-init 融合分支 -> delta=0 -> 輸出等於 feat
    assert torch.allclose(out, feat, atol=1e-6)
```

- [ ] **Step 2: 跑 test 確認 fail**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python -m pytest tests/test_fused_sam2_forward.py::test_adapter_zero_init_is_identity -v`
Expected: FAIL `ModuleNotFoundError: No module named 'model_fused_sam2'`

- [ ] **Step 3: 寫 `FeatFusionAdapter`(最小實作)**

```python
# model_fused_sam2.py
"""SAM2 prompt-refine + 凍結 DINOv2 語意注入。

FeatFusionAdapter 把離線 cache 的 DINOv2 token map 殘差注入 SAM2 的 feat (fpn[-1]);
zero-init 使初始 feat'==feat,訓練起點等於 baseline,只學增量。high_res 不經此模組。
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_prompted_sam2 import PromptedSAM2Seg


class FeatFusionAdapter(nn.Module):
    def __init__(self, sam_dim: int = 256, dino_dim: int = 384, hidden: int = 256):
        super().__init__()
        self.dino_proj = nn.Conv2d(dino_dim, hidden, kernel_size=1)
        self.fuse = nn.Conv2d(sam_dim + hidden, sam_dim, kernel_size=1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, feat: torch.Tensor, dino: torch.Tensor) -> torch.Tensor:
        # feat [B,sam_dim,h,w]; dino [B,dino_dim,gh,gw]
        d = self.dino_proj(dino)
        if d.shape[-2:] != feat.shape[-2:]:
            d = F.interpolate(d, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        delta = self.fuse(torch.cat([feat, d], dim=1))
        return feat + delta
```

- [ ] **Step 4: 跑 test 確認 pass**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python -m pytest tests/test_fused_sam2_forward.py::test_adapter_zero_init_is_identity -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/model_fused_sam2.py crack_detection_sam2/tests/test_fused_sam2_forward.py
git commit -m "feat(craq): FeatFusionAdapter (residual zero-init DINOv2->feat)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: FusedPromptedSAM2Seg

**Files:**
- Modify: `crack_detection_sam2/model_fused_sam2.py`
- Test: `crack_detection_sam2/tests/test_fused_sam2_forward.py`

- [ ] **Step 1: 加 failing test(forward shape + 初始融合 = identity)**

在 `tests/test_fused_sam2_forward.py` 末尾追加:

```python
def test_fused_forward_shape_and_init_identity():
    from model_fused_sam2 import FusedPromptedSAM2Seg
    model = FusedPromptedSAM2Seg(variant="small", image_size=512, device="cpu").eval()
    img = torch.randn(1, 3, 512, 512)
    dino = torch.randn(1, 384, 37, 37)
    coords = torch.zeros(1, 1, 2)
    labels = -torch.ones(1, 1, dtype=torch.long)
    out = model(img, dino, coords, labels, None)
    assert out.shape == (1, 1, 512, 512)
    # zero-init fusion: feat' == feat
    enc = model.encode_image(img)
    assert torch.allclose(model.fusion(enc["feat"], dino), enc["feat"], atol=1e-6)
```

- [ ] **Step 2: 跑 test 確認 fail**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python -m pytest tests/test_fused_sam2_forward.py::test_fused_forward_shape_and_init_identity -v`
Expected: FAIL `ImportError: cannot import name 'FusedPromptedSAM2Seg'`

- [ ] **Step 3: 在 `model_fused_sam2.py` 加 `FusedPromptedSAM2Seg`**

```python
class FusedPromptedSAM2Seg(PromptedSAM2Seg):
    """PromptedSAM2Seg + DINOv2 殘差注入 feat。forward 多收一個 dino_feat。
    DINOv2 本身不在此建構(離線 cache),只持有 adapter。"""

    def __init__(self, variant="small", image_size=512, dino_dim: int = 384,
                 mask_prompt_size=None, device: Optional[str] = None):
        super().__init__(variant=variant, image_size=image_size,
                         mask_prompt_size=mask_prompt_size, device=device)
        sam_dim = self.image_encoder.neck.d_model  # = fpn channel = 256
        self.fusion = FeatFusionAdapter(sam_dim=sam_dim, dino_dim=dino_dim)
        if device:
            self.fusion = self.fusion.to(device)

    def forward(self, x, dino_feat, point_coords, point_labels, prev_mask=None):
        enc = self.encode_image(x)
        enc["feat"] = self.fusion(enc["feat"], dino_feat)
        masks, _ = self.decode(enc, point_coords, point_labels, prev_mask)
        return masks

    def param_groups(self, base_lr, encoder_lr_mult=0.1):
        dec, enc = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "fusion" in n or "sam_mask_decoder" in n or "sam_prompt_encoder" in n:
                dec.append(p)
            else:
                enc.append(p)
        groups = [{"params": dec, "lr": base_lr, "name": "decoder"}]
        if enc:
            groups.append({"params": enc, "lr": base_lr * encoder_lr_mult, "name": "encoder"})
        return [g for g in groups if g["params"]]
```

- [ ] **Step 4: 跑 test 確認 pass**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python -m pytest tests/test_fused_sam2_forward.py -v`
Expected: 兩個 test 全 PASS

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/model_fused_sam2.py crack_detection_sam2/tests/test_fused_sam2_forward.py
git commit -m "feat(craq): FusedPromptedSAM2Seg (SAM2 refine + DINOv2 feat fusion)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Trainer 接上 dino(guarded,不破壞 C0)

**Files:**
- Modify: `crack_detection_sam2/train_craq_promptrefine.py`

- [ ] **Step 1: `PromptTileDS` 支援選讀 dino**

把 `PromptTileDS` 改成(`__init__` 加 `dino_dir=None`,flip 共用同一個隨機決定):

```python
class PromptTileDS(Dataset):
    def __init__(self, tiles_root, prob_dir, names, train=False, dino_dir=None):
        self.timg = Path(tiles_root) / "images"
        self.tmsk = Path(tiles_root) / "masks"
        self.prob = Path(prob_dir) / "prob"
        self.dino = Path(dino_dir) if dino_dir else None
        self.names = names
        self.train = train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        img = np.array(Image.open(self.timg / n).convert("RGB"))
        msk = (np.array(Image.open(self.tmsk / n)) > 0).astype(np.float32)
        prob = np.load(self.prob / (Path(n).stem + ".npy"))[1].astype(np.float32)  # craq channel
        dino = None
        if self.dino is not None:
            dino = np.load(self.dino / (Path(n).stem + ".npy")).astype(np.float32)  # [C,gh,gw]
        flip = self.train and np.random.rand() < 0.5
        if flip:
            img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy(); prob = prob[:, ::-1].copy()
            if dino is not None:
                dino = dino[..., ::-1].copy()  # flip width
        x = torch.from_numpy(img).float().div_(255).permute(2, 0, 1)
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        out = {"image": (x - m) / s, "mask": torch.from_numpy(msk),
               "prob": torch.from_numpy(prob), "name": n}
        if dino is not None:
            out["dino"] = torch.from_numpy(dino)
        return out
```

- [ ] **Step 2: 加 model-call helper + `exists_only` 選檢查 dino**

在 `evaluate` 之前加 helper,並改 `exists_only`:

```python
def run_model(model, img, dino, c, l, pm):
    return model(img, dino, c, l, pm) if dino is not None else model(img, c, l, pm)


def exists_only(tiles_root, prob_dir, names, dino_dir=None):
    img = Path(tiles_root) / "images"; prob = Path(prob_dir) / "prob"
    dino = Path(dino_dir) if dino_dir else None
    keep = []
    for n in names:
        if not ((img / n).exists() and (prob / (Path(n).stem + ".npy")).exists()):
            continue
        if dino is not None and not (dino / (Path(n).stem + ".npy")).exists():
            continue
        keep.append(n)
    return keep
```

- [ ] **Step 3: `evaluate` 使用 dino**

把 `evaluate` 內的 model 呼叫改成經 helper:

```python
@torch.no_grad()
def evaluate(model, loader, mode, device, mask_hw):
    model.eval()
    tp = fp = fn = 0
    for batch in loader:
        img = batch["image"].to(device)
        gt = (batch["mask"] > 0.5).to(device)
        dino = batch["dino"].to(device) if "dino" in batch else None
        c, l, pm = build_prompts(batch, mode, device, mask_hw)
        logits = run_model(model, img, dino, c, l, pm)
        pred = logits.squeeze(1) > 0
        tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum()); fn += int((~pred & gt).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {"craq_iou": iou, "craq_f1": f1, "precision": prec, "recall": rec}
```

- [ ] **Step 4: `main` 加 args + 條件建模 + 訓練迴圈用 dino**

在 argparse 區塊加(放在 `--output_dir` 之後):

```python
    ap.add_argument("--dino_feat_dir", default=None,
                    help="給定則啟用 DINOv2 fusion (FusedPromptedSAM2Seg);不給為 baseline")
    ap.add_argument("--dino_dim", type=int, default=384)
```

把 dataset/`exists_only`/model 建構改為:

```python
    tr_names, va_names = load_split(args.split, args.fold)
    tr_names = exists_only(args.tiles_root, args.prob_dir, tr_names, args.dino_feat_dir)
    va_names = exists_only(args.tiles_root, args.prob_dir, va_names, args.dino_feat_dir)
    print(f"mode={args.prompt_mode} train={len(tr_names)} val={len(va_names)} "
          f"dino={'on' if args.dino_feat_dir else 'off'}")

    tr = DataLoader(PromptTileDS(args.tiles_root, args.prob_dir, tr_names, train=True,
                                 dino_dir=args.dino_feat_dir),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, drop_last=True)
    va = DataLoader(PromptTileDS(args.tiles_root, args.prob_dir, va_names, train=False,
                                 dino_dir=args.dino_feat_dir),
                    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    pin_memory=True)

    if args.dino_feat_dir:
        from model_fused_sam2 import FusedPromptedSAM2Seg
        model = FusedPromptedSAM2Seg(variant=args.variant, image_size=args.image_size,
                                     dino_dim=args.dino_dim,
                                     mask_prompt_size=args.mask_prompt_size, device=device).to(device)
    else:
        model = PromptedSAM2Seg(variant=args.variant, image_size=args.image_size,
                                mask_prompt_size=args.mask_prompt_size, device=device).to(device)
```

把訓練迴圈內兩處 `model(img, c, l, pm)` 改為經 helper。先在迴圈頂端取 dino:

```python
        for it, batch in enumerate(tr):
            step = ep * len(tr) + it
            scale = (step / warmup) if step < warmup else 0.5 * (1 + math.cos(
                math.pi * (step - warmup) / max(1, total_steps - warmup)))
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * scale
            img = batch["image"].to(device)
            target = (batch["mask"] > 0.5).float().to(device)
            dino = batch["dino"].to(device) if "dino" in batch else None
            c, l, pm = build_prompts(batch, args.prompt_mode, device, mask_hw)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = run_model(model, img, dino, c, l, pm)
                    loss = dice_bce_loss(logits.float(), target,
                                         alpha=args.tversky_alpha, beta=args.tversky_beta)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                scaler.step(opt); scaler.update()
            else:
                logits = run_model(model, img, dino, c, l, pm)
                loss = dice_bce_loss(logits.float(), target)
                loss.backward(); opt.step()
            run += float(loss.detach()); nb += 1
```

- [ ] **Step 5: 回歸測試 — C0 路徑未變(dino off 時 smoke 1 epoch 不報錯)**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
PY=/home/zzz90/research/sam2_env/bin/python
$PY train_craq_promptrefine.py \
  --tiles_root /home/zzz90/research/_data/craq_0-94_v1/tiles_512 \
  --split /home/zzz90/research/_data/craq_0-94_v1/tiles_512/group_split_stem.json \
  --fold 4 --prob_dir /home/zzz90/research/_data/craq_0-94_v1/tiles_512/resunet_prob \
  --prompt_mode mask --tversky_alpha 0.2 --tversky_beta 0.8 \
  --epochs 1 --batch_size 2 --output_dir /tmp/c0_smoke 2>&1 | tail -5
```
Expected: 印出 `dino=off`,跑完 1 epoch 印 `ep1/1 ... craq_iou=...`,無 exception。

- [ ] **Step 6: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/train_craq_promptrefine.py
git commit -m "feat(craq): trainer optional --dino_feat_dir (fused model), C0 path unchanged

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Gate 0b — baseline FP map dump(質化確認 FP 來源)

**Files:**
- Create: `crack_detection_sam2/scripts/dump_fp_map.py`

- [ ] **Step 1: 寫 FP overlay 腳本**

```python
"""載入 baseline best.pt,在指定 fold 的 val tiles 上 dump FP(pred & ~gt)紅色 overlay,
目視確認 false positive 是否集中在彩繪/背景紋理。mask-prompt 模式。"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_prompted_sam2 import PromptedSAM2Seg

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--prob_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev)
    model = PromptedSAM2Seg(variant=ck["args"]["variant"], image_size=ck["args"]["image_size"],
                            mask_prompt_size=ck["args"].get("mask_prompt_size"), device=dev).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    mask_hw = tuple(model.sam_prompt_encoder.mask_input_size)

    va = json.load(open(args.split))["folds"][args.fold]["val"]
    timg = Path(args.tiles_root) / "images"; tmsk = Path(args.tiles_root) / "masks"
    prob_d = Path(args.prob_dir) / "prob"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    n = 0
    for name in va:
        if n >= args.n:
            break
        ip = timg / name
        pp = prob_d / (Path(name).stem + ".npy")
        if not (ip.exists() and pp.exists()):
            continue
        img = np.array(Image.open(ip).convert("RGB"))
        gt = (np.array(Image.open(tmsk / name)) > 0)
        prob = np.load(pp)[1].astype(np.float32)
        x = torch.from_numpy(img).float().div_(255).permute(2, 0, 1)
        x = ((x - MEAN) / STD).unsqueeze(0).to(dev)
        p = np.clip(prob, 1e-4, 1 - 1e-4)
        logit = torch.from_numpy(np.log(p / (1 - p)))[None, None].float()
        pm = F.interpolate(logit, size=mask_hw, mode="bilinear", align_corners=False).to(dev)
        coords = torch.zeros(1, 1, 2, device=dev); labels = -torch.ones(1, 1, dtype=torch.long, device=dev)
        with torch.no_grad():
            out_logits = model(x, coords, labels, pm)
        pred = (out_logits.squeeze().cpu().numpy() > 0)
        ov = img.copy()
        ov[pred & ~gt] = [255, 0, 0]      # FP 紅
        ov[pred & gt] = [0, 255, 0]       # TP 綠
        ov[~pred & gt] = [0, 0, 255]      # FN 藍
        blend = (0.5 * img + 0.5 * ov).astype(np.uint8)
        Image.fromarray(blend).save(out / f"{Path(name).stem}_fp.png")
        n += 1
    print(f"dumped {n} overlays to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Gate 0b — 對 baseline best.pt 跑 fold 0 dump**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
PY=/home/zzz90/research/sam2_env/bin/python
$PY scripts/dump_fp_map.py \
  --ckpt runs/craq-sam2prompt-tversky28-2026-06-10/best.pt \
  --tiles_root /home/zzz90/research/_data/craq_0-94_v1/tiles_512 \
  --split /home/zzz90/research/_data/craq_0-94_v1/tiles_512/group_split_stem.json \
  --fold 0 --prob_dir /home/zzz90/research/_data/craq_0-94_v1/tiles_512/resunet_prob \
  --out display/gate0_fp_fold0 --n 20
```
Expected: `dumped 20 overlays`。人工檢視 `display/gate0_fp_fold0/*.png`:紅色(FP)是否集中在彩繪/裝飾紋理區。把判斷(集中 / 分散)記到 commit message,作為假設前提的證據。

- [ ] **Step 3: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/scripts/dump_fp_map.py
git commit -m "feat(craq): Gate0 FP-map dump script + fold0 baseline observation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: 執行矩陣 — cache 特徵 + C0/E1 跨 5 fold

**Files:**
- Create: `crack_detection_sam2/scripts/run_craq_fused.sh`

實驗追蹤:用 experiment-tracking skill 的慣例,run 目錄命名日期放最後(依使用者偏好),如 `runs/craq-fused-e1-fold{k}-2026-06-11`、`runs/craq-base-c0-fold{k}-2026-06-11`。

- [ ] **Step 1: 全資料集 cache DINOv2 特徵**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
PY=/home/zzz90/research/sam2_env/bin/python
$PY scripts/cache_dinov2_feats.py \
  --images /home/zzz90/research/_data/craq_0-94_v1/tiles_512/images \
  --out /home/zzz90/research/_data/craq_0-94_v1/tiles_512/dinov2_feat
$PY -c "import glob; print('cached', len(glob.glob('/home/zzz90/research/_data/craq_0-94_v1/tiles_512/dinov2_feat/*.npy')))"
```
Expected: cached 數量 == tiles_512/images 數量(約 530)。

- [ ] **Step 2: 寫執行腳本(C0 baseline + E1 fused,各 5 fold)**

```bash
#!/usr/bin/env bash
# scripts/run_craq_fused.sh — C0 (baseline) 與 E1 (DINOv2 fused) 跨 5 fold
set -euo pipefail
PY=/home/zzz90/research/sam2_env/bin/python
ROOT=/home/zzz90/research/_data/craq_0-94_v1/tiles_512
SPLIT=$ROOT/group_split_stem.json
PROB=$ROOT/resunet_prob
DINO=$ROOT/dinov2_feat
DATE=2026-06-11
COMMON="--tiles_root $ROOT --split $SPLIT --prob_dir $PROB --prompt_mode mask \
  --tversky_alpha 0.2 --tversky_beta 0.8 --epochs 60 --batch_size 4 --base_lr 2e-4"

for k in 0 1 2 3 4; do
  echo "=== C0 baseline fold $k ==="
  $PY train_craq_promptrefine.py $COMMON --fold $k \
    --output_dir runs/craq-base-c0-fold${k}-${DATE}
  echo "=== E1 fused fold $k ==="
  $PY train_craq_promptrefine.py $COMMON --fold $k \
    --dino_feat_dir $DINO --dino_dim 384 \
    --output_dir runs/craq-fused-e1-fold${k}-${DATE}
done
```

- [ ] **Step 3: 跑單一 fold 驗證 E1 能訓練(fold 4,val 最小最快)**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
PY=/home/zzz90/research/sam2_env/bin/python
ROOT=/home/zzz90/research/_data/craq_0-94_v1/tiles_512
$PY train_craq_promptrefine.py \
  --tiles_root $ROOT --split $ROOT/group_split_stem.json --prob_dir $ROOT/resunet_prob \
  --prompt_mode mask --tversky_alpha 0.2 --tversky_beta 0.8 --fold 4 \
  --dino_feat_dir $ROOT/dinov2_feat --dino_dim 384 \
  --epochs 60 --batch_size 4 --base_lr 2e-4 \
  --output_dir runs/craq-fused-e1-fold4-2026-06-11 2>&1 | tail -8
```
Expected: 印 `dino=on`,60 epoch 跑完,印出 best craq_iou;`runs/craq-fused-e1-fold4-2026-06-11/metrics.json` 存在。若 OOM,降 `--batch_size 2`。

- [ ] **Step 4: 跑完整矩陣(背景執行)**

Run: `cd /home/zzz90/research/crack_detection_sam2 && bash scripts/run_craq_fused.sh 2>&1 | tee runs/run_fused_${USER}_2026-06-11.log`
Expected: 產生 `runs/craq-base-c0-fold{0..4}-2026-06-11/metrics.json` 與 `runs/craq-fused-e1-fold{0..4}-2026-06-11/metrics.json` 共 10 個。

- [ ] **Step 5: Commit 腳本(不 commit runs 權重)**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/scripts/run_craq_fused.sh
git commit -m "feat(craq): run script C0 baseline vs E1 DINOv2-fused across 5 folds

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: 聚合 + threshold sweep + 結論

**Files:**
- Create: `crack_detection_sam2/scripts/eval_fused_sweep.py`

- [ ] **Step 1: 寫跨-fold 聚合 + threshold sweep 腳本**

```python
"""聚合 C0 vs E1 的 5-fold metrics.json (mean±std),並對每個 best.pt 做 threshold sweep。
baseline 與 fused 共用:依 ckpt args 內有無 dino_feat_dir 自動切模型。"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_prompted_sam2 import PromptedSAM2Seg

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def agg(prefix):
    ious, ps, rs = [], [], []
    for k in range(5):
        mp = Path(f"runs/{prefix}-fold{k}-2026-06-11/metrics.json")
        if not mp.exists():
            print(f"  missing {mp}"); continue
        d = json.load(open(mp))
        ious.append(d["craq_iou"]); ps.append(d["precision"]); rs.append(d["recall"])
    f = lambda a: (float(np.mean(a)), float(np.std(a)))
    return {"n": len(ious), "iou": f(ious), "precision": f(ps), "recall": f(rs)}


def sweep(ckpt, tiles_root, split, fold, prob_dir, dino_dir, thresholds):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=dev)
    use_dino = bool(ck["args"].get("dino_feat_dir"))
    if use_dino:
        from model_fused_sam2 import FusedPromptedSAM2Seg
        model = FusedPromptedSAM2Seg(variant=ck["args"]["variant"], image_size=ck["args"]["image_size"],
                                     dino_dim=ck["args"].get("dino_dim", 384),
                                     mask_prompt_size=ck["args"].get("mask_prompt_size"), device=dev).to(dev)
    else:
        model = PromptedSAM2Seg(variant=ck["args"]["variant"], image_size=ck["args"]["image_size"],
                                mask_prompt_size=ck["args"].get("mask_prompt_size"), device=dev).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    mask_hw = tuple(model.sam_prompt_encoder.mask_input_size)
    va = json.load(open(split))["folds"][fold]["val"]
    timg = Path(tiles_root) / "images"; tmsk = Path(tiles_root) / "masks"; prob_d = Path(prob_dir) / "prob"
    dino_p = Path(dino_dir) if (use_dino and dino_dir) else None
    stats = {t: [0, 0, 0] for t in thresholds}  # tp,fp,fn
    for name in va:
        ip = timg / name; pp = prob_d / (Path(name).stem + ".npy")
        if not (ip.exists() and pp.exists()):
            continue
        img = np.array(Image.open(ip).convert("RGB"))
        gt = (np.array(Image.open(tmsk / name)) > 0)
        prob = np.load(pp)[1].astype(np.float32)
        x = ((torch.from_numpy(img).float().div_(255).permute(2, 0, 1) - MEAN) / STD).unsqueeze(0).to(dev)
        p = np.clip(prob, 1e-4, 1 - 1e-4)
        pm = F.interpolate(torch.from_numpy(np.log(p / (1 - p)))[None, None].float(),
                           size=mask_hw, mode="bilinear", align_corners=False).to(dev)
        coords = torch.zeros(1, 1, 2, device=dev); labels = -torch.ones(1, 1, dtype=torch.long, device=dev)
        with torch.no_grad():
            if use_dino:
                dino = torch.from_numpy(np.load(dino_p / (Path(name).stem + ".npy")).astype(np.float32))[None].to(dev)
                logits = model(x, dino, coords, labels, pm)
            else:
                logits = model(x, coords, labels, pm)
        prob_pred = torch.sigmoid(logits.squeeze()).cpu().numpy()
        for t in thresholds:
            pred = prob_pred > t
            stats[t][0] += int((pred & gt).sum()); stats[t][1] += int((pred & ~gt).sum()); stats[t][2] += int((~pred & gt).sum())
    res = {}
    for t, (tp, fp, fn) in stats.items():
        res[t] = {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--prob_dir", required=True)
    ap.add_argument("--dino_dir", required=True)
    args = ap.parse_args()
    print("== 5-fold aggregate (logit>0, train-time metric) ==")
    print("C0 baseline:", json.dumps(agg("craq-base-c0"), indent=2))
    print("E1 fused   :", json.dumps(agg("craq-fused-e1"), indent=2))
    thr = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
    for prefix in ["craq-base-c0", "craq-fused-e1"]:
        print(f"== threshold sweep {prefix} (per-fold) ==")
        for k in range(5):
            ck = Path(f"runs/{prefix}-fold{k}-2026-06-11/best.pt")
            if not ck.exists():
                print(f"  fold{k} missing"); continue
            r = sweep(str(ck), args.tiles_root, args.split, k, args.prob_dir, args.dino_dir, thr)
            best_t = max(r, key=lambda t: r[t]["iou"])
            print(f"  fold{k} best_thr={best_t} iou={r[best_t]['iou']:.4f} "
                  f"P={r[best_t]['precision']:.3f} R={r[best_t]['recall']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑聚合 + sweep**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
PY=/home/zzz90/research/sam2_env/bin/python
ROOT=/home/zzz90/research/_data/craq_0-94_v1/tiles_512
$PY scripts/eval_fused_sweep.py \
  --tiles_root $ROOT --split $ROOT/group_split_stem.json \
  --prob_dir $ROOT/resunet_prob --dino_dir $ROOT/dinov2_feat 2>&1 | tee runs/fused_compare_2026-06-11.txt
```
Expected: 印出 C0 與 E1 的 5-fold `iou/precision/recall` mean±std,及各 fold threshold sweep 的 best。

- [ ] **Step 3: 判讀 Gate 1 並寫結論**

判準(寫進 `runs/fused_compare_2026-06-11.txt` 末尾或 docs note):
- Gate 1 通過 = E1 的 5-fold mean IoU > C0,且 precision 提升、recall 未顯著下降(matched recall 下比較)。
- 與外部標竿對照:baseline fold0 IoU≈0.571(`craq-sam2prompt-tversky28-2026-06-10`)。
- 質化:對 E1 best.pt 重跑 `scripts/dump_fp_map.py`(需 fused 版 dump,見下 Step)對比 FP 是否在彩繪區下降。

若 Gate 1 通過 → 後續 spec 接「背景 pseudo-label aux class」。若未通過 → 記錄 DINOv2 注入無效的觀察(spec 假設證偽),不進一步。

- [ ] **Step 4: 更新 memory + commit 結論**

更新 `[[project_craquelure_dual_model]]`:加一行 E1 vs C0 的 5-fold 結果與 Gate 1 結論。

```bash
cd /home/zzz90/research
git add crack_detection_sam2/scripts/eval_fused_sweep.py crack_detection_sam2/runs/fused_compare_2026-06-11.txt
git commit -m "exp(craq): DINOv2-fused E1 vs C0 baseline 5-fold compare + threshold sweep

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review notes

- **Spec 覆蓋**:架構注入點(feat,high_res 不動)= Task 2/3;凍結策略 = Task 3(image_encoder 凍結繼承自 parent,DINOv2 不建構);cache = Task 1/6;C0/E1 矩陣 = Task 6;Gate 0a = Task 1 Step 2,Gate 0b = Task 5,Gate 1 = Task 7;loss/optimizer 沿用 = Task 4(未改 `dice_bce_loss`/optimizer);跨-fold mean±std = Task 7;threshold sweep = Task 7;質化 FP map = Task 5/7。E2(融合變體)為 spec 標「可選」,Gate 1 通過後才做,故不在必經 plan。
- **型別一致**:`FeatFusionAdapter(sam_dim, dino_dim, hidden)`、`FusedPromptedSAM2Seg(variant, image_size, dino_dim, mask_prompt_size, device)`、forward `(x, dino_feat, point_coords, point_labels, prev_mask)`、cache 形狀 `[384,37,37]` — 全 plan 一致。
- **DINOv2 dump for E1(Task 7 Step3 提到)**:目前 `dump_fp_map.py` 只支援 baseline;E1 質化 dump 需小幅擴充(加 `--dino_dir` 並走 fused 分支),屬 Gate 1 通過後的 follow-up,不阻擋主路徑。
