# Click-promptable SAM2 experts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train two SAM2 experts (craquelure, crack) that turn a user's positive/negative click points on an image into a binary mask, SAM-style, fine-tuned on the heritage tiles.

**Architecture:** Reuse the existing `PromptedSAM2Seg` (frozen SAM2 image encoder + trainable prompt encoder + mask decoder, real external point prompts). Add a `prev_mask` dense-prompt input and an `encode_image`/`decode` split so iterative clicks reuse the frozen backbone. Train with SAM-style iterative click simulation (sample a positive seed, then correction points from the error region), evaluate with interactive metrics (IoU@k clicks, NoC@0.8). Ship a `predict_click` inference helper for the later CVAT Nuclio interactor (sub-project 2).

**Tech Stack:** PyTorch 2.11 cu128 (`/home/zzz90/research/sam2_env`), vendored SAM-2, numpy, scipy, albumentations. No system `python` — always call `/home/zzz90/research/sam2_env/bin/python`. No pytest in the env: tests are plain scripts with a `__main__` runner that prints `OK`.

**Spec:** `docs/superpowers/specs/2026-06-09-click-promptable-sam2-experts-design.md`

**Conventions:**
- Repo root for code: `/home/zzz90/research/crack_detection_sam2` (run commands from here).
- Tests live in `crack_detection_sam2/tests/` and start with the existing sys.path boilerplate
  (`sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))`).
- Shared lib `crackseg_common` is importable (installed/on path) from this repo.
- Data: craq `/home/zzz90/research/_data/labeled32_craq_v3/tiles_512`,
  crack `/home/zzz90/research/_data/labeled32_crack_v3/tiles_512`, split file
  `group_split_stem.json` (4 folds).
- Commit messages must end with the trailer (per this environment's rule):
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` — the `-m` templates
  below omit it for brevity; add it when committing.

---

## Task 1: Add `prev_mask` + encode/decode split to `PromptedSAM2Seg`

**Files:**
- Modify: `crack_detection_sam2/model_prompted_sam2.py`
- Test: `crack_detection_sam2/tests/test_click_forward.py`

- [ ] **Step 1: Write the failing test**

Create `crack_detection_sam2/tests/test_click_forward.py`:

```python
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_prompted_sam2 import PromptedSAM2Seg


def test_encode_decode_prev_mask_and_backcompat():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = PromptedSAM2Seg(variant="small", image_size=512, device=dev).to(dev)
    x = torch.randn(2, 3, 512, 512, device=dev)
    coords = torch.tensor([[[100., 100.]], [[150., 150.]]], device=dev)   # [B,1,2] (x,y)
    labels = torch.tensor([[1], [1]], dtype=torch.int32, device=dev)      # [B,1]

    enc = m.encode_image(x)
    assert enc["hw"] == (512, 512)
    masks, low = m.decode(enc, coords, labels, prev_mask=None)
    assert masks.shape == (2, 1, 512, 512), masks.shape
    assert low.shape == (2, 1, 128, 128), low.shape   # 4 * (512//16)

    # second click + prev-mask refinement; shapes stay stable
    coords2 = torch.cat([coords, coords + 10], dim=1)
    labels2 = torch.cat([labels, torch.zeros_like(labels)], dim=1)        # add a negative click
    masks2, low2 = m.decode(enc, coords2, labels2, prev_mask=low)
    assert masks2.shape == (2, 1, 512, 512)

    # backward-compatible forward (existing test_prompted_sam2_forward relies on this)
    y = m(x, coords, labels)
    assert y.shape == (2, 1, 512, 512)
    print("OK test_click_forward")


if __name__ == "__main__":
    test_encode_decode_prev_mask_and_backcompat()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python tests/test_click_forward.py`
Expected: FAIL with `AttributeError: 'PromptedSAM2Seg' object has no attribute 'encode_image'`.

- [ ] **Step 3: Implement encode/decode split + prev_mask**

In `crack_detection_sam2/model_prompted_sam2.py`, replace the `forward` method (lines ~35-49) with:

```python
    def encode_image(self, x):
        B, _, H, W = x.shape
        enc_grad = any(p.requires_grad for p in self.image_encoder.parameters())
        with torch.set_grad_enabled(enc_grad):
            bb = self.image_encoder(x)
        fpn = bb["backbone_fpn"]
        high_res = ([self.sam_mask_decoder.conv_s0(fpn[0]),
                     self.sam_mask_decoder.conv_s1(fpn[1])] if self.use_high_res else None)
        return {"feat": fpn[-1], "high_res": high_res, "hw": (H, W)}

    def decode(self, enc, point_coords, point_labels, prev_mask=None):
        sparse, dense = self.sam_prompt_encoder(
            points=(point_coords, point_labels), boxes=None, masks=prev_mask)
        low, _, _, _ = self.sam_mask_decoder(
            image_embeddings=enc["feat"],
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=False, high_res_features=enc["high_res"])
        masks = F.interpolate(low.float(), size=enc["hw"], mode="bilinear", align_corners=False)
        return masks, low

    def forward(self, x, point_coords, point_labels, prev_mask=None):
        enc = self.encode_image(x)
        masks, _ = self.decode(enc, point_coords, point_labels, prev_mask)
        return masks
```

- [ ] **Step 4: Run new test + existing model test to verify both pass**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python tests/test_click_forward.py && /home/zzz90/research/sam2_env/bin/python tests/test_prompted_sam2_forward.py`
Expected: `OK test_click_forward` then `OK test_prompted_sam2_forward`.

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/model_prompted_sam2.py crack_detection_sam2/tests/test_click_forward.py
git commit -m "feat(click-seg): encode/decode split + prev_mask refinement on PromptedSAM2Seg"
```

---

## Task 2: Click-sampling utilities

**Files:**
- Create: `crack_detection_sam2/click_sampling.py`
- Test: `crack_detection_sam2/tests/test_click_sampling.py`

- [ ] **Step 1: Write the failing test**

Create `crack_detection_sam2/tests/test_click_sampling.py`:

```python
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from click_sampling import mask_iou, sample_initial_point, sample_correction_point


def test_mask_iou():
    a = np.zeros((10, 10), bool); a[2:5, 2:5] = True
    b = np.zeros((10, 10), bool)
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, b) == 0.0
    assert mask_iou(b, b) == 1.0   # both empty -> defined as 1.0


def test_initial_positive_inside_gt():
    gt = np.zeros((20, 20), bool); gt[5:10, 5:10] = True
    (r, c), lbl = sample_initial_point(gt, np.random.default_rng(0))
    assert lbl == 1 and gt[r, c]


def test_initial_negative_when_empty():
    gt = np.zeros((20, 20), bool)
    (r, c), lbl = sample_initial_point(gt, np.random.default_rng(0))
    assert lbl == 0


def test_correction_false_negative_is_positive():
    gt = np.zeros((20, 20), bool); gt[5:15, 5:15] = True
    pred = np.zeros((20, 20), bool)                 # all FN
    (r, c), lbl = sample_correction_point(pred, gt, np.random.default_rng(0))
    assert lbl == 1 and gt[r, c]


def test_correction_false_positive_is_negative():
    gt = np.zeros((20, 20), bool)
    pred = np.zeros((20, 20), bool); pred[5:15, 5:15] = True   # all FP
    (r, c), lbl = sample_correction_point(pred, gt, np.random.default_rng(0))
    assert lbl == 0 and pred[r, c]


def test_correction_none_when_perfect():
    gt = np.zeros((20, 20), bool); gt[5:10, 5:10] = True
    assert sample_correction_point(gt, gt, np.random.default_rng(0)) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
    print("OK test_click_sampling")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python tests/test_click_sampling.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'click_sampling'`.

- [ ] **Step 3: Implement `click_sampling.py`**

Create `crack_detection_sam2/click_sampling.py`:

```python
"""Point sampling for SAM-style interactive click training/eval.

Pure numpy/scipy (no torch / model deps) so it is unit-testable in isolation.
Coordinates are (row, col) i.e. (y, x) in image space; conversion to the
prompt encoder's (x, y) order happens in the trainer via build helpers.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def _largest_component_point(mask: np.ndarray):
    """A pixel inside the largest connected component of `mask` (boolean), chosen as the
    in-component pixel closest to that component's centroid. Returns (row, col) or None."""
    if mask.sum() == 0:
        return None
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(lab == big)
    cy, cx = ys.mean(), xs.mean()
    i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    return int(ys[i]), int(xs[i])


def sample_initial_point(gt: np.ndarray, rng):
    """First click. Positive at the center of GT's largest component; if GT is empty,
    a negative click at image center. Returns ((row, col), label)."""
    gt = gt.astype(bool)
    if gt.sum() == 0:
        h, w = gt.shape
        return (h // 2, w // 2), 0
    return _largest_component_point(gt), 1


def sample_correction_point(pred: np.ndarray, gt: np.ndarray, rng):
    """Correction click from the larger error region. False-negative (missed GT) -> positive(1);
    false-positive -> negative(0). Returns ((row, col), label) or None if pred == gt."""
    pred = pred.astype(bool); gt = gt.astype(bool)
    fn = np.logical_and(gt, ~pred)
    fp = np.logical_and(pred, ~gt)
    if fn.sum() == 0 and fp.sum() == 0:
        return None
    if fn.sum() >= fp.sum():
        p = _largest_component_point(fn); lbl = 1
    else:
        p = _largest_component_point(fp); lbl = 0
    if p is None:
        return None
    return p, lbl
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python tests/test_click_sampling.py`
Expected: `OK test_click_sampling`.

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/click_sampling.py crack_detection_sam2/tests/test_click_sampling.py
git commit -m "feat(click-seg): GT click-sampling utilities (initial + correction points)"
```

---

## Task 3: Interactive trainer `train_click.py`

**Files:**
- Create: `crack_detection_sam2/train_click.py`
- Test: `crack_detection_sam2/tests/test_click_metrics.py` (pure-metric unit test)

The heavy end-to-end behavior is verified by the overfit sanity run in Task 3b. Here we unit-test
the click-sequence batching helpers and the metric aggregation, which are deterministic and fast.

- [ ] **Step 1: Write the failing test**

Create `crack_detection_sam2/tests/test_click_metrics.py`:

```python
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train_click import batch_point_tensors, aggregate_click_metrics


def test_batch_point_tensors_xy_order():
    pts = [[(3, 7), (1, 2)]]      # (row, col)
    labs = [[1, 0]]
    coords, labels = batch_point_tensors(pts, labs, "cpu")
    assert coords.shape == (1, 2, 2) and labels.shape == (1, 2)
    # (row=3,col=7) -> (x=7, y=3)
    assert coords[0, 0].tolist() == [7.0, 3.0]
    assert labels[0].tolist() == [1, 0]


def test_aggregate_click_metrics():
    # 2 samples, n_clicks=3. per_click_iou[k] holds the IoU of each sample after click k+1.
    per_click_iou = [[0.5, 0.2], [0.85, 0.4], [0.9, 0.82]]
    noc = [2, 3]   # sample A reached 0.8 at click 2, sample B at click 3
    m = aggregate_click_metrics(per_click_iou, noc, n_clicks=3, iou_target=0.8)
    assert abs(m["iou@1"] - 0.35) < 1e-6
    assert abs(m["iou@3"] - 0.86) < 1e-6
    assert abs(m["noc@0.8"] - 2.5) < 1e-6
    print("OK test_click_metrics")


if __name__ == "__main__":
    test_batch_point_tensors_xy_order()
    test_aggregate_click_metrics()
    print("OK test_click_metrics")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python tests/test_click_metrics.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'train_click'`.

- [ ] **Step 3: Implement `train_click.py`**

Create `crack_detection_sam2/train_click.py`:

```python
"""Train a click-promptable SAM2 expert (binary segmentation) with SAM-style iterative clicks.

Reuses PromptedSAM2Seg (model_prompted_sam2.py), the loss/schedule from train_prompt.py, and the
shared dataset/augment. Each tile is annotated by simulating `n_clicks` clicks: a positive seed,
then correction points sampled from the error region. Loss is averaged over clicks; eval reports
IoU@{1,3,5} clicks and NoC@0.8.

Example (craquelure expert, fold 0):
    /home/zzz90/research/sam2_env/bin/python train_click.py \
        --tiles_root /home/zzz90/research/_data/labeled32_craq_v3/tiles_512 \
        --split /home/zzz90/research/_data/labeled32_craq_v3/tiles_512/group_split_stem.json \
        --fold 0 --variant small --image_size 512 --epochs 80 --batch_size 2 --n_clicks 8 \
        --class_names background,craquelure \
        --output_dir runs/2026-06-09-click-craq-4fold/fold0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from crackseg_common.augment import train_transforms, val_transforms
import crackseg_common.dataset as _dataset
from crackseg_common.dataset import TileSegDataset, compute_class_weights, load_tile_index, set_class_names
from model_prompted_sam2 import PromptedSAM2Seg
from train_prompt import BinaryCEDiceLoss, cosine_with_warmup
from click_sampling import mask_iou, sample_initial_point, sample_correction_point


def batch_point_tensors(pts, labs, device):
    """pts: list (len B) of lists of (row, col); labs: same shape of int labels.
    All rows have equal length N. Returns coords [B,N,2] as (x,y) float, labels [B,N] int32."""
    B = len(pts); N = len(pts[0])
    coords = torch.zeros(B, N, 2)
    labels = torch.zeros(B, N, dtype=torch.int32)
    for b in range(B):
        for i, (r, c) in enumerate(pts[b]):
            coords[b, i, 0] = float(c)   # x = col
            coords[b, i, 1] = float(r)   # y = row
            labels[b, i] = int(labs[b][i])
    return coords.to(device), labels.to(device)


def _init_points(gt_np, rng):
    B = gt_np.shape[0]
    pts = [[] for _ in range(B)]; labs = [[] for _ in range(B)]
    for b in range(B):
        (r, c), l = sample_initial_point(gt_np[b], rng)
        pts[b].append((r, c)); labs[b].append(l)
    return pts, labs


def _append_correction(pts, labs, pred_np, gt_np, rng):
    for b in range(pred_np.shape[0]):
        nxt = sample_correction_point(pred_np[b], gt_np[b], rng)
        if nxt is None:                       # already perfect -> repeat last click (uniform N)
            pts[b].append(pts[b][-1]); labs[b].append(labs[b][-1])
        else:
            (r, c), l = nxt; pts[b].append((r, c)); labs[b].append(l)


def clicks_train_loss(model, img, gt, n_clicks, rng, criterion):
    """Run n_clicks iterations on a batch, return loss averaged over clicks (graph retained)."""
    enc = model.encode_image(img)
    gt_np = (gt > 0).cpu().numpy()
    pts, labs = _init_points(gt_np, rng)
    prev = None
    total = 0.0
    for k in range(n_clicks):
        coords, labels = batch_point_tensors(pts, labs, img.device)
        masks, low = model.decode(enc, coords, labels, prev_mask=prev)
        loss, _ = criterion(masks, gt)
        total = total + loss
        prev = low.detach()
        if k < n_clicks - 1:
            pred_np = (masks.detach().squeeze(1) > 0).cpu().numpy()
            _append_correction(pts, labs, pred_np, gt_np, rng)
    return total / n_clicks


def aggregate_click_metrics(per_click_iou, noc, n_clicks, iou_target=0.8):
    def at(k):
        k = min(k, n_clicks)
        vals = per_click_iou[k - 1]
        return float(np.mean(vals)) if vals else 0.0
    return {
        "iou@1": at(1), "iou@3": at(3), "iou@5": at(5),
        "noc@0.8": float(np.mean(noc)) if noc else float(n_clicks),
        "cap_rate": float(np.mean([r >= n_clicks for r in noc])) if noc else 1.0,
    }


@torch.no_grad()
def evaluate_clicks(model, loader, device, n_clicks, iou_target=0.8, seed=0):
    model.eval()
    rng = np.random.default_rng(seed)
    per_click_iou = [[] for _ in range(n_clicks)]
    noc = []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        gt = batch["mask"].to(device, non_blocking=True)
        B = img.size(0)
        enc = model.encode_image(img)
        gt_np = (gt > 0).cpu().numpy()
        pts, labs = _init_points(gt_np, rng)
        prev = None
        reached = [None] * B
        for k in range(n_clicks):
            coords, labels = batch_point_tensors(pts, labs, device)
            masks, low = model.decode(enc, coords, labels, prev_mask=prev)
            prev = low
            pred_np = (masks.squeeze(1) > 0).cpu().numpy()
            for b in range(B):
                iou = mask_iou(pred_np[b], gt_np[b])
                per_click_iou[k].append(iou)
                if reached[b] is None and iou >= iou_target:
                    reached[b] = k + 1
            if k < n_clicks - 1:
                _append_correction(pts, labs, pred_np, gt_np, rng)
        for b in range(B):
            noc.append(reached[b] if reached[b] is not None else n_clicks)
    return aggregate_click_metrics(per_click_iou, noc, n_clicks, iou_target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiles_root", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--variant", default="small")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--n_clicks", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--base_lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ce_weight", type=float, default=0.5)
    p.add_argument("--dice_weight", type=float, default=0.5)
    p.add_argument("--class_weight_mode", default="median_freq",
                   choices=["median_freq", "inv_sqrt", "none"])
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_items", type=int, default=0, help="0 = all; >0 caps for sanity runs")
    p.add_argument("--output_dir", default="outputs/click_run")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--class_names", default=None)
    args = p.parse_args()

    if args.class_names:
        set_class_names([s.strip() for s in args.class_names.split(",") if s.strip()])

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    tile_index = load_tile_index(args.tiles_root)
    with open(args.split) as f:
        fd = json.load(f)["folds"][args.fold]
    by_name = {it["tile"]: it for it in tile_index["items"]}
    train_items = [by_name[n] for n in fd["train"] if n in by_name]
    val_items = [by_name[n] for n in fd["val"] if n in by_name]
    if args.max_train_items > 0:
        train_items = train_items[:args.max_train_items]
    print(f"fold={args.fold} train={len(train_items)} val={len(val_items)}")

    train_ds = TileSegDataset(args.tiles_root, train_items,
                              transforms=train_transforms(image_size=args.image_size))
    val_ds = TileSegDataset(args.tiles_root, val_items,
                            transforms=val_transforms(image_size=args.image_size))
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)

    pos_weight = None
    if args.class_weight_mode != "none":
        _, counts = compute_class_weights(train_items, args.tiles_root,
                                          num_classes=_dataset.NUM_CLASSES,
                                          mode=args.class_weight_mode)
        if len(counts) >= 2 and counts[1] > 0:
            pos_weight = min(float(counts[0]) / float(counts[1]), 100.0)
        print(f"pixel counts: {counts.tolist()}, pos_weight={pos_weight}")

    model = PromptedSAM2Seg(variant=args.variant, image_size=args.image_size, device=device).to(device)
    groups = model.param_groups(base_lr=args.base_lr)
    base_lrs = [g["lr"] for g in groups]
    optimizer = torch.optim.AdamW(groups, lr=args.base_lr, weight_decay=args.weight_decay)
    criterion = BinaryCEDiceLoss(ce_weight=args.ce_weight, dice_weight=args.dice_weight,
                                 pos_weight=pos_weight).to(device)
    scaler = None if args.no_amp or device != "cuda" else torch.amp.GradScaler("cuda")

    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = max(1, args.warmup_epochs * len(train_loader))
    log = {"args": vars(args), "history": []}
    best = -1.0
    rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0; n = 0; t0 = time.time()
        for it, batch in enumerate(train_loader):
            scale = cosine_with_warmup(epoch * len(train_loader) + it, total_steps, warmup_steps)
            for g, lr in zip(optimizer.param_groups, base_lrs):
                g["lr"] = lr * scale
            img = batch["image"].to(device, non_blocking=True)
            gt = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = clicks_train_loss(model, img, gt, args.n_clicks, rng, criterion)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters() if hasattr(model, "trainable_parameters")
                                               else [p for p in model.parameters() if p.requires_grad], 5.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss = clicks_train_loss(model, img, gt, args.n_clicks, rng, criterion)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                optimizer.step()
            run_loss += float(loss.detach()) * img.size(0); n += img.size(0)
            if (it + 1) % args.log_interval == 0 or (it + 1) == len(train_loader):
                print(f"  ep{epoch+1}/{args.epochs} it{it+1}/{len(train_loader)} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e} loss={run_loss/max(1,n):.4f} "
                      f"{(time.time()-t0):.1f}s", flush=True)

        ev = evaluate_clicks(model, val_loader, device, args.n_clicks, seed=args.seed)
        print(f"[val] ep{epoch+1} IoU@1={ev['iou@1']:.3f} IoU@3={ev['iou@3']:.3f} "
              f"IoU@5={ev['iou@5']:.3f} NoC@0.8={ev['noc@0.8']:.2f} cap={ev['cap_rate']:.2f}", flush=True)
        log["history"].append({"epoch": epoch + 1, "train_loss": run_loss / max(1, n), "val": ev})
        with open(out_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)
        torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                    "args": vars(args), "val": ev}, out_dir / "last.pt")
        if ev["iou@5"] > best:
            best = ev["iou@5"]
            torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                        "args": vars(args), "val": ev}, out_dir / "best.pt")
            print(f"[best] ep{epoch+1} IoU@5={best:.4f}")
    # final metrics summary for experiment tracking
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"best_iou@5": best, "last_val": log["history"][-1]["val"]}, f, indent=2)
    print(f"done best_iou@5={best:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python tests/test_click_metrics.py`
Expected: `OK test_click_metrics`.

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/train_click.py crack_detection_sam2/tests/test_click_metrics.py
git commit -m "feat(click-seg): iterative-click trainer + interactive metrics (IoU@k, NoC)"
```

---

## Task 3b: Overfit-one-tile sanity (end-to-end GPU check)

**Files:** none (uses Task 3 code). This proves the click path actually drives the mask.

- [ ] **Step 1: Run a short overfit on a single craq tile**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
D=/home/zzz90/research/_data/labeled32_craq_v3/tiles_512
/home/zzz90/research/sam2_env/bin/python train_click.py \
  --tiles_root "$D" --split "$D/group_split_stem.json" --fold 0 \
  --variant small --image_size 512 --epochs 60 --batch_size 1 --n_clicks 8 \
  --max_train_items 1 --warmup_epochs 1 --base_lr 5e-4 \
  --class_names background,craquelure \
  --output_dir /tmp/click_overfit_craq
```
Expected: the `[val]` line's `IoU@5` climbs across epochs. Because val != the single train tile,
do not require val>0.7; instead confirm **train loss drops below ~0.2** and IoU@5 trends upward
(the model is learning to respond to clicks). If train loss stays flat near its start, STOP and
debug (likely a points/coords order or prev_mask wiring bug) before any full run.

- [ ] **Step 2: Record the result**

Note the final train loss and IoU@5 trend in the task hand-off. No commit (scratch output in /tmp).

---

## Task 4: Inference helper `predict_click.py`

**Files:**
- Create: `crack_detection_sam2/predict_click.py`
- Test: `crack_detection_sam2/tests/test_predict_click_roundtrip.py`

- [ ] **Step 1: Write the failing test**

Create `crack_detection_sam2/tests/test_predict_click_roundtrip.py`:

```python
import os, sys, tempfile
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_prompted_sam2 import PromptedSAM2Seg
from predict_click import load_model_from_ckpt, predict_click


def test_roundtrip():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = PromptedSAM2Seg(variant="small", image_size=512, device=dev).to(dev)
    fd, path = tempfile.mkstemp(suffix=".pt"); os.close(fd)
    torch.save({"model": m.state_dict(), "args": {"variant": "small", "image_size": 512}}, path)

    model, payload = load_model_from_ckpt(path, dev)
    img = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    mask, low = predict_click(model, img, pos_points=[(256, 256)], neg_points=[],
                              prev_mask=None, device=dev)
    assert mask.shape == (512, 512) and mask.dtype == bool, (mask.shape, mask.dtype)
    assert tuple(low.shape[-2:]) == (128, 128), low.shape

    # refinement click reusing prev low-res logits
    mask2, low2 = predict_click(model, img, pos_points=[(256, 256)], neg_points=[(10, 10)],
                                prev_mask=low, device=dev)
    assert mask2.shape == (512, 512)
    os.remove(path)
    print("OK test_predict_click_roundtrip")


if __name__ == "__main__":
    test_roundtrip()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python tests/test_predict_click_roundtrip.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'predict_click'`.

- [ ] **Step 3: Implement `predict_click.py`**

Create `crack_detection_sam2/predict_click.py`:

```python
"""Inference helper for click-promptable SAM2 experts.

Single entry point consumed by the CVAT Nuclio interactor (sub-project 2). Same contract names
as the existing predict_full.py modules (`load_model_from_ckpt`). v0 assumes `img` is already at
the model working resolution (a tile); the CVAT-side coordinate mapping for arbitrary-size frames
is handled in sub-project 2.
"""
from __future__ import annotations

import numpy as np
import torch

from model_prompted_sam2 import PromptedSAM2Seg
from crackseg_common.augment import val_transforms


def load_model_from_ckpt(ckpt, device):
    payload = torch.load(ckpt, map_location=device)
    a = payload.get("args", {})
    model = PromptedSAM2Seg(variant=a.get("variant", "small"),
                            image_size=a.get("image_size", 512), device=device)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, payload


@torch.no_grad()
def predict_click(model, img, pos_points, neg_points, prev_mask, device, image_size=512):
    """img: HxWx3 uint8 RGB ndarray. pos/neg_points: lists of (x, y) in img pixel space.
    Returns (mask: bool HxW at working resolution, low_res_logits: tensor [1,1,128,128])."""
    tf = val_transforms(image_size=image_size)
    out = tf(image=img, mask=np.zeros(img.shape[:2], np.uint8))
    x = out["image"].unsqueeze(0).to(device)

    pts = [(y, xx) for (xx, y) in pos_points] + [(y, xx) for (xx, y) in neg_points]   # -> (row, col)
    labs = [1] * len(pos_points) + [0] * len(neg_points)
    if pts:
        coords = torch.tensor([[[float(c), float(r)] for (r, c) in pts]], device=device)  # (x,y)
        labels = torch.tensor([labs], dtype=torch.int32, device=device)
    else:
        coords = torch.zeros(1, 0, 2, device=device)
        labels = torch.zeros(1, 0, dtype=torch.int32, device=device)

    enc = model.encode_image(x)
    masks, low = model.decode(enc, coords, labels, prev_mask=prev_mask)
    mask = (masks[0, 0] > 0).cpu().numpy()
    return mask, low
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/zzz90/research/crack_detection_sam2 && /home/zzz90/research/sam2_env/bin/python tests/test_predict_click_roundtrip.py`
Expected: `OK test_predict_click_roundtrip`.

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/predict_click.py crack_detection_sam2/tests/test_predict_click_roundtrip.py
git commit -m "feat(click-seg): predict_click inference helper for the CVAT interactor"
```

---

## Task 5: Full training runs + experiment tracking

**Files:**
- Create: `crack_detection_sam2/runs/2026-06-09-click-craq-4fold/` (outputs + manifest)
- Create: `crack_detection_sam2/runs/2026-06-09-click-crack-4fold/` (outputs + manifest)

**Before running:** invoke the `experiment-tracking` skill and follow it (it owns the manifest /
code_snapshot / metrics format used by the sibling `2026-06-06-decoder-craq-4fold` run).

- [ ] **Step 1: Train craquelure expert, 4 folds**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
D=/home/zzz90/research/_data/labeled32_craq_v3/tiles_512
for f in 0 1 2 3; do
  /home/zzz90/research/sam2_env/bin/python train_click.py \
    --tiles_root "$D" --split "$D/group_split_stem.json" --fold $f \
    --variant small --image_size 512 --epochs 80 --batch_size 2 --n_clicks 8 \
    --class_names background,craquelure \
    --output_dir runs/2026-06-09-click-craq-4fold/fold$f
done
```
Expected: each fold writes `last.pt`, `best.pt`, `log.json`, `metrics.json`; `[val]` IoU@5 should
exceed the single-positive-click baseline (IoU@1) by a clear margin (clicks are helping).

- [ ] **Step 2: Train crack expert, 4 folds**

Run:
```bash
cd /home/zzz90/research/crack_detection_sam2
D=/home/zzz90/research/_data/labeled32_crack_v3/tiles_512
for f in 0 1 2 3; do
  /home/zzz90/research/sam2_env/bin/python train_click.py \
    --tiles_root "$D" --split "$D/group_split_stem.json" --fold $f \
    --variant small --image_size 512 --epochs 80 --batch_size 2 --n_clicks 8 \
    --class_names background,crack \
    --output_dir runs/2026-06-09-click-crack-4fold/fold$f
done
```
Expected: same artifacts under `runs/2026-06-09-click-crack-4fold/foldN`.

- [ ] **Step 3: Write run manifests**

For each run dir create `manifest.md` (mirror `runs/2026-06-06-decoder-craq-4fold/manifest.md`):
exact commands above, data path, env (`/home/zzz90/research/sam2_env`, torch 2.11.0+cu128, GPU),
git SHA, and a per-fold table of `best_iou@5` / `noc@0.8` read from each `metrics.json`. Snapshot
the code (`model_prompted_sam2.py`, `click_sampling.py`, `train_click.py`, `predict_click.py`)
into a `code_snapshot/` subdir per the experiment-tracking convention.

- [ ] **Step 4: Commit**

```bash
cd /home/zzz90/research
git add crack_detection_sam2/runs/2026-06-09-click-craq-4fold crack_detection_sam2/runs/2026-06-09-click-crack-4fold
git commit -m "exp(click-seg): 4-fold click-promptable craq + crack experts (runs + manifests)"
```

- [ ] **Step 5: Decide go / fallback per class**

Read each run's `metrics.json`. Decision rule (from spec section 8):
- If a class reaches usable interactive quality (e.g. IoU@5 >= ~0.6 and NoC@0.8 well under the
  8-click cap), keep its click-promptable checkpoint for sub-project 2.
- If a class fails to learn a click response (IoU@5 not materially above IoU@1, or cap_rate high),
  flag it: sub-project 2 will serve that class's existing **dense** expert via a
  "click-selects-connected-component" interactor instead. Record the decision in the manifest.

---

## Self-Review notes (already reconciled)

- Spec coverage: model (Task 1), trainer + iterative clicks + eval metrics (Task 3/3b), inference
  helper (Task 4), experiments + tracking + go/fallback (Task 5). Sampling utilities (Task 2)
  back the iterative loop. predict_click feeds sub-project 2.
- Spec deviation (intentional, DRY): the model is the **extended** `model_prompted_sam2.py`, not a
  new `model_click_seg.py` — `PromptedSAM2Seg` already implements the external-point forward. The
  spec was updated to match.
- Symbol consistency: `encode_image`/`decode`/`prev_mask`, `batch_point_tensors`,
  `sample_initial_point`/`sample_correction_point`, `mask_iou`, `aggregate_click_metrics`,
  `evaluate_clicks`, `load_model_from_ckpt`/`predict_click` are used identically across tasks.
```
