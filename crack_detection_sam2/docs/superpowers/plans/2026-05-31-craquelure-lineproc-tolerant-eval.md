# craquelure 線條後處理 + 正確指標重評 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 對三個 craquelure 方法(PromptedSAM2 oracle/yolo、dense-seg、SepSAM-YOLO)套用線條後處理(CC 雜訊抑制 + centerline)並改用 clDice/tolerant-F1 重評,看換正確口徑後名次/差距是否改變。

**Architecture:** 各方法在同一份 craq val(labeled32_craq_v3 4-fold,fold 對齊)dump per-tile 二值預測 png → 共用 `lineproc.py`(post-proc + 指標)→ `eval_lineproc_craq.py` 統一讀 png+GT,raw/pp × {vanilla,tolerant,clDice} 出對照表。

**Tech Stack:** Python 3.12 / `sam2_env`(主)+ `SepSAM2_env`(僅 dump YOLO);PyTorch、SAM2、skimage(skeletonize/dilation)、scipy.ndimage(label/binary_dilation)。

> **環境前提:**
> - 主 python `/home/zzz90/research/sam2_env/bin/python`(簡稱 `$PY`),cwd `/home/zzz90/research/crack_detection_sam2`。SepSAM2 dump 用 `/home/zzz90/research/SepSAM2_env/bin/python`,cwd `/home/zzz90/research/crack_detection_SepSAM2/sepsam`。
> - 非 git → commit 選用。skimage 已裝(本 session 裝過)。
> - 資料: `data/labeled32_craq_v3/tiles_512/{images,masks}`(.png,0/1),`craqfold{k}/{val_images,val_masks}`(已存在),`craqfold{k}/yolo_points.json`(已存在)。
> - **已確認:** `augment.py` 的 CLAHE 只在 `train_transforms`(`A.CLAHE p=0.5`),`val_transforms` 無 CLAHE → **dense-seg dump 用 raw val(不套 CLAHE)**,對齊其 0.634 評估協定。dense-seg = `model_seg.SAM2SemSeg`,class_names=`background,craquelure`(num_classes=2),variant small,512 tile(=image_size,無需 sliding)。`eval_promptsam2_craq.py` 有可重用的 `load_img`/`predict`。
> - 對照基準(舊 vanilla pixel-F1,排除 fold2 平均):dense-seg 0.634、SepSAM-YOLO 0.541、PromptedSAM2 oracle 0.554 / yolo 0.500。
> - 預測統一存 512×512 二值 png(0/255),GT 同尺寸,直接比對。

---

## File Structure（皆在 `crack_detection_sam2/`,除 Task 4）
- `lineproc.py`(新)— `cc_filter`/`skeleton_centerline`/`cldice`/`tolerant_f1`,純函式。
- `tests/test_lineproc.py`(新)— 合成 mask 單元測試。
- `dump_preds_promptsam2.py`(新)— oracle + yolo 兩 method,sam2_env。
- `dump_preds_denseseg.py`(新)— dense-seg,sam2_env,raw val。
- `crack_detection_SepSAM2/sepsam/scripts/dump_preds_sepsam_yolo.py`(新)— SepSAM2 env。
- `eval_lineproc_craq.py`(新)— 統一讀 png + GT,raw/pp × 3 指標,對照表。
- 預測輸出: `preds/{method}/fold{k}/{stem}.png`(method ∈ promptsam2_oracle, promptsam2_yolo, denseseg, sepsam_yolo),GT 用 `craqfold{k}/val_masks/{stem}.png`。

---

## Task 1: lineproc 後處理 + 指標模組

**Files:** Create `lineproc.py`, `tests/test_lineproc.py`

- [ ] **Step 1: 失敗測試** — `tests/test_lineproc.py`:
```python
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lineproc import cc_filter, skeleton_centerline, cldice, tolerant_f1


def _line(h=64, w=64, row=32):
    m = np.zeros((h, w), np.uint8); m[row-1:row+2, 5:60] = 1; return m


def test_cc_filter_removes_small():
    m = _line()
    m[2:4, 2:4] = 1  # 4px 小雜訊塊
    out = cc_filter(m, min_area=20)
    assert out[2, 2] == 0          # 小塊被移除
    assert out[32, 30] == 1        # 主線保留


def test_skeleton_centerline_thins_then_dilates():
    m = np.zeros((64, 64), np.uint8); m[28:36, 5:60] = 1  # 8px 寬帶
    out = skeleton_centerline(m, width=3)
    col = out[:, 30]
    assert 1 <= col.sum() <= 5     # 變細到 ~3px
    assert out.dtype == bool or out.max() <= 1


def test_cldice_identical_is_one_disjoint_zero():
    m = _line().astype(bool)
    assert cldice(m, m) > 0.99
    other = np.zeros_like(m); other[5:8, 5:60] = 1
    assert cldice(m, other.astype(bool)) < 0.2


def test_tolerant_f1_shifted_line_high_vanilla_low():
    m = _line().astype(bool)
    shifted = np.roll(m, 1, axis=0)   # 平移 1px
    assert tolerant_f1(m, shifted, tol=3) > 0.9
    # vanilla overlap of 3px line shifted 1px is much lower
    inter = (m & shifted).sum(); van = 2*inter/(m.sum()+shifted.sum())
    assert van < tolerant_f1(m, shifted, tol=3)


def test_empty_cases():
    z = np.zeros((16, 16), bool)
    assert cldice(z, z) == 1.0 and tolerant_f1(z, z) == 1.0
    nz = z.copy(); nz[5, 5] = True
    assert cldice(nz, z) == 0.0 and tolerant_f1(nz, z) == 0.0


if __name__ == "__main__":
    for f in [test_cc_filter_removes_small, test_skeleton_centerline_thins_then_dilates,
              test_cldice_identical_is_one_disjoint_zero, test_tolerant_f1_shifted_line_high_vanilla_low,
              test_empty_cases]:
        f()
    print("OK test_lineproc")
```

- [ ] **Step 2: 跑→失敗** `$PY tests/test_lineproc.py` → ModuleNotFoundError: lineproc

- [ ] **Step 3: 實作 `lineproc.py`:**
```python
"""lineproc.py — 線狀預測後處理 + 對細線公平的指標(對齊 Notion:clDice / tolerant)。"""
import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure, iterate_structure
from skimage.morphology import skeletonize, dilation, disk


def cc_filter(mask, min_area):
    """移除像素數 < min_area 的連通元件。回傳 bool。"""
    mask = np.asarray(mask).astype(bool)
    lab, n = label(mask)
    if n == 0:
        return mask
    out = np.zeros_like(mask)
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() >= min_area:
            out |= comp
    return out


def skeleton_centerline(mask, width=3):
    """skeletonize 後膨脹到 ~width px(對齊 centerline 標準)。回傳 bool。"""
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    sk = skeletonize(mask)
    r = max(1, width // 2)
    return dilation(sk, disk(r))


def _dilate(mask, tol):
    st = iterate_structure(generate_binary_structure(2, 1), tol)
    return binary_dilation(mask, structure=st)


def cldice(pred, gt):
    """centerline Dice:對寬度不敏感、對拓樸敏感。"""
    pred = np.asarray(pred).astype(bool); gt = np.asarray(gt).astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    sp = skeletonize(pred); sg = skeletonize(gt)
    tprec = (sp & gt).sum() / max(sp.sum(), 1)
    tsens = (sg & pred).sum() / max(sg.sum(), 1)
    if (tprec + tsens) == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))


def tolerant_f1(pred, gt, tol=3):
    """容差 F1:pred 落在 gt 的 tol 膨脹內算命中,反之算 recall。"""
    pred = np.asarray(pred).astype(bool); gt = np.asarray(gt).astype(bool)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0
    gt_d = _dilate(gt, tol); pred_d = _dilate(pred, tol)
    prec = (pred & gt_d).sum() / max(pred.sum(), 1)
    rec = (gt & pred_d).sum() / max(gt.sum(), 1)
    if (prec + rec) == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))
```

- [ ] **Step 4: 跑→通過** `$PY tests/test_lineproc.py` → `OK test_lineproc`
- [ ] **Step 5(選用) commit**

---

## Task 2: dump PromptedSAM2 預測（oracle + yolo）

**Files:** Create `dump_preds_promptsam2.py`

- [ ] **Step 1: 實作 `dump_preds_promptsam2.py`**(重用 `eval_promptsam2_craq` 的 load_img/predict + gt_points):
```python
"""dump_preds_promptsam2.py — 把 PromptedSAM2 oracle/yolo 預測存成 preds/{method}/fold{k}/{stem}.png。"""
import argparse, glob, json, os
import cv2, numpy as np, torch
from model_prompted_sam2 import PromptedSAM2Seg
from gt_points import gt_points
from eval_promptsam2_craq import load_img, predict

TILES = "data/labeled32_craq_v3/tiles_512"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--n_points", type=int, default=10)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for k in args.folds:
        model = PromptedSAM2Seg(variant="small", image_size=args.image_size, device=dev).to(dev)
        model.load_state_dict(torch.load(f"outputs/promptsam2_craq_fold{k}/best.pt", map_location=dev)["model"])
        model.eval()
        vi = os.path.join(TILES, f"craqfold{k}", "val_images")
        vm = os.path.join(TILES, f"craqfold{k}", "val_masks")
        yp = json.load(open(os.path.join(TILES, f"craqfold{k}", "yolo_points.json")))
        for method, getpts in [("promptsam2_oracle", "oracle"), ("promptsam2_yolo", "yolo")]:
            od = f"preds/{method}/fold{k}"; os.makedirs(od, exist_ok=True)
            for p in sorted(glob.glob(os.path.join(vi, "*.png"))):
                st = os.path.splitext(os.path.basename(p))[0]
                img_t, hw = load_img(p, args.image_size)
                if getpts == "oracle":
                    gt = cv2.imread(os.path.join(vm, st + ".png"), 0)
                    pts, _ = gt_points(gt > 0, args.n_points); pts = pts.tolist()
                else:
                    pts = yp.get(st, [])
                pred = predict(model, img_t, pts, args.image_size, dev, hw)
                cv2.imwrite(os.path.join(od, st + ".png"), (pred.astype(np.uint8) * 255))
            print(f"{method} fold{k}: dumped -> {od}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑** `$PY dump_preds_promptsam2.py --folds 0 1 2 3`
Expected: 8 行(2 method × 4 fold),產生 `preds/promptsam2_oracle/fold{k}/*.png` 與 `preds/promptsam2_yolo/fold{k}/*.png`。抽查一張非空。
- [ ] **Step 3(選用) commit**

---

## Task 3: dump dense-seg 預測（raw val,無 CLAHE）

**Files:** Create `dump_preds_denseseg.py`

- [ ] **Step 1: 實作 `dump_preds_denseseg.py`**(SAM2SemSeg,val_transforms 無 CLAHE,argmax 取 craquelure):
```python
"""dump_preds_denseseg.py — dense-seg craquelure expert(expert_craq_v3_clahe)在 raw val 上的預測,
存 preds/denseseg/fold{k}/{stem}.png。注意:CLAHE 只是 train aug,val 不套(對齊其 0.634 評估)。"""
import argparse, glob, os
import cv2, numpy as np, torch
from dataset import set_class_names
from augment import val_transforms
from model_seg import SAM2SemSeg

TILES = "data/labeled32_craq_v3/tiles_512"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--run_prefix", default="outputs/expert_craq_v3_clahe")
    ap.add_argument("--image_size", type=int, default=512)
    args = ap.parse_args()
    set_class_names(["background", "craquelure"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = val_transforms(image_size=args.image_size)
    for k in args.folds:
        model = SAM2SemSeg(variant="small", num_classes=2, device=dev).to(dev)
        ck = torch.load(f"{args.run_prefix}_fold{k}_small/best.pt", map_location=dev)
        model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
        model.eval()
        vi = os.path.join(TILES, f"craqfold{k}", "val_images")
        od = f"preds/denseseg/fold{k}"; os.makedirs(od, exist_ok=True)
        for p in sorted(glob.glob(os.path.join(vi, "*.png"))):
            st = os.path.splitext(os.path.basename(p))[0]
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            out = tfm(image=rgb, mask=np.zeros((h, w), np.uint8))
            x = out["image"].unsqueeze(0).to(dev)
            with torch.no_grad():
                logits = model(x)                       # [1,2,H,W]
            pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)  # 0/1
            if pred.shape != (h, w):
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(od, st + ".png"), ((pred == 1).astype(np.uint8) * 255))
        print(f"denseseg fold{k}: dumped -> {od}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: smoke(fold0)確認模型載入 + 輸出非全空**
Run: `$PY dump_preds_denseseg.py --folds 0`
Expected: `denseseg fold0: dumped -> preds/denseseg/fold0`;抽查 `preds/denseseg/fold0/*.png` 至少有非空 mask。
- 若 `SAM2SemSeg` 建構簽名或 ckpt key 不符,先 READ `model_seg.py` 的 `__init__` 與 `predict_full.py` 的載入方式對齊(那是 known-working),報告調整內容;真錯則 BLOCKED 附 traceback。
- [ ] **Step 3: 跑完整 4-fold** `$PY dump_preds_denseseg.py --folds 0 1 2 3`
- [ ] **Step 4(選用) commit**

---

## Task 4: dump SepSAM-YOLO 預測（SepSAM2 env）

**Files:** Create `/home/zzz90/research/crack_detection_SepSAM2/sepsam/scripts/dump_preds_sepsam_yolo.py`

- [ ] **Step 1: 實作 `scripts/dump_preds_sepsam_yolo.py`:**
```python
"""dump_preds_sepsam_yolo.py — craq YOLO agent(conf0.05/iou0.5)union mask,
存到 crack_detection_sam2 的 preds/sepsam_yolo/fold{k}/{stem}.png。"""
import argparse, glob, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import Agent

TILES = "/home/zzz90/research/crack_detection_sam2/data/labeled32_craq_v3/tiles_512"
PREDS = "/home/zzz90/research/crack_detection_sam2/preds/sepsam_yolo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()
    from ultralytics.nn.tasks import C2f_SEA  # noqa
    for k in args.folds:
        agent = Agent(f"runs/segment/runs/sepsam_agent_craq_cv_fold{k}/weights/best.pt", device="cuda")
        vi = os.path.join(TILES, f"craqfold{k}", "val_images")
        od = os.path.join(PREDS, f"fold{k}"); os.makedirs(od, exist_ok=True)
        for p in sorted(glob.glob(os.path.join(vi, "*.png"))):
            st = os.path.splitext(os.path.basename(p))[0]
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            mask, _ = agent.predict(rgb, conf=args.conf, iou=0.5)  # 0/255 union
            cv2.imwrite(os.path.join(od, st + ".png"), mask)
        print(f"sepsam_yolo fold{k}: dumped -> {od}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑(SepSAM2 env)**
```bash
cd /home/zzz90/research/crack_detection_SepSAM2/sepsam
/home/zzz90/research/SepSAM2_env/bin/python scripts/dump_preds_sepsam_yolo.py --folds 0 1 2 3 --conf 0.05
```
Expected: 4 行,產生 `crack_detection_sam2/preds/sepsam_yolo/fold{k}/*.png`。
- [ ] **Step 3(選用) commit**

---

## Task 5: 統一評估（raw vs post-proc × 3 指標）

**Files:** Create `eval_lineproc_craq.py`

- [ ] **Step 1: 實作 `eval_lineproc_craq.py`:**
```python
"""eval_lineproc_craq.py — 讀各 method 的 preds + GT,raw 與 post-proc 各算 vanilla-F1/tolerant-F1/clDice,出對照表。"""
import argparse, glob, os
import cv2, numpy as np
from lineproc import cc_filter, skeleton_centerline, cldice, tolerant_f1

TILES = "data/labeled32_craq_v3/tiles_512"
METHODS = ["denseseg", "sepsam_yolo", "promptsam2_oracle", "promptsam2_yolo"]
DENSE_VANILLA = {"denseseg": 0.634, "sepsam_yolo": 0.541,
                 "promptsam2_oracle": 0.554, "promptsam2_yolo": 0.500}  # 舊 vanilla 排除fold2 參考


def vanilla_f1(p, g):
    p = p.astype(bool); g = g.astype(bool)
    tp = (p & g).sum(); fp = (p & ~g).sum(); fn = (~p & g).sum()
    if (tp + fp + fn) == 0:
        return 1.0
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return 0.0 if (pr + rc) == 0 else float(2 * pr * rc / (pr + rc))


def eval_method(method, k, min_area, tol, postproc):
    pd = f"preds/{method}/fold{k}"
    vm = os.path.join(TILES, f"craqfold{k}", "val_masks")
    van = tolm = cld = 0.0; n = 0
    for pp in sorted(glob.glob(os.path.join(pd, "*.png"))):
        st = os.path.splitext(os.path.basename(pp))[0]
        pred = cv2.imread(pp, 0) > 0
        gt = cv2.imread(os.path.join(vm, st + ".png"), 0) > 0
        if postproc:
            pred = cc_filter(pred, min_area)
            pred = skeleton_centerline(pred, width=3)
        van += vanilla_f1(pred, gt); tolm += tolerant_f1(pred, gt, tol); cld += cldice(pred, gt); n += 1
    return {"van": van / max(n, 1), "tol": tolm / max(n, 1), "cld": cld / max(n, 1), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--min_area", type=int, default=64)
    ap.add_argument("--tol", type=int, default=3)
    args = ap.parse_args()
    no2 = [k for k in args.folds if k != 2]
    for postproc in [False, True]:
        tag = "post-proc" if postproc else "raw"
        print(f"\n==== {tag} (min_area={args.min_area}, tol={args.tol}) — 排除fold2 平均 ====")
        print(f"{'method':22} {'vanilla':>8} {'tolerant':>9} {'clDice':>8}  (舊vanilla)")
        for m in METHODS:
            per = {k: eval_method(m, k, args.min_area, args.tol, postproc) for k in args.folds}
            def mean(key, folds): return sum(per[k][key] for k in folds) / max(len(folds), 1)
            print(f"{m:22} {mean('van',no2):>8.3f} {mean('tol',no2):>9.3f} {mean('cld',no2):>8.3f}  ({DENSE_VANILLA[m]})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑評估** `$PY eval_lineproc_craq.py --folds 0 1 2 3`
Expected: raw 與 post-proc 兩段,各 method 的 vanilla/tolerant/clDice(排除-fold2 平均)+ 舊 vanilla 對照。
- [ ] **Step 3: 寫 `outputs/craq_lineproc_summary.md`,判定:**
  1. 後處理(CC 濾雜訊)對各 method vanilla/tolerant/clDice 的提升;
  2. 換 clDice/tolerant 後,四個 method 名次/差距 vs 舊 vanilla 是否改變(尤其 PromptedSAM2 滿屏 FP 在 tolerant/clDice 下是否翻盤)。
- [ ] **Step 4(選用) commit**

---

## Self-Review 註記
- Spec §3 三方法 dump→Task 2/3/4;§4 lineproc→Task 1;§5 eval→Task 5;§6 元件全列。
- **修正 spec**:dense-seg dump 用 **raw val(無 CLAHE)** —— 已確認 CLAHE 只在 train_transforms;spec 原寫「套 CLAHE」是錯的,plan 以 raw 為準。
- 一致性:method 名 `denseseg/sepsam_yolo/promptsam2_oracle/promptsam2_yolo`、`preds/{method}/fold{k}/{stem}.png`、`lineproc` 函式名跨 Task 一致。Task 5 的 METHODS 與 dump 輸出對應。
- 跨 env:Task 4 在 SepSAM2 env dump png 到 crack_detection_sam2/preds;Task 5 在 sam2_env 只讀 png。
- fold 對齊:全部用 craqfold{k}/val_masks 當 GT;排除-fold2 並列。
- 已知坑:dense-seg SAM2SemSeg 建構/ckpt key 若不符,Task3 Step2 先對齊 predict_full.py(known-working)。
