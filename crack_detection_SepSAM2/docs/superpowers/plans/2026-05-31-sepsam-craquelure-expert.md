# SepSAM craquelure 專家 — 4-fold CV 實驗 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 SepSAM(YOLOv8-Seg SEA agent + SAM2 CMC)做 craquelure 專家,4-fold CV,得 YOLO-only 與 CMC 的 F1,與既有 dense-seg craquelure expert(~0.63)在同一份 split 上對照,並判定 SAM2 refine 對 craquelure 是否有正貢獻。

**Architecture:** 沿用 `labeled32_craq_v3/tiles_512` 的 craquelure tile 與 `group_split_stem.json`(與 dense-seg expert 同一份 → fold 對齊)。把 craquelure mask 轉成 YOLO-seg 標籤訓 SEA agent,再跑既有 CMC pipeline,逐 fold 評估 YOLO-only / SAM-only / CMC。

**Tech Stack:** Python 3.12 / venv `/home/zzz90/research/SepSAM2_env`、ultralytics 8.4.57(YOLOv8n-seg + C2f_SEA)、SAM2、OpenCV。

> **環境前提:**
> - python: `/home/zzz90/research/SepSAM2_env/bin/python`(簡稱 `$PY`)。cwd: `/home/zzz90/research/crack_detection_SepSAM2/sepsam`。
> - 載入任何 SEA ckpt 前需 `from ultralytics.nn.tasks import C2f_SEA`(各腳本內已處理;互動環境先跑 `python sea_setup.py` 一次)。
> - 非 git repo → commit 步驟為選用(可先 `git init`)。
> - **資料**: `/home/zzz90/research/crack_detection_sam2/data/labeled32_craq_v3/tiles_512/{images,masks}`(images/masks 皆 `.png`,mask 值 0/1)、`group_split_stem.json`(頂層 `folds`,每 fold 有 `fold`/`val_groups`/`train`(png 檔名清單)/`val`)。
> - **對照基準(dense-seg craquelure expert + CLAHE,同 split 同 fold)**: fold0=0.634 / fold1=0.614 / fold2=0.04(崩) / fold3=0.655(craqF1)。fold2 是 domain-shift(彩繪面板),預期 SepSAM 照樣崩 → 同時看「排除 fold2」3-fold 平均。
> - **已知**: `binary_mask_to_yolo_seg.py` 用 `(m>0)` 二值化,**直接吃 0/1 mask,不需 pre-scale**(修正 spec 的顧慮)。

---

## File Structure
**新增:**
- `scripts/build_craq_cv.py` — 由 split 寫各 fold 的 YOLO `train.txt`/`val.txt`、複製 val 的 images/masks 到 `craqfold{k}/{val_images,val_masks}`(供 calibrate/eval 用 dir 介面)、產 `configs/data_craq_cv_fold{k}.yaml`。
- `scripts/eval_craq_cv.py` — 4-fold CMC 評估(逐 fold 用該 fold agent + val),輸出 YOLO-only/SAM-only/CMC 的 P/R/F1/IoU + 對照表。
- `configs/cmc_craq.yaml` — craquelure CMC 超參(Stage 2 校準產出)。
- `configs/data_craq_cv_fold{0..3}.yaml` — 各 fold data yaml(build 產出)。

**修改:**
- `scripts/finetune_heritage_cv.py` — 加 `--data_prefix`(default `data_heritage_cv`),讓同一支腳本能訓 craquelure(用 `data_craq_cv`)。

**標籤產出(指令,非新檔):** 用既有 `scripts/binary_mask_to_yolo_seg.py` 把 masks→`tiles_512/labels/`。

---

## Task 1: Stage 0 — 標籤 + 各 fold 清單/設定

**Files:** Create `scripts/build_craq_cv.py`

- [ ] **Step 1: 產 YOLO-seg 標籤(指令)**

Run:
```bash
/home/zzz90/research/SepSAM2_env/bin/python scripts/binary_mask_to_yolo_seg.py \
  --images /home/zzz90/research/crack_detection_sam2/data/labeled32_craq_v3/tiles_512/images \
  --masks  /home/zzz90/research/crack_detection_sam2/data/labeled32_craq_v3/tiles_512/masks \
  --out    /home/zzz90/research/crack_detection_sam2/data/labeled32_craq_v3/tiles_512/labels \
  --epsilon 0.002 --min_area 64
```
Expected: 印出處理張數與 polygon 數;`.../tiles_512/labels/` 出現 `*.txt`(cls 0 = craquelure)。

- [ ] **Step 2: 實作 `scripts/build_craq_cv.py`**
```python
"""build_craq_cv.py — 由 labeled32_craq_v3 group_split 寫各 fold YOLO 清單 + data yaml,
並複製 val 的 images/masks 到 craqfold{k}/{val_images,val_masks}(供 calibrate/eval 的 dir 介面)。
標籤需先用 binary_mask_to_yolo_seg.py 產到 <tiles>/labels。"""
import argparse, json, os, shutil

DEF_TILES = "/home/zzz90/research/crack_detection_sam2/data/labeled32_craq_v3/tiles_512"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default=DEF_TILES)
    ap.add_argument("--split", default=None)
    ap.add_argument("--configs_dir", default="configs")
    ap.add_argument("--yaml_prefix", default="data_craq_cv")
    args = ap.parse_args()

    tiles = os.path.abspath(args.tiles)
    split = args.split or os.path.join(tiles, "group_split_stem.json")
    img_dir = os.path.join(tiles, "images")
    msk_dir = os.path.join(tiles, "masks")
    cfg_dir = os.path.abspath(args.configs_dir)
    os.makedirs(cfg_dir, exist_ok=True)

    payload = json.load(open(split, encoding="utf-8"))
    for fd in payload["folds"]:
        k = fd["fold"]
        def img_of(name):
            return os.path.join(img_dir, os.path.splitext(name)[0] + ".png")
        train = [img_of(n) for n in fd["train"]]
        val = [img_of(n) for n in fd["val"]]

        fdir = os.path.join(tiles, f"craqfold{k}")
        vi = os.path.join(fdir, "val_images"); vm = os.path.join(fdir, "val_masks")
        os.makedirs(vi, exist_ok=True); os.makedirs(vm, exist_ok=True)
        for n in fd["val"]:
            stem = os.path.splitext(n)[0]
            shutil.copyfile(os.path.join(img_dir, stem + ".png"), os.path.join(vi, stem + ".png"))
            shutil.copyfile(os.path.join(msk_dir, stem + ".png"), os.path.join(vm, stem + ".png"))

        tr = os.path.join(fdir, "train.txt"); va = os.path.join(fdir, "val.txt")
        with open(tr, "w") as f: f.write("\n".join(train) + "\n")
        with open(va, "w") as f: f.write("\n".join(val) + "\n")

        yml = os.path.join(cfg_dir, f"{args.yaml_prefix}_fold{k}.yaml")
        with open(yml, "w", encoding="utf-8") as f:
            f.write(f"# craquelure CV fold {k} (val_groups={fd.get('val_groups')})\n"
                    f"path: {tiles}\ntrain: {tr}\nval: {va}\nnc: 1\nnames:\n  - craquelure\n")
        print(f"fold{k}: train={len(train)} val={len(val)} val_groups={fd.get('val_groups')} -> {yml}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 跑 build + 驗證**
Run: `/home/zzz90/research/SepSAM2_env/bin/python scripts/build_craq_cv.py`
Expected: 4 行 fold 摘要(train/val tile 數、val_groups),產生 `configs/data_craq_cv_fold{0..3}.yaml` 與各 `craqfold{k}/{train.txt,val.txt,val_images/,val_masks/}`。

- [ ] **Step 4: 驗證 ultralytics 能對應 image→label**
Run:
```bash
/home/zzz90/research/SepSAM2_env/bin/python - <<'PY'
import os
tiles="/home/zzz90/research/crack_detection_sam2/data/labeled32_craq_v3/tiles_512"
imgs=[l.strip() for l in open(os.path.join(tiles,"craqfold0/train.txt")) if l.strip()]
lab=lambda p:p.replace("/images/","/labels/").rsplit(".",1)[0]+".txt"
have=sum(os.path.isfile(lab(p)) for p in imgs)
print(f"train imgs={len(imgs)} labels_found={have}")
PY
```
Expected: `labels_found` 接近 `train imgs`(允許少數無前景 tile 無 .txt → 視為背景圖)。

- [ ] **Step 5(選用): commit** `git add scripts/build_craq_cv.py configs/data_craq_cv_fold*.yaml`

---

## Task 2: Stage 1 — 訓 SEA craquelure agent（4 fold）

**Files:** Modify `scripts/finetune_heritage_cv.py`(加 `--data_prefix`)

- [ ] **Step 1: 加 `--data_prefix` 參數**
在 `scripts/finetune_heritage_cv.py` 的 argparse 區,`--configs_dir` 之後加:
```python
    ap.add_argument("--data_prefix", default="data_heritage_cv",
                    help="data yaml 檔名前綴;craquelure 用 data_craq_cv")
```
並把組 data_yaml 的那行(目前 `data_yaml = os.path.join(args.configs_dir, f"data_heritage_cv_fold{k}.yaml")`)改為:
```python
        data_yaml = os.path.join(args.configs_dir, f"{args.data_prefix}_fold{k}.yaml")
```
(其餘不動;預設 `data_heritage_cv` → heritage 行為不變。)

- [ ] **Step 2: 驗證 --help 與預設不變**
Run: `/home/zzz90/research/SepSAM2_env/bin/python scripts/finetune_heritage_cv.py --help`
Expected: 含 `--data_prefix`,default `data_heritage_cv`;無錯誤。

- [ ] **Step 3: 訓 4-fold craquelure agent(背景,長任務)**
init 權重用既有 crack SEA agent(同影像 domain,tile 少,轉移優於 COCO/隨機;**與 spec 寫的 CONT 不同 — 見下方註**)。
Run:
```bash
cd /home/zzz90/research/crack_detection_SepSAM2/sepsam
nohup /home/zzz90/research/SepSAM2_env/bin/python scripts/finetune_heritage_cv.py \
  --folds 0 1 2 3 --data_prefix data_craq_cv \
  --name_prefix sepsam_agent_craq_cv \
  --init_ckpt runs/segment/runs/sepsam_agent_v8n_200ep/weights/best.pt \
  --epochs 150 --imgsz 512 --batch 16 \
  > logs/craq_cv.log 2>&1 &
echo "launched PID $!"
```
Expected: 產生 `runs/segment/runs/sepsam_agent_craq_cv_fold{0..3}/weights/best.pt`;`logs/craq_cv.log` 各 fold 完成。

> **註(spec 偏差):** spec 寫「從 COCO 起訓」。實作改用 crack SEA agent 作 init,理由:同砌體影像 domain、craquelure tile 僅 ~156/ fold,從相關 seg 任務轉移比 COCO/隨機更穩,且直接複用 `finetune_heritage_cv.py`。若要嚴格照 spec,改 `--init_ckpt` 指向 COCO yolov8n-seg-sea 權重即可;建議先用 crack-init 跑,結果不佳再試 COCO。

- [ ] **Step 4(選用): commit** `git add scripts/finetune_heritage_cv.py`

---

## Task 3: Stage 2 — 校準 CMC（craquelure）

**Files:** Create `configs/cmc_craq.yaml`

- [ ] **Step 1: 建初始 `configs/cmc_craq.yaml`**(以 heritage config 為模板,agent 先指向 fold0)
```yaml
# craquelure CMC 超參(2026-05-31 起;Stage 2 校準後更新數值)
YOLO_CONF_1: 0.25
YOLO_CONF_2: 0.50
SAM_THRESH: 0.80
CONFLICTION_RATIO: 0.50
POINTS_DIVISOR: 50
agent_ckpt: runs/segment/runs/sepsam_agent_craq_cv_fold0/weights/best.pt
device: cuda
sam_backend: sam2
sam_model_type: vit_h
sam_ckpt: weights/sam_vit_h_4b8939.pth
sam2_cfg: configs/sam2.1/sam2.1_hiera_b+.yaml
sam2_ckpt: weights/sam2.1_hiera_base_plus.pt
```

- [ ] **Step 2: 在 fold0 val 上掃 SAM_THRESH / CONFLICTION / YOLO_CONF_2**
Run:
```bash
/home/zzz90/research/SepSAM2_env/bin/python scripts/calibrate_sam_thresh.py \
  --config configs/cmc_craq.yaml \
  --images /home/zzz90/research/crack_detection_sam2/data/labeled32_craq_v3/tiles_512/craqfold0/val_images \
  --masks  /home/zzz90/research/crack_detection_sam2/data/labeled32_craq_v3/tiles_512/craqfold0/val_masks \
  --limit 80 --sweep-conflict --sweep-conf2
```
Expected: 印出最佳 `(SAM_THRESH, CONFLICTION_RATIO, YOLO_CONF_2)` 與其平均 F1,以及 YOLO-only / SAM-only F1。
**觀察點:** SAM-only F1 是否 > crack 時的 0.07(若 craquelure 上 SAM2 有效,SAM-only 與 CM>YOLO 會明顯)。conf=0.02 下限(避 NMS-flood)。

- [ ] **Step 3: 把校準最佳值寫回 `configs/cmc_craq.yaml`**
依 Step 2 結果更新 `SAM_THRESH` / `CONFLICTION_RATIO` / `YOLO_CONF_2`(必要時連同 `YOLO_CONF_1`、`POINTS_DIVISOR`)。記錄校準摘要於檔頭註解。

- [ ] **Step 4(選用): commit** `git add configs/cmc_craq.yaml`

### 校準結果(Step 2/3 後填)
- SAM_THRESH=`__` CONFLICTION_RATIO=`__` YOLO_CONF_2=`__`;fold0 val: YOLO-only F1=`__` SAM-only F1=`__` best CMC F1=`__`

---

## Task 4: Stage 3 — 4-fold CMC 評估與對照

**Files:** Create `scripts/eval_craq_cv.py`

- [ ] **Step 1: 實作 `scripts/eval_craq_cv.py`**
```python
"""eval_craq_cv.py — 逐 fold 用該 fold 的 craquelure agent + cmc_craq.yaml 在 fold val 上跑 CMC,
輸出 YOLO-only / SAM-only / CMC 的 P/R/F1/IoU,並與 dense-seg expert 對照。"""
import argparse, glob, os, sys
import cv2, numpy as np, yaml
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import Agent
from src.cmc import conflict_ratio
from src.filters import contour_filter
from src.geometry import mask_to_points_and_width
from src.large_model import build_large_model
from src.metrics import prf_iou, aggregate

TILES = "/home/zzz90/research/crack_detection_sam2/data/labeled32_craq_v3/tiles_512"
# dense-seg expert(+CLAHE)per-fold craqF1 對照
DENSE = {0: 0.634, 1: 0.614, 2: 0.040, 3: 0.655}


def eval_fold(k, hp_dict):
    hp = SimpleNamespace(**hp_dict)
    hp.agent_ckpt = f"runs/segment/runs/sepsam_agent_craq_cv_fold{k}/weights/best.pt"
    from ultralytics.nn.tasks import C2f_SEA  # noqa: F401
    agent = Agent(hp.agent_ckpt, device=hp.device)
    predictor, prompt_fn = build_large_model(hp)
    vi = os.path.join(TILES, f"craqfold{k}", "val_images")
    vm = os.path.join(TILES, f"craqfold{k}", "val_masks")
    stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(vi, "*.png")))
    ry, rs, rc = [], [], []
    n_sam = 0
    for st in stems:
        bgr = cv2.imread(os.path.join(vi, st + ".png"))
        gt = cv2.imread(os.path.join(vm, st + ".png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); gtb = gt > 0
        mask_y, yc = agent.predict(rgb, conf=hp.YOLO_CONF_1)
        n_pts = max(rgb.shape[:2]) // hp.POINTS_DIVISOR
        pts, _ = mask_to_points_and_width(mask_y > 0, n_pts)
        m_raw, score = prompt_fn(predictor, rgb, pts)
        m_s = contour_filter(m_raw, yc, hp.YOLO_CONF_2)
        cr = conflict_ratio(m_s, mask_y)
        accept = (cr < hp.CONFLICTION_RATIO) and (score > hp.SAM_THRESH)
        m_c = m_s if accept else mask_y
        n_sam += int(accept)
        ry.append(prf_iou(mask_y, gtb)); rs.append(prf_iou(m_s, gtb)); rc.append(prf_iou(m_c, gtb))
    return aggregate(ry), aggregate(rs), aggregate(rc), n_sam, len(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cmc_craq.yaml")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = ap.parse_args()
    hp_dict = yaml.safe_load(open(args.config, encoding="utf-8"))
    rows = []
    for k in args.folds:
        ry, rs, rc, n_sam, n = eval_fold(k, hp_dict)
        rows.append((k, ry, rs, rc, n_sam, n))
        print(f"fold{k} n={n} SAM採用={n_sam}/{n} | "
              f"YOLO F1={ry['F1']:.3f} | SAM F1={rs['F1']:.3f} | CMC F1={rc['F1']:.3f} "
              f"| dense-seg={DENSE.get(k,'?')}", flush=True)
    def mean(key, idx, folds=None):
        sel = [r for r in rows if folds is None or r[0] in folds]
        return sum(r[idx][key] for r in sel) / max(len(sel), 1)
    no2 = [k for k in args.folds if k != 2]
    print("\n==== 4-fold 平均 ====")
    print(f"YOLO-only F1={mean('F1',1):.3f}  SAM-only F1={mean('F1',2):.3f}  CMC F1={mean('F1',3):.3f}")
    print(f"排除fold2: YOLO F1={mean('F1',1,no2):.3f}  CMC F1={mean('F1',3,no2):.3f}  (dense-seg≈0.634/0.614/0.655)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 4-fold 評估**
Run: `/home/zzz90/research/SepSAM2_env/bin/python scripts/eval_craq_cv.py --config configs/cmc_craq.yaml`
Expected: 每 fold 一行(YOLO/SAM/CMC F1 + SAM 採用比例 + dense-seg 對照),最後 4-fold 與排除 fold2 平均。

- [ ] **Step 3: 判定假說 + 寫總結**
寫 `runs/craq_sepsam_summary.md`:per-fold 與平均的 YOLO-only / SAM-only / CMC vs dense-seg;明確結論:
  1. SepSAM(YOLO-only 或 CMC)排除 fold2 平均是否接近/超過 dense-seg ~0.63;
  2. **CMC F1 是否 > YOLO-only F1**(SAM2 refine 對 craquelure 有無正貢獻 — 本實驗核心)。

- [ ] **Step 4(選用): commit** `git add scripts/eval_craq_cv.py runs/craq_sepsam_summary.md`

---

## Self-Review 註記
- Spec Stage 0-3 → Task 1/2/3/4 全覆蓋。
- 0/1 mask:已確認 `binary_mask_to_yolo_seg.py` 用 `>0`,直接可用(修正 spec 顧慮)。
- spec「COCO 起訓」偏差:Task 2 Step 3 改用 crack SEA agent init 並註明理由與還原方式。
- fold 對齊:沿用 labeled32_craq_v3 split,dense-seg 對照值(0.634/0.614/0.04/0.655)逐 fold 對應;fold2 以「排除 fold2」並列。
- 一致性:`data_craq_cv` / `sepsam_agent_craq_cv` / `craqfold{k}/val_images,val_masks` / `cmc_craq.yaml` 命名跨 Task 一致。
- 假說檢驗(CMC vs YOLO-only)在 Task 4 明確輸出。
