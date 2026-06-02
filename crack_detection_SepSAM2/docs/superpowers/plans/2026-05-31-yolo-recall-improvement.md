# SepSAM2 YOLO 召回提升 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用分階段量化的方式提升 SepSAM2 YOLO agent 的裂紋召回(免費槓桿 conf/NMS → CLAHE 重訓 → 密集面板過採樣),先在 4-fold CV 驗證、再套到生產 agent。

**Architecture:** heritage domain 的 SAM2 已校準停用,CMC ≡ YOLO-only,因此只動 YOLO。共用一份確定性 CLAHE 前處理(train tile 建構與 CMC 推論端共用,避免不一致)。每個槓桿對照前一階段做 A/B,以 mask recall 為主指標、mask F1 為護欄。

**Tech Stack:** Python 3.12 / venv `/home/zzz90/research/SepSAM2_env`、ultralytics 8.4.57(YOLOv8n-seg + C2f_SEA)、OpenCV、numpy。

> **環境前提:**
> - 所有 python 指令用 `/home/zzz90/research/SepSAM2_env/bin/python`(以下簡稱 `$PY`)。
> - cwd 一律 `/home/zzz90/research/crack_detection_SepSAM2/sepsam`。
> - 載入 agent ckpt 前環境需已 `import` 過 C2f_SEA(`from ultralytics.nn.tasks import C2f_SEA`),否則 unpickle 失敗。各腳本內已處理。
> - **本專案非 git repo**,故每個 Task 結尾的 commit 為「選用」:若要逐 Task commit,先在 Task 0 執行 `git init`;不想用 git 就略過所有 commit 步驟,其餘步驟不受影響。
> - 既有基準(Stage 0,已測得):conf=0.25 4-fold 平均 mask F1=0.171 / recall=0.140;max-F1 conf 平均 mask F1=0.215。各 fold best.pt 在 `runs/segment/runs/sepsam_agent_heritage_cv_fold{0..3}/weights/best.pt`。
> - **fold2 是 domain-shift outlier(彩繪面板),預期這些槓桿救不了它**;所有比較同時看「4-fold 平均」與「排除 fold2 的 3-fold 平均」。

---

## File Structure

**新增:**
- `src/preprocess.py` — 共用 CLAHE 前處理:`clahe_rgb(img_rgb, clip, grid)`(LAB 的 L 通道)。tile 建構與 CMC 推論端共用同一份。
- `scripts/sweep_cv_recall.py` — Stage 1:在各 fold val 上掃 conf/iou/max_det,輸出 csv + markdown 表。
- `scripts/oversample_dense_panels.py` — Stage 3:依 panel 實例數過採樣某 fold 的 train.txt,產生新的 train list + data yaml。
- `tests/test_preprocess.py` — CLAHE 形狀/確定性/對比上升的單元測試(plain assert,不依賴 pytest)。
- `tests/test_oversample.py` — 過採樣倍率公式單元測試。

**修改:**
- `scripts/build_heritage_cv.py` — 加 `--clahe`(+ `--clahe-clip`、`--clahe-grid`)旗標,切 tile 前對來源圖套 `clahe_rgb`。
- `scripts/eval_heritage_before_after.py`、`scripts/infer.py` — Stage 4:在 `agent.predict(rgb, ...)` 前依 `hp.clahe` 套相同 CLAHE。
- `configs/cmc_heritage.yaml` — Stage 4:更新 `YOLO_CONF_1`,新增 `clahe` / `clahe_clip` / `clahe_grid` 欄位與 `agent_ckpt`。

---

## Task 0(選用): 啟用 git 以便逐 Task commit

**Files:** 無(repo 設定)

- [ ] **Step 1: 若要用 git,初始化**

Run:
```bash
cd /home/zzz90/research/crack_detection_SepSAM2/sepsam
git init && printf "runs/\noutputs/\ndatasets/\n*.pt\n__pycache__/\n" > .gitignore
git add -A && git commit -m "chore: init repo for recall-improvement work"
```
若不使用 git,跳過本 Task 與後續所有 commit 步驟。

---

## Task 1: 共用 CLAHE 前處理模組

**Files:**
- Create: `src/preprocess.py`
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: 寫 failing test**

Create `tests/test_preprocess.py`:
```python
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import clahe_rgb


def test_shape_dtype_preserved():
    img = (np.random.default_rng(0).integers(90, 110, (64, 64, 3))).astype(np.uint8)
    out = clahe_rgb(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_deterministic():
    img = (np.random.default_rng(1).integers(90, 110, (64, 64, 3))).astype(np.uint8)
    a = clahe_rgb(img)
    b = clahe_rgb(img)
    assert np.array_equal(a, b)


def test_contrast_increases_on_low_contrast():
    # 低對比影像(值集中在 100..110)→ CLAHE 後 L 通道 std 應上升
    img = (np.random.default_rng(2).integers(100, 110, (128, 128, 3))).astype(np.uint8)
    out = clahe_rgb(img, clip=3.0, grid=8)
    assert out.std() > img.std()


if __name__ == "__main__":
    test_shape_dtype_preserved()
    test_deterministic()
    test_contrast_increases_on_low_contrast()
    print("OK test_preprocess")
```

- [ ] **Step 2: 跑 test 確認失敗**

Run: `/home/zzz90/research/SepSAM2_env/bin/python tests/test_preprocess.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.preprocess'`

- [ ] **Step 3: 實作 `src/preprocess.py`**

```python
"""preprocess.py — 共用確定性影像前處理。train tile 建構與 CMC 推論端共用,
確保 CLAHE 在訓練與推論一致(避免 domain 不一致)。"""
import cv2
import numpy as np


def clahe_rgb(img_rgb, clip=2.0, grid=8):
    """對 RGB 影像在 LAB 的 L 通道套 CLAHE(確定性)。

    Args:
        img_rgb: HxWx3 uint8 RGB
        clip:    clipLimit
        grid:    tileGridSize 邊長(grid x grid)
    Returns:
        HxWx3 uint8 RGB
    """
    img_rgb = np.asarray(img_rgb)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def clahe_bgr(img_bgr, clip=2.0, grid=8):
    """BGR 版包裝(cv2.imread 預設 BGR)。"""
    rgb = cv2.cvtColor(np.asarray(img_bgr), cv2.COLOR_BGR2RGB)
    out = clahe_rgb(rgb, clip=clip, grid=grid)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
```

- [ ] **Step 4: 跑 test 確認通過**

Run: `/home/zzz90/research/SepSAM2_env/bin/python tests/test_preprocess.py`
Expected: `OK test_preprocess`

- [ ] **Step 5(選用): commit**

```bash
git add src/preprocess.py tests/test_preprocess.py && git commit -m "feat: shared deterministic CLAHE preprocessing"
```

---

## Task 2: Stage 1 — conf/NMS/max_det 免費槓桿掃描

**Files:**
- Create: `scripts/sweep_cv_recall.py`
- Output: `runs/cv_recall_sweep.csv`、`runs/cv_recall_sweep.md`

- [ ] **Step 1: 實作 sweep 腳本**

Create `scripts/sweep_cv_recall.py`:
```python
"""sweep_cv_recall.py — Stage 1:在 4-fold CV val 上掃 conf/iou/max_det,
量化免費槓桿對 box/mask P/R/F1 的影響,輸出 csv + markdown 表。

用法:
  $PY scripts/sweep_cv_recall.py --folds 0 1 2 3
"""
import argparse, csv, itertools, os


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--conf", type=float, nargs="+", default=[0.25, 0.10, 0.05, 0.02])
    ap.add_argument("--iou", type=float, nargs="+", default=[0.7, 0.5, 0.4])
    ap.add_argument("--max_det", type=int, nargs="+", default=[300, 1000])
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--run_root", default="runs/segment/runs")
    ap.add_argument("--ckpt_prefix", default="sepsam_agent_heritage_cv",
                    help="run 目錄前綴;CLAHE 版傳 sepsam_agent_heritage_cv_clahe,過採樣版傳 sepsam_agent_heritage_cv_os")
    ap.add_argument("--out_csv", default="runs/cv_recall_sweep.csv")
    ap.add_argument("--out_md", default="runs/cv_recall_sweep.md")
    args = ap.parse_args()

    from ultralytics.nn.tasks import C2f_SEA  # noqa: F401  (註冊)
    from ultralytics import YOLO

    rows = []
    for conf, iou, md in itertools.product(args.conf, args.iou, args.max_det):
        per_fold = []
        for k in args.folds:
            w = f"{args.run_root}/{args.ckpt_prefix}_fold{k}/weights/best.pt"
            data = f"configs/data_heritage_cv_fold{k}.yaml"
            m = YOLO(w)
            r = m.val(data=data, conf=conf, iou=iou, max_det=md, imgsz=args.imgsz,
                      batch=16, device="0", plots=False, verbose=False, split="val")
            per_fold.append({
                "fold": k,
                "boxP": float(r.box.mp), "boxR": float(r.box.mr),
                "maskP": float(r.seg.mp), "maskR": float(r.seg.mr),
            })
        for pf in per_fold:
            pf["boxF1"] = f1(pf["boxP"], pf["boxR"])
            pf["maskF1"] = f1(pf["maskP"], pf["maskR"])
        def mean(key, folds=None):
            sel = [p for p in per_fold if folds is None or p["fold"] in folds]
            return sum(p[key] for p in sel) / max(len(sel), 1)
        no2 = [k for k in args.folds if k != 2]
        rows.append({
            "conf": conf, "iou": iou, "max_det": md,
            "maskR_4f": mean("maskR"), "maskF1_4f": mean("maskF1"),
            "maskP_4f": mean("maskP"),
            "maskR_no2": mean("maskR", no2), "maskF1_no2": mean("maskF1", no2),
            "boxR_4f": mean("boxR"), "boxF1_4f": mean("boxF1"),
        })
        print(f"conf={conf} iou={iou} max_det={md} -> "
              f"maskR_4f={rows[-1]['maskR_4f']:.3f} maskF1_4f={rows[-1]['maskF1_4f']:.3f} "
              f"maskF1_no2={rows[-1]['maskF1_no2']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)
    with open(args.out_md, "w") as f:
        f.write("| conf | iou | max_det | maskR_4f | maskF1_4f | maskP_4f | maskR_no2 | maskF1_no2 |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['conf']} | {r['iou']} | {r['max_det']} | {r['maskR_4f']:.3f} | "
                    f"{r['maskF1_4f']:.3f} | {r['maskP_4f']:.3f} | {r['maskR_no2']:.3f} | "
                    f"{r['maskF1_no2']:.3f} |\n")
    print(f"\nwritten {args.out_csv} / {args.out_md}  ({len(rows)} combos)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑掃描**

Run: `/home/zzz90/research/SepSAM2_env/bin/python scripts/sweep_cv_recall.py --folds 0 1 2 3`
Expected: 印出每個 combo 的 maskR/maskF1,最後寫出 `runs/cv_recall_sweep.csv` 與 `.md`(24 組合)。

- [ ] **Step 3: 選操作點**

開 `runs/cv_recall_sweep.md`,挑選**在 maskP_4f ≥ 0.15 前提下使 maskR_4f 最大**的 (conf, iou, max_det)。
把選定值記到本 plan 下方「選定操作點」區塊(供 Stage 2/3/4 沿用)。
Expected: 確認降 conf 提升 maskR;選定一組,例如 conf=0.05 / iou=0.5 / max_det=1000(實際依表)。

- [ ] **Step 4(選用): commit**

```bash
git add scripts/sweep_cv_recall.py runs/cv_recall_sweep.* && git commit -m "feat: stage1 conf/nms/max_det recall sweep"
```

### 選定操作點(Step 3 已測,2026-05-31)
- **conf = 0.10、iou = 0.5、max_det = 300**(依「precision 地板下最大化 recall」規則選出)
- 此操作點:maskR_4f=0.215 / maskF1_4f=0.219 / maskP_4f=0.223 / maskR_no2=0.276 / maskF1_no2=0.279
- vs Stage 0 conf0.25:recall 0.140→0.215(+54% 相對),F1 0.171→0.219。**免費槓桿已達成成功標準(recall↑ 且 F1≥0.215)。**
- 重要觀察:max_det 300 vs 1000 結果完全相同(未觸頂);conf 再降到 0.05/0.02 recall 反而略降(0.215→0.208)且 precision 下降,故 conf=0.10 是甜蜜點,不取更低;iou=0.4 給最高 F1(0.222)但 recall 略低(0.212),取 iou=0.5 以 recall 為主。

---

## Task 3: Stage 2 — CLAHE 重訓

**Files:**
- Modify: `scripts/build_heritage_cv.py`(加 `--clahe`)
- Output: `datasets/heritage_ft_cv_clahe/`、`runs/segment/runs/sepsam_agent_heritage_cv_clahe_fold{0..3}/`

- [ ] **Step 1: 在 build_heritage_cv.py 接上 CLAHE**

Modify `scripts/build_heritage_cv.py`。在頂部 import 區(第 28-29 行 `import cv2` / `import numpy as np` 之後)加入:
```python
sys.path.insert(0, PROJECT_ROOT)  # 讓 src 可被 import
from src.preprocess import clahe_bgr  # noqa: E402
```
(注意:`PROJECT_ROOT` 已在第 32 行定義,此 import 放在第 33 行 `sys.path.insert(0, SCRIPT_DIR)` 之後即可。)

把 `build_pool` 的簽名(第 66 行)改為帶 CLAHE 參數:
```python
def build_pool(src, splits, pool, tile, stride, min_area, epsilon, keep_empty,
               clahe=False, clahe_clip=2.0, clahe_grid=8):
```
在 `img = cv2.imread(imgs[stem])` 與形狀檢查之後、開始切 tile(`H, W = img.shape[:2]`)之前,插入:
```python
            if clahe:
                img = clahe_bgr(img, clip=clahe_clip, grid=clahe_grid)
```

在 `main()` 的 argparse 區(第 126 行 `--seed` 之後)加:
```python
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--clahe-clip", type=float, default=2.0, dest="clahe_clip")
    ap.add_argument("--clahe-grid", type=int, default=8, dest="clahe_grid")
```
並把呼叫(第 135-136 行)改為:
```python
    tile_stems = build_pool(src, args.splits, pool, args.tile, args.stride,
                            args.min_area, args.epsilon, args.keep_empty,
                            clahe=args.clahe, clahe_clip=args.clahe_clip,
                            clahe_grid=args.clahe_grid)
```

- [ ] **Step 2: 驗證改動沒破壞 import / argparse**

Run: `/home/zzz90/research/SepSAM2_env/bin/python scripts/build_heritage_cv.py --help`
Expected: help 內含 `--clahe`、`--clahe-clip`、`--clahe-grid`,無 ImportError。

- [ ] **Step 3: 建 CLAHE 版 CV pool(沿用相同 panel 分組)**

Run:
```bash
/home/zzz90/research/SepSAM2_env/bin/python scripts/build_heritage_cv.py \
  --src datasets/heritage_ft --pool datasets/heritage_ft_cv_clahe \
  --configs_dir configs --tile 512 --stride 256 --n_splits 4 --seed 42 --clahe
```
Expected: 印出 `panels(4)` 與各 fold tile 數(train 216 / val 72,與非 CLAHE 版一致 → seed 相同保證分組相同),產生 `configs/data_heritage_cv_fold{0..3}.yaml`。

> ⚠️ 注意:上面會**覆寫** `configs/data_heritage_cv_fold*.yaml` 指向 CLAHE pool。為保留原始 yaml,先備份:
> ```bash
> for k in 0 1 2 3; do cp configs/data_heritage_cv_fold$k.yaml configs/data_heritage_cv_fold$k.orig.yaml; done
> ```
> 在 build 後,把 CLAHE 版 yaml 改名以免之後混淆:
> ```bash
> for k in 0 1 2 3; do mv configs/data_heritage_cv_fold$k.yaml configs/data_heritage_cv_clahe_fold$k.yaml; mv configs/data_heritage_cv_fold$k.orig.yaml configs/data_heritage_cv_fold$k.yaml; done
> ```

- [ ] **Step 4: 確認分組與非 CLAHE 版完全一致(乾淨 A/B)**

Run:
```bash
/home/zzz90/research/SepSAM2_env/bin/python - <<'PY'
import json
a=json.load(open("datasets/heritage_ft_cv/folds.json"))
b=json.load(open("datasets/heritage_ft_cv_clahe/folds.json"))
for fa,fb in zip(a["folds"],b["folds"]):
    assert fa["val_panels"]==fb["val_panels"], (fa["val_panels"],fb["val_panels"])
print("OK 分組一致")
PY
```
Expected: `OK 分組一致`

- [ ] **Step 5: 重訓 CLAHE 4 fold**

把 `finetune_heritage_cv.py` 指向 CLAHE 的 data yaml。該腳本用固定檔名 `data_heritage_cv_fold{k}.yaml`,故用 `--configs_dir` 無法切換命名;改為直接傳入 prefix。檢視 `finetune_heritage_cv.py` 第 41 行:它組 `data_heritage_cv_fold{k}.yaml`。最省事做法:訓練時暫時把 CLAHE yaml 換回標準名。執行:
```bash
cd /home/zzz90/research/crack_detection_SepSAM2/sepsam
for k in 0 1 2 3; do cp configs/data_heritage_cv_clahe_fold$k.yaml configs/data_heritage_cv_fold$k.yaml.bak_std 2>/dev/null; done
# 用 clahe yaml 覆蓋標準名來訓練,訓練後還原
for k in 0 1 2 3; do cp configs/data_heritage_cv_fold$k.yaml /tmp/std_fold$k.yaml; cp configs/data_heritage_cv_clahe_fold$k.yaml configs/data_heritage_cv_fold$k.yaml; done
/home/zzz90/research/SepSAM2_env/bin/python scripts/finetune_heritage_cv.py \
  --folds 0 1 2 3 --name_prefix sepsam_agent_heritage_cv_clahe \
  > logs/heritage_cv_clahe.log 2>&1
for k in 0 1 2 3; do cp /tmp/std_fold$k.yaml configs/data_heritage_cv_fold$k.yaml; done
```
Expected: `logs/heritage_cv_clahe.log` 各 fold 完成,產生 `runs/segment/runs/sepsam_agent_heritage_cv_clahe_fold{0..3}/weights/best.pt`。
(訓練是長時間任務:用背景執行,完成後再進下一步。)

- [ ] **Step 6: 用 Stage-1 操作點評估 CLAHE 版,比較增量**

Run(把 `<conf>/<iou>/<md>` 換成 Task 2 選定值):
```bash
/home/zzz90/research/SepSAM2_env/bin/python scripts/sweep_cv_recall.py \
  --folds 0 1 2 3 --conf <conf> --iou <iou> --max_det <md> \
  --ckpt_prefix sepsam_agent_heritage_cv_clahe \
  --run_root runs/segment/runs --out_csv runs/cv_recall_clahe.csv --out_md runs/cv_recall_clahe.md
```
> 注意:`sweep_cv_recall.py` 已有 `--ckpt_prefix` 參數(Task 2 已實作),評 CLAHE 版直接傳 `--ckpt_prefix sepsam_agent_heritage_cv_clahe` 即可。
Expected: 得到 CLAHE 版 maskR_4f / maskF1_4f / maskF1_no2,與 Task 2 基準操作點對照,記錄 recall/F1 增量。

- [ ] **Step 7(選用): commit**

```bash
git add scripts/build_heritage_cv.py scripts/sweep_cv_recall.py configs/data_heritage_cv_clahe_fold*.yaml runs/cv_recall_clahe.* logs/heritage_cv_clahe.log && git commit -m "feat: stage2 CLAHE-baked tiles + retrain + eval"
```

---

## Task 4: Stage 3 — 密集面板過採樣重訓

**Files:**
- Create: `scripts/oversample_dense_panels.py`
- Test: `tests/test_oversample.py`
- Output: `datasets/<pool>/fold{k}/train_oversampled.txt`、`configs/data_heritage_cv_os_fold{k}.yaml`、`runs/segment/runs/sepsam_agent_heritage_cv_os_fold{0..3}/`

- [ ] **Step 1: 寫倍率公式 failing test**

Create `tests/test_oversample.py`:
```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.oversample_dense_panels import repeat_factor


def test_repeat_factor_formula():
    # median=100: 50->1(round0.5=0 →clip 下限1), 100->1, 250->3(round2.5=2→ wait)
    # 公式 clip(round(n/median),1,3)
    assert repeat_factor(50, 100) == 1     # round(0.5)=0 -> clip ->1
    assert repeat_factor(100, 100) == 1    # round(1.0)=1
    assert repeat_factor(150, 100) == 2    # round(1.5)=2
    assert repeat_factor(250, 100) == 2    # round(2.5)=2 (banker's? 用一般 round)
    assert repeat_factor(350, 100) == 3    # round(3.5)=4 -> clip ->3
    assert repeat_factor(1000, 100) == 3   # clip 上限 3


if __name__ == "__main__":
    test_repeat_factor_formula()
    print("OK test_oversample")
```
> 註:Python 內建 `round` 用 banker's rounding(round(2.5)=2, round(3.5)=4)。本測試已對齊該行為,實作直接用內建 `round`。

- [ ] **Step 2: 跑 test 確認失敗**

Run: `/home/zzz90/research/SepSAM2_env/bin/python tests/test_oversample.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.oversample_dense_panels'`

- [ ] **Step 3: 實作 `scripts/oversample_dense_panels.py`**

```python
"""oversample_dense_panels.py — Stage 3:依 panel 裂紋實例數過採樣某 fold 的 train list。

倍率公式:repeat = clip(round(panel_instances / median_panel_instances), 1, 3)
(panel_instances = 該 panel 所有 train tile 的 label 行數總和;median 取各 train panel 的中位數)

用法:
  $PY scripts/oversample_dense_panels.py --pool datasets/heritage_ft_cv_clahe --fold 0 \
      --configs_dir configs --yaml_prefix data_heritage_cv_os
"""
import argparse, os, re, statistics, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from build_heritage_cv import tile_to_panel  # noqa: E402


def repeat_factor(n, median):
    if median <= 0:
        return 1
    return int(max(1, min(3, round(n / median))))


def tile_instance_count(label_path):
    if not os.path.isfile(label_path):
        return 0
    with open(label_path) as f:
        return sum(1 for line in f if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--configs_dir", default="configs")
    ap.add_argument("--yaml_prefix", default="data_heritage_cv_os")
    args = ap.parse_args()

    pool = os.path.abspath(args.pool)
    lbl_dir = os.path.join(pool, "labels")
    train_txt = os.path.join(pool, f"fold{args.fold}", "train.txt")
    img_paths = [l.strip() for l in open(train_txt) if l.strip()]

    # 每個 tile 的 panel 與實例數
    panel_inst, panel_tiles = {}, {}
    for ip in img_paths:
        stem = os.path.splitext(os.path.basename(ip))[0]
        panel = tile_to_panel(stem)
        n = tile_instance_count(os.path.join(lbl_dir, stem + ".txt"))
        panel_inst[panel] = panel_inst.get(panel, 0) + n
        panel_tiles.setdefault(panel, []).append(ip)

    med = statistics.median(panel_inst.values())
    out_lines = []
    for panel, tiles in panel_tiles.items():
        rep = repeat_factor(panel_inst[panel], med)
        out_lines += tiles * rep
        print(f"panel {panel}: inst={panel_inst[panel]} rep={rep} ({len(tiles)} tiles)")

    out_txt = os.path.join(pool, f"fold{args.fold}", "train_oversampled.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    # 產生對應 data yaml(val 不變)
    val_txt = os.path.join(pool, f"fold{args.fold}", "val.txt")
    yaml_path = os.path.join(os.path.abspath(args.configs_dir),
                             f"{args.yaml_prefix}_fold{args.fold}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"# oversampled train (Stage 3) fold {args.fold}, median_inst={med}\n"
                f"path: {pool}\ntrain: {out_txt}\nval: {val_txt}\nnc: 1\nnames:\n  - crack\n")
    print(f"median_inst={med}  train {len(img_paths)}->{len(out_lines)} tiles  yaml={yaml_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑 test 確認通過**

Run: `/home/zzz90/research/SepSAM2_env/bin/python tests/test_oversample.py`
Expected: `OK test_oversample`

- [ ] **Step 5: 對 4 fold 產生過採樣 train list(用 Stage 2 較佳的 pool)**

> `<POOL>` = Stage 2 結論中較佳者:CLAHE 有效則用 `datasets/heritage_ft_cv_clahe`,否則 `datasets/heritage_ft_cv`。
Run:
```bash
for k in 0 1 2 3; do
  /home/zzz90/research/SepSAM2_env/bin/python scripts/oversample_dense_panels.py \
    --pool <POOL> --fold $k --configs_dir configs --yaml_prefix data_heritage_cv_os
done
```
Expected: 各 fold 印出每 panel 的 inst/rep 與 train tile 數從 216 增加;產生 `configs/data_heritage_cv_os_fold{0..3}.yaml`。

- [ ] **Step 6: 重訓過採樣 4 fold**

同 Task 3 Step 5 的暫時換名手法,用 `data_heritage_cv_os_fold{k}.yaml`:
```bash
cd /home/zzz90/research/crack_detection_SepSAM2/sepsam
for k in 0 1 2 3; do cp configs/data_heritage_cv_fold$k.yaml /tmp/std_fold$k.yaml; cp configs/data_heritage_cv_os_fold$k.yaml configs/data_heritage_cv_fold$k.yaml; done
/home/zzz90/research/SepSAM2_env/bin/python scripts/finetune_heritage_cv.py \
  --folds 0 1 2 3 --name_prefix sepsam_agent_heritage_cv_os \
  > logs/heritage_cv_os.log 2>&1
for k in 0 1 2 3; do cp /tmp/std_fold$k.yaml configs/data_heritage_cv_fold$k.yaml; done
```
Expected: 產生 `runs/segment/runs/sepsam_agent_heritage_cv_os_fold{0..3}/weights/best.pt`。

- [ ] **Step 7: 評估過採樣版,比較增量**

Run(操作點同 Task 2;`--ckpt_prefix` 來自 Task 3 Step 6 加的參數):
```bash
/home/zzz90/research/SepSAM2_env/bin/python scripts/sweep_cv_recall.py \
  --folds 0 1 2 3 --conf <conf> --iou <iou> --max_det <md> \
  --ckpt_prefix sepsam_agent_heritage_cv_os \
  --out_csv runs/cv_recall_os.csv --out_md runs/cv_recall_os.md
```
Expected: 得過採樣版 maskR/maskF1,與 baseline、CLAHE 版三方對照;記錄是否在守住 F1(≥0.215)下提升 recall。**注意 fold2 不期待改善,看 maskF1_no2。**

- [ ] **Step 8(選用): commit**

```bash
git add scripts/oversample_dense_panels.py tests/test_oversample.py configs/data_heritage_cv_os_fold*.yaml runs/cv_recall_os.* logs/heritage_cv_os.log && git commit -m "feat: stage3 dense-panel oversampling + retrain + eval"
```

---

## Task 5: Stage 4 — 套到生產 agent + CMC 推論端同步 CLAHE

**Files:**
- Modify: `scripts/eval_heritage_before_after.py`、`scripts/infer.py`、`configs/cmc_heritage.yaml`
- Output: 新生產 agent ckpt、`heritage_1_31test` 重評結果

- [ ] **Step 1: CMC 推論端加 CLAHE(與訓練一致)**

在 `scripts/eval_heritage_before_after.py` 與 `scripts/infer.py` 頂部 import 區加:
```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clahe_rgb
```
在每個讀進 RGB 影像、即將呼叫 `agent.predict(rgb, ...)` 或 `cmc_predict*(rgb, ...)` 的位置之前(eval_heritage_before_after.py 第 52 行 `mask_y, yc = agent.predict(rgb, ...)` 之前;infer.py 對應的 cmc_predict 呼叫之前),插入:
```python
        if getattr(hp, "clahe", False):
            rgb = clahe_rgb(rgb, clip=getattr(hp, "clahe_clip", 2.0), grid=getattr(hp, "clahe_grid", 8))
```
> 確保是在 `set_image` / agent 之前套用,且 mask GT 不套 CLAHE(只處理輸入影像)。

- [ ] **Step 2: 更新 cmc_heritage.yaml**

在 `configs/cmc_heritage.yaml` 加入(Stage 1 選定 conf 與 CLAHE 參數):
```yaml
clahe: true
clahe_clip: 2.0
clahe_grid: 8
```
並把 `YOLO_CONF_1` 改為 Task 2 選定值,`agent_ckpt` 改為 Step 4 訓練出的生產 agent 權重路徑。

- [ ] **Step 3: 訓練生產 agent(CLAHE 全量 + 過採樣,依 Stage 2/3 結論)**

用全量 heritage 資料(非 CV 切分)以贏的組合重訓。沿用 `finetune_heritage.py`(全量微調腳本),先建 CLAHE 全量 tile:
```bash
# (a) 建 CLAHE 全量 tile pool(若 finetune_heritage.py 吃既有 tile,套相同 --clahe 流程建立)
# (b) 以 CLAHE 資料微調,輸出 sepsam_agent_heritage_ft_clahe
```
> 具體指令依 `finetune_heritage.py` 的介面(本 Task 執行時先 `Read` 該檔確認旗標),原則:輸入用 CLAHE tile、conf 評估用 Stage-1 操作點。
Expected: 產生 `runs/.../sepsam_agent_heritage_ft_clahe/weights/best.pt`。

- [ ] **Step 4: 在 heritage_1_31test 重評,對照 0.434 基準**

Run:
```bash
/home/zzz90/research/SepSAM2_env/bin/python scripts/eval_heritage_before_after.py \
  --config configs/cmc_heritage.yaml --images datasets/heritage_1_31test/images \
  --masks datasets/heritage_1_31test/masks
```
> 執行前先 `Read scripts/eval_heritage_before_after.py` 確認其 CLI 旗標名稱與輸出位置。
Expected: 得到新的 YOLO-only(CMC≡YOLO)P/R/F1,對照 cmc_heritage 註解的 0.434;確認 recall 與 F1 是否提升。

- [ ] **Step 5: 記錄最終結論**

把四階段(baseline → +conf/NMS → +CLAHE → +過採樣 → 生產)的 recall/F1 整理成一張表,寫進 `runs/recall_improvement_summary.md`,fold2 與 3-fold(排除 fold2)分開列。

- [ ] **Step 6(選用): commit**

```bash
git add scripts/eval_heritage_before_after.py scripts/infer.py configs/cmc_heritage.yaml runs/recall_improvement_summary.md && git commit -m "feat: stage4 production agent CLAHE + conf update + final eval"
```

---

## Self-Review 註記
- Spec 四個 stage 全部對應到 Task 2/3/4/5;Stage 0 baseline 在 header 記錄。
- 成功標準(recall↑ 且 mask F1≥0.215)在 Task 2 Step 3、Task 3 Step 6、Task 4 Step 7、Task 5 Step 4 反覆檢核。
- fold2 domain-shift 在 header 與每個評估步驟以「3-fold(排除 fold2)」並列處理。
- CLAHE train/infer 一致性:同一份 `src/preprocess.py` 被 build_heritage_cv(Task 3)與 CMC 推論端(Task 5 Step 1)共用。
- 已知坑:conf 掃描下限 0.02 不取 0.0(ultralytics 8.4.57 NMS 陷阱);yaml 覆寫用備份/換名處理。
