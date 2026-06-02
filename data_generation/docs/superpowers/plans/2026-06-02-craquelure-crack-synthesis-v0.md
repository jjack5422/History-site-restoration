# 木質彩繪文物 crack/craquelure 程序式合成器 v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建一個純程序式合成器,在真實木質彩繪底圖上生成 crack 與 craquelure 劣化,輸出對齊 `crackseg_common.TileSegDataset` 的分型別 binary tile 資料集(`_data/synth_crack_v0`、`_data/synth_craq_v0`,tile 數比 8:1)。

**Architecture:** 可重用模組於 `src/synthgen/`(雙幾何引擎 + 資料驅動外觀 + 底圖篩選 + 切片索引),入口腳本於 `scripts/`,設定於 `configs/synth_v0.yaml`。幾何決定形狀與精確 GT,外觀只改像素值不動 GT。輸出資料集放共用 `_data/`,run 記錄放 `runs/`。

**Tech Stack:** Python 3、numpy、scipy(`scipy.spatial.Voronoi`)、opencv-python-headless、Pillow、PyYAML、pytest;共用套件 `crackseg_common`(editable install)。

**對應 spec:** `docs/superpowers/specs/2026-06-02-craquelure-crack-synthesis-v0-design.md`

**Commit 注意:** `research/` 是 git repo,但使用者慣例為「只在被要求時 commit」。各 Task 末的 commit step 於執行階段由使用者決定是否執行。

---

## File Structure

```
research/data_generation/
├─ src/synthgen/
│  ├─ __init__.py
│  ├─ config.py              # load_config(path)->dict (含 defaults)
│  ├─ base_selection.py      # score(), build_manifest()
│  ├─ appearance.py          # fit_profile(), save/load_profile(), render()
│  ├─ compose.py             # compose()
│  ├─ tiling.py              # tile_and_write(), finalize_index()
│  └─ geometry/
│     ├─ __init__.py
│     ├─ bezier.py           # cubic_bezier()
│     ├─ crack.py            # generate()
│     └─ craquelure.py       # generate()
├─ scripts/
│  ├─ scan_clean_slices.py        # (已存在)
│  ├─ fit_appearance_profile.py   # 包 appearance.fit_profile
│  ├─ select_bases.py             # 包 base_selection.build_manifest
│  ├─ synthesize.py               # 主流程
│  └─ validate_synth.py           # 統計比對 + preview
├─ configs/synth_v0.yaml
└─ tests/
   ├─ conftest.py
   ├─ test_bezier.py
   ├─ test_geometry_crack.py
   ├─ test_geometry_craquelure.py
   ├─ test_base_selection.py
   ├─ test_appearance.py
   ├─ test_compose.py
   └─ test_tiling.py
```

**資料契約(全模組共用):**
- image: `np.ndarray` shape `(H,W,3)` dtype `uint8`,RGB。
- mask: `np.ndarray` shape `(H,W)` dtype `uint8`,值 `{0,1}`。
- 幾何 `generate(size:int, params:dict, rng:np.random.Generator) -> mask(uint8 {0,1})`。
- rng 一律用 `np.random.Generator`(`np.random.default_rng`);可重現靠傳入固定 seed 衍生的 rng。

---

## Task 0: 專案環境與套件骨架

**Files:**
- Create: `research/datagen_env/`(venv,不進 git)
- Create: `research/data_generation/src/synthgen/__init__.py`
- Create: `research/data_generation/src/synthgen/geometry/__init__.py`
- Create: `research/data_generation/tests/conftest.py`
- Create: `research/data_generation/pytest.ini`
- Create: `research/data_generation/requirements.txt`

- [ ] **Step 1: 建 venv 並裝套件**

Run:
```bash
python3 -m venv /home/zzz90/research/datagen_env
/home/zzz90/research/datagen_env/bin/pip install -U pip
/home/zzz90/research/datagen_env/bin/pip install numpy scipy opencv-python-headless pillow pyyaml pytest tqdm
/home/zzz90/research/datagen_env/bin/pip install -e /home/zzz90/research/_lib
```
Expected: 安裝成功;最後一行顯示 `Successfully installed crackseg-common` 或類似 editable 安裝訊息。

- [ ] **Step 2: 驗證 crackseg_common import**

Run:
```bash
/home/zzz90/research/datagen_env/bin/python -c "from crackseg_common.data_utils import tile_image; from crackseg_common.dataset import TileSegDataset; print('ok')"
```
Expected: 印出 `ok`。

- [ ] **Step 3: 建 requirements.txt**

`research/data_generation/requirements.txt`:
```
numpy
scipy
opencv-python-headless
pillow
pyyaml
pytest
tqdm
-e /home/zzz90/research/_lib
```

- [ ] **Step 4: 建套件 __init__ 與 pytest 設定**

`research/data_generation/src/synthgen/__init__.py`:
```python
"""木質彩繪文物 crack/craquelure 程序式合成器 (v0)。"""
```

`research/data_generation/src/synthgen/geometry/__init__.py`:
```python
```

`research/data_generation/pytest.ini`:
```ini
[pytest]
pythonpath = src
testpaths = tests
addopts = -q
```

`research/data_generation/tests/conftest.py`:
```python
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)
```

- [ ] **Step 5: 驗證 pytest 可跑(零測試)**

Run: `cd /home/zzz90/research/data_generation && /home/zzz90/research/datagen_env/bin/python -m pytest`
Expected: `no tests ran` 且 exit code 5(無測試);無 import error。

- [ ] **Step 6: Commit**

```bash
cd /home/zzz90/research
git add data_generation/src data_generation/tests data_generation/pytest.ini data_generation/requirements.txt
git commit -m "chore(data_generation): scaffold synthgen package, venv deps, pytest"
```

---

## Task 1: Bézier 曲線輔助函式

**Files:**
- Create: `research/data_generation/src/synthgen/geometry/bezier.py`
- Test: `research/data_generation/tests/test_bezier.py`

- [ ] **Step 1: 寫失敗測試**

`tests/test_bezier.py`:
```python
import numpy as np
from synthgen.geometry.bezier import cubic_bezier


def test_endpoints_match():
    p0, p1, p2, p3 = (0, 0), (10, 0), (20, 10), (30, 30)
    pts = cubic_bezier(p0, p1, p2, p3, n=50)
    assert pts.shape == (50, 2)
    assert np.allclose(pts[0], p0)
    assert np.allclose(pts[-1], p3)


def test_straight_line_midpoint():
    # 控制點落在直線上 -> 結果為直線, 中點為兩端中點
    pts = cubic_bezier((0, 0), (1, 1), (2, 2), (3, 3), n=4)
    assert np.allclose(pts[:, 0], pts[:, 1])  # x==y 全程
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `cd /home/zzz90/research/data_generation && /home/zzz90/research/datagen_env/bin/python -m pytest tests/test_bezier.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'synthgen.geometry.bezier'`。

- [ ] **Step 3: 實作**

`src/synthgen/geometry/bezier.py`:
```python
import numpy as np


def cubic_bezier(p0, p1, p2, p3, n):
    """回傳 cubic Bézier 取樣點 (n,2) float。p* 為 (x,y)。"""
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    p0, p1, p2, p3 = map(lambda p: np.asarray(p, dtype=np.float64), (p0, p1, p2, p3))
    return ((1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2
            + t ** 3 * p3)
```

- [ ] **Step 4: 跑測試確認通過**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_bezier.py -v`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add data_generation/src/synthgen/geometry/bezier.py data_generation/tests/test_bezier.py
git commit -m "feat(synthgen): cubic bezier helper"
```

---

## Task 2: crack 幾何引擎

**Files:**
- Create: `research/data_generation/src/synthgen/geometry/crack.py`
- Test: `research/data_generation/tests/test_geometry_crack.py`

**介面:** `generate(size:int, params:dict, rng:np.random.Generator) -> np.ndarray(uint8 {0,1})`
**params:** `n_curves:[lo,hi]`、`taper_alpha:float(px)`、`taper_sigma:float`、`branch_p:[lo,hi]`。

- [ ] **Step 1: 寫失敗測試**

`tests/test_geometry_crack.py`:
```python
import numpy as np
from synthgen.geometry.crack import generate

PARAMS = {"n_curves": [2, 6], "taper_alpha": 2.0, "taper_sigma": 0.5, "branch_p": [0.3, 0.5]}


def test_shape_dtype_binary():
    m = generate(256, PARAMS, np.random.default_rng(0))
    assert m.shape == (256, 256)
    assert m.dtype == np.uint8
    assert set(np.unique(m)).issubset({0, 1})


def test_sparse_nonempty():
    m = generate(512, PARAMS, np.random.default_rng(1))
    fg = m.mean()
    assert fg > 0.0, "crack mask 不可全空"
    assert fg < 0.05, f"crack 應稀疏 (<5%), got {fg:.4f}"


def test_deterministic_with_same_seed():
    a = generate(256, PARAMS, np.random.default_rng(7))
    b = generate(256, PARAMS, np.random.default_rng(7))
    assert np.array_equal(a, b)
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_geometry_crack.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'synthgen.geometry.crack'`。

- [ ] **Step 3: 實作**

`src/synthgen/geometry/crack.py`:
```python
import numpy as np
import cv2
from .bezier import cubic_bezier


def _draw_curve(mask, p0, p3, rng, taper_alpha, taper_sigma):
    """沿一條 Bézier 曲線畫 tapered(兩端細中段粗)裂縫。"""
    size = mask.shape[0]
    # 內控制點 = 端點連線附近加高斯擾動
    jitter = size * 0.08
    p1 = np.array(p0) + rng.normal(0, jitter, 2)
    p2 = np.array(p3) + rng.normal(0, jitter, 2)
    length = np.linalg.norm(np.array(p3) - np.array(p0))
    n = int(np.clip(length, 80, 180))
    pts = cubic_bezier(p0, p1, p2, p3, n)
    for i, (x, y) in enumerate(pts):
        t = i / max(n - 1, 1)
        r = rng.normal(taper_alpha * (1.0 - abs(t - 0.5) * 2.0), taper_sigma)
        r = max(int(round(r)), 0)
        cv2.circle(mask, (int(round(x)), int(round(y))), r, 1, -1)
    return pts


def generate(size, params, rng):
    """回傳 (size,size) uint8 {0,1} 的稀疏 crack mask。"""
    mask = np.zeros((size, size), dtype=np.uint8)
    lo, hi = params["n_curves"]
    n_curves = int(rng.integers(lo, hi + 1))
    blo, bhi = params["branch_p"]
    for _ in range(n_curves):
        p0 = rng.uniform(0, size, 2)
        p3 = rng.uniform(0, size, 2)
        pts = _draw_curve(mask, p0, p3, rng,
                          params["taper_alpha"], params["taper_sigma"])
        # 機率從曲線中段長出一條分支
        if rng.uniform() < rng.uniform(blo, bhi):
            mid = pts[len(pts) // 2]
            direction = pts[-1] - pts[0]
            ang = rng.uniform(-np.pi / 3, np.pi / 3)
            rot = np.array([[np.cos(ang), -np.sin(ang)],
                            [np.sin(ang), np.cos(ang)]])
            end = mid + rot @ direction * rng.uniform(0.2, 0.5)
            _draw_curve(mask, mid, end, rng,
                        params["taper_alpha"] * 0.7, params["taper_sigma"])
    return (mask > 0).astype(np.uint8)
```

- [ ] **Step 4: 跑測試確認通過**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_geometry_crack.py -v`
Expected: 3 passed。若 `test_sparse_nonempty` 偶發超過 5%,調小 `taper_alpha` 或 `n_curves` 上限後再跑(此測試 seed 固定,應穩定通過)。

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add data_generation/src/synthgen/geometry/crack.py data_generation/tests/test_geometry_crack.py
git commit -m "feat(synthgen): bezier-based tapered crack geometry engine"
```

---

## Task 3: craquelure 幾何引擎(Voronoi)

**Files:**
- Create: `research/data_generation/src/synthgen/geometry/craquelure.py`
- Test: `research/data_generation/tests/test_geometry_craquelure.py`

**介面:** `generate(size:int, params:dict, rng:np.random.Generator) -> np.ndarray(uint8 {0,1})`
**params:** `cell_px:[lo,hi]`、`jitter:float(0-1)`、`edge_w:int`、`break_p:float(0-1)`。

- [ ] **Step 1: 寫失敗測試**

`tests/test_geometry_craquelure.py`:
```python
import numpy as np
import cv2
from synthgen.geometry.craquelure import generate

PARAMS = {"cell_px": [25, 60], "jitter": 0.3, "edge_w": 1, "break_p": 0.1}


def test_shape_dtype_binary():
    m = generate(256, PARAMS, np.random.default_rng(0))
    assert m.shape == (256, 256)
    assert m.dtype == np.uint8
    assert set(np.unique(m)).issubset({0, 1})


def test_forms_cells():
    # 背景(非裂縫)應被裂縫網切成多個 island(連通元件)
    m = generate(512, PARAMS, np.random.default_rng(2))
    bg = (m == 0).astype(np.uint8)
    n_components, _ = cv2.connectedComponents(bg)
    assert n_components - 1 >= 20, f"cell 數應 >=20, got {n_components - 1}"


def test_fg_in_craquelure_range():
    m = generate(512, PARAMS, np.random.default_rng(3))
    fg = m.mean() * 100
    assert 0.5 <= fg <= 12.0, f"craq fg% 應落在真實範圍, got {fg:.2f}%"
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_geometry_craquelure.py -v`
Expected: FAIL,`ModuleNotFoundError`。

- [ ] **Step 3: 實作**

`src/synthgen/geometry/craquelure.py`:
```python
import numpy as np
import cv2
from scipy.spatial import Voronoi


def generate(size, params, rng):
    """以 Voronoi cell 邊界生成 craquelure 網狀 mask, 回傳 (size,size) uint8 {0,1}。"""
    clo, chi = params["cell_px"]
    cell = float(rng.uniform(clo, chi))
    jitter = params["jitter"]
    edge_w = int(params["edge_w"])
    break_p = params["break_p"]

    # 在略大於影像的範圍撒抖動格點種子(避免邊界 cell 缺失)
    step = cell
    coords = np.arange(-cell, size + cell, step)
    gx, gy = np.meshgrid(coords, coords)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    pts += rng.normal(0, cell * jitter, pts.shape)

    vor = Voronoi(pts)
    mask = np.zeros((size, size), dtype=np.uint8)
    for a, b in vor.ridge_vertices:
        if a < 0 or b < 0:
            continue  # 跳過延伸到無窮遠的脊
        if rng.uniform() < break_p:
            continue  # 隨機斷裂模擬不連續
        pa = vor.vertices[a]
        pb = vor.vertices[b]
        cv2.line(mask,
                 (int(round(pa[0])), int(round(pa[1]))),
                 (int(round(pb[0])), int(round(pb[1]))),
                 1, edge_w, lineType=cv2.LINE_AA)
    # LINE_AA 會產生中間值, 二值化
    return (mask > 0).astype(np.uint8)
```

- [ ] **Step 4: 跑測試確認通過**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_geometry_craquelure.py -v`
Expected: 3 passed。若 `test_fg_in_craquelure_range` 偏高(>12%),調大 `cell_px` 下限;偏低則調小 cell 或調大 `edge_w`。

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add data_generation/src/synthgen/geometry/craquelure.py data_generation/tests/test_geometry_craquelure.py
git commit -m "feat(synthgen): voronoi-based craquelure geometry engine"
```

---

## Task 4: 底圖篩選

**Files:**
- Create: `research/data_generation/src/synthgen/base_selection.py`
- Test: `research/data_generation/tests/test_base_selection.py`

**介面:**
- `score(gray:np.ndarray, kernel:int=11, blackhat_t:int=18) -> float`(black-hat 候選像素佔比)
- `build_manifest(slices_dir, thresh, out_path, kernel=11, blackhat_t=18) -> dict`

- [ ] **Step 1: 寫失敗測試**

`tests/test_base_selection.py`:
```python
import json
import numpy as np
import cv2
from PIL import Image
from synthgen.base_selection import score, build_manifest


def test_clean_lower_than_cracky():
    clean = np.full((256, 256), 180, np.uint8)
    cracky = clean.copy()
    for y in range(0, 256, 16):
        cv2.line(cracky, (0, y), (255, y), 30, 1)
    assert score(clean) < score(cracky)
    assert score(clean) < 0.01


def test_build_manifest(tmp_path):
    sd = tmp_path / "slices"
    sd.mkdir()
    Image.fromarray(np.full((128, 128, 3), 200, np.uint8)).save(sd / "clean.jpg")
    cracky = np.full((128, 128, 3), 200, np.uint8)
    for y in range(0, 128, 6):
        cv2.line(cracky, (0, y), (127, y), 20, 1)
    Image.fromarray(cracky).save(sd / "dirty.jpg")
    out = tmp_path / "manifest.json"
    man = build_manifest(str(sd), thresh=0.015, out_path=str(out))
    assert out.exists()
    assert "clean.jpg" in man["clean"]
    assert "dirty.jpg" not in man["clean"]
    assert json.load(open(out))["thresh"] == 0.015
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_base_selection.py -v`
Expected: FAIL,`ModuleNotFoundError`。

- [ ] **Step 3: 實作**

`src/synthgen/base_selection.py`:
```python
import json
import os
import glob
import numpy as np
import cv2


def score(gray, kernel=11, blackhat_t=18):
    """black-hat 突顯暗細結構, 回傳候選像素佔比 (0-1)。gray 為單通道 uint8。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    return float((bh > blackhat_t).mean())


def build_manifest(slices_dir, thresh, out_path, kernel=11, blackhat_t=18):
    """掃描 slices_dir 所有影像, 依 score<thresh 篩乾淨底圖, 寫 manifest json。"""
    files = sorted(glob.glob(os.path.join(slices_dir, "*.jpg")))
    scores = []
    for f in files:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        scores.append({"name": os.path.basename(f), "score": score(g, kernel, blackhat_t)})
    clean = [s["name"] for s in scores if s["score"] < thresh]
    manifest = {
        "slices_dir": slices_dir,
        "thresh": thresh,
        "kernel": kernel,
        "blackhat_t": blackhat_t,
        "n_total": len(scores),
        "n_clean": len(clean),
        "clean": clean,
        "scores": scores,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return manifest
```

- [ ] **Step 4: 跑測試確認通過**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_base_selection.py -v`
Expected: 2 passed。

- [ ] **Step 5: 建入口腳本 select_bases.py**

`scripts/select_bases.py`:
```python
"""掃描 _data/image_1024_slices 篩乾淨底圖, 輸出 data/base_manifest.json。

用法: python scripts/select_bases.py [--slices DIR] [--thresh 0.015] [--out PATH]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from synthgen.base_selection import build_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="/home/zzz90/research/_data/image_1024_slices")
    ap.add_argument("--thresh", type=float, default=0.015)
    ap.add_argument("--out", default="data/base_manifest.json")
    args = ap.parse_args()
    man = build_manifest(args.slices, args.thresh, args.out)
    print(f"掃描 {man['n_total']} 張, 乾淨 {man['n_clean']} 張 (thresh={args.thresh}) -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 跑真實 base manifest**

Run: `cd /home/zzz90/research/data_generation && /home/zzz90/research/datagen_env/bin/python scripts/select_bases.py`
Expected: 印出類似 `掃描 581 張, 乾淨 ~31 張 (thresh=0.015) -> data/base_manifest.json`;`data/base_manifest.json` 產生。

- [ ] **Step 7: Commit**

```bash
cd /home/zzz90/research
git add data_generation/src/synthgen/base_selection.py data_generation/tests/test_base_selection.py data_generation/scripts/select_bases.py
git commit -m "feat(synthgen): base-slice cleanliness selection + manifest"
```

---

## Task 5: 資料驅動外觀(profile 擬合 + 渲染)

**Files:**
- Create: `research/data_generation/src/synthgen/appearance.py`
- Test: `research/data_generation/tests/test_appearance.py`

**介面:**
- `fit_profile(img_dir, mask_dir) -> dict`(keys: `dL,da,db` 各為 `[mean,std]`,`n`)
- `save_profile(profile, path)` / `load_profile(path)`(JSON)
- `render(base_rgb, geo_mask, profile, cfg, rng) -> np.ndarray(uint8 RGB)`
  - `cfg` keys: `min_contrast:float`、`erosion:int`、`blur_sigma:float`

裂縫普遍比鄰域暗 → `dL` mean 預期為負;render 以 soft-alpha 把 ΔL/Δa/Δb 疊到 Lab,再轉回 RGB。

- [ ] **Step 1: 寫失敗測試**

`tests/test_appearance.py`:
```python
import numpy as np
from synthgen.appearance import fit_profile, save_profile, load_profile, render


def test_fit_profile_detects_dark_cracks(tmp_path):
    import cv2
    from PIL import Image
    idir = tmp_path / "img"; mdir = tmp_path / "msk"
    idir.mkdir(); mdir.mkdir()
    for i in range(3):
        img = np.full((64, 64, 3), 200, np.uint8)
        m = np.zeros((64, 64), np.uint8)
        cv2.line(img, (0, 32), (63, 32), (40, 40, 40), 2)  # 暗裂縫
        cv2.line(m, (0, 32), (63, 32), 1, 2)
        Image.fromarray(img).save(idir / f"{i}.png")
        Image.fromarray(m).save(mdir / f"{i}.png")
    prof = fit_profile(str(idir), str(mdir))
    assert prof["n"] > 0
    assert prof["dL"][0] < 0, "裂縫應比鄰域暗 (dL<0)"


def test_save_load_roundtrip(tmp_path):
    prof = {"dL": [-20.0, 5.0], "da": [0.0, 1.0], "db": [0.0, 1.0], "n": 10}
    p = tmp_path / "p.json"
    save_profile(prof, str(p))
    assert load_profile(str(p)) == prof


def test_render_darkens_cracks_preserves_elsewhere():
    base = np.full((64, 64, 3), 200, np.uint8)
    geo = np.zeros((64, 64), np.uint8)
    geo[30:34, :] = 1
    prof = {"dL": [-30.0, 0.0], "da": [0.0, 0.0], "db": [0.0, 0.0], "n": 100}
    cfg = {"min_contrast": 12, "erosion": 0, "blur_sigma": 0}
    out = render(base, geo, prof, cfg, np.random.default_rng(0))
    assert out.shape == base.shape and out.dtype == np.uint8
    assert out[32].mean() < base[32].mean() - 10, "裂縫處應變暗"
    assert np.array_equal(out[0], base[0]), "非裂縫列不應改變"
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_appearance.py -v`
Expected: FAIL,`ModuleNotFoundError`。

- [ ] **Step 3: 實作**

`src/synthgen/appearance.py`:
```python
import json
import os
import glob
import numpy as np
import cv2


def _to_lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _to_rgb(lab):
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def fit_profile(img_dir, mask_dir):
    """統計裂縫前景相對其周邊鄰域(dilate ring)的 Lab 差, 聚合成分佈。"""
    dL, da, db = [], [], []
    for ip in sorted(glob.glob(os.path.join(img_dir, "*"))):
        stem = os.path.splitext(os.path.basename(ip))[0]
        mp = None
        for ext in (".png", ".jpg", ".jpeg"):
            cand = os.path.join(mask_dir, stem + ext)
            if os.path.exists(cand):
                mp = cand
                break
        if mp is None:
            continue
        rgb = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if rgb is None or m is None:
            continue
        fg = m > 0
        if fg.sum() == 0:
            continue
        ring = cv2.dilate(fg.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        ring &= ~fg
        if ring.sum() == 0:
            continue
        lab = _to_lab(rgb)
        for ch, acc in zip(range(3), (dL, da, db)):
            acc.append(float(lab[..., ch][fg].mean() - lab[..., ch][ring].mean()))
    def ms(x):
        return [float(np.mean(x)), float(np.std(x))] if x else [0.0, 0.0]
    return {"dL": ms(dL), "da": ms(da), "db": ms(db), "n": len(dL)}


def save_profile(profile, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(profile, fh, ensure_ascii=False, indent=2)


def load_profile(path):
    with open(path) as fh:
        return json.load(fh)


def render(base_rgb, geo_mask, profile, cfg, rng):
    """把 geo_mask 以資料驅動 Lab offset 渲染到 base_rgb, 回傳 RGB uint8。"""
    geo = (geo_mask > 0).astype(np.uint8)
    if cfg.get("erosion", 0) > 0:
        k = np.ones((cfg["erosion"], cfg["erosion"]), np.uint8)
        geo = cv2.erode(geo, k)
    alpha = geo.astype(np.float32)
    if cfg.get("blur_sigma", 0) > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), cfg["blur_sigma"])
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]

    dL = rng.normal(*profile["dL"])
    mc = cfg.get("min_contrast", 0)
    if mc and abs(dL) < mc:
        dL = -mc if dL <= 0 else mc  # 維持原方向但拉到最小對比
    da = rng.normal(*profile["da"])
    db = rng.normal(*profile["db"])
    delta = np.array([dL, da, db], dtype=np.float32)

    lab = _to_lab(base_rgb)
    lab = lab + alpha * delta
    return _to_rgb(lab)
```

- [ ] **Step 4: 跑測試確認通過**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_appearance.py -v`
Expected: 3 passed。

- [ ] **Step 5: 建入口腳本 fit_appearance_profile.py**

`scripts/fit_appearance_profile.py`:
```python
"""由 labeled32_{crack,craq}_v3 擬合外觀 profile, 存到 data/appearance/。

用法: python scripts/fit_appearance_profile.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from synthgen.appearance import fit_profile, save_profile

DATA = "/home/zzz90/research/_data"
JOBS = {
    "crack": (f"{DATA}/labeled32_crack_v3/images", f"{DATA}/labeled32_crack_v3/masks"),
    "craq": (f"{DATA}/labeled32_craq_v3/images", f"{DATA}/labeled32_craq_v3/masks"),
}


def main():
    for typ, (idir, mdir) in JOBS.items():
        prof = fit_profile(idir, mdir)
        out = f"data/appearance/appearance_profile_{typ}.json"
        save_profile(prof, out)
        print(f"{typ}: n={prof['n']} dL={prof['dL']} -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 跑真實 profile 擬合**

Run: `cd /home/zzz90/research/data_generation && /home/zzz90/research/datagen_env/bin/python scripts/fit_appearance_profile.py`
Expected: 印出兩型別 `n=32` 與 `dL=[負值, std]`;`data/appearance/appearance_profile_{crack,craq}.json` 產生。若 dL 非負,檢查 mask/影像對應是否正確。

- [ ] **Step 7: Commit**

```bash
cd /home/zzz90/research
git add data_generation/src/synthgen/appearance.py data_generation/tests/test_appearance.py data_generation/scripts/fit_appearance_profile.py
git commit -m "feat(synthgen): data-driven Lab appearance profile + renderer"
```

---

## Task 6: compose(底圖 + 幾何 → 影像/mask 對)

**Files:**
- Create: `research/data_generation/src/synthgen/compose.py`
- Test: `research/data_generation/tests/test_compose.py`

**介面:** `compose(base_rgb, geo_mask, profile, cfg, rng) -> (img_rgb:uint8, mask:uint8{0,1})`
GT mask = `(geo_mask>0)` 二值(blur 前),與外觀渲染解耦。

- [ ] **Step 1: 寫失敗測試**

`tests/test_compose.py`:
```python
import numpy as np
from synthgen.compose import compose


def test_mask_is_pre_blur_binary():
    base = np.full((64, 64, 3), 200, np.uint8)
    geo = np.zeros((64, 64), np.uint8)
    geo[30:33, 10:50] = 1
    prof = {"dL": [-30.0, 0.0], "da": [0.0, 0.0], "db": [0.0, 0.0], "n": 1}
    cfg = {"min_contrast": 12, "erosion": 0, "blur_sigma": 2}
    img, mask = compose(base, geo, prof, cfg, np.random.default_rng(0))
    assert img.shape == base.shape and img.dtype == np.uint8
    assert mask.dtype == np.uint8 and set(np.unique(mask)).issubset({0, 1})
    # mask 等於原 geo(blur 不影響 GT)
    assert np.array_equal(mask, (geo > 0).astype(np.uint8))
    # 影像在裂縫處變暗
    assert img[31, 30].mean() < 190
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_compose.py -v`
Expected: FAIL,`ModuleNotFoundError`。

- [ ] **Step 3: 實作**

`src/synthgen/compose.py`:
```python
import numpy as np
from .appearance import render


def compose(base_rgb, geo_mask, profile, cfg, rng):
    """回傳 (渲染後影像 RGB uint8, GT mask uint8 {0,1})。

    GT 取 blur 前的二值 geo_mask, 與外觀解耦以保證像素級精確。
    """
    mask = (geo_mask > 0).astype(np.uint8)
    img = render(base_rgb, geo_mask, profile, cfg, rng)
    return img, mask
```

- [ ] **Step 4: 跑測試確認通過**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_compose.py -v`
Expected: 1 passed。

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add data_generation/src/synthgen/compose.py data_generation/tests/test_compose.py
git commit -m "feat(synthgen): compose base+geometry into image/mask pair"
```

---

## Task 7: 切片與索引

**Files:**
- Create: `research/data_generation/src/synthgen/tiling.py`
- Test: `research/data_generation/tests/test_tiling.py`

**介面:**
- `tile_and_write(img, mask, stem, out_root, tile_size, stride, keep_negative_ratio, rng) -> list[dict]`
  - 寫 `out_root/images/<tile>.png`(RGB)與 `out_root/masks/<tile>.png`(uint8 {0,1});
  - 含前景 tile 全留;零前景 tile 依 `keep_negative_ratio` 抽樣保留;
  - 回傳每個保留 tile 的 item dict:`{tile,stem,y,x,has_fg,tile_std,fg_pixels}`。
- `finalize_index(out_root, items, summary_extra, seed) -> None`
  - 寫 `out_root/tile_index.json`(`{"summary":..., "items":...}`)與 `out_root/nofold_all_train.json`(group_by stem,單 fold 全 train)。

tile 命名沿用下游:`<stem>__y{y:05d}_x{x:05d}.png`。切片用 `crackseg_common.data_utils.tile_image`。

- [ ] **Step 1: 寫失敗測試**

`tests/test_tiling.py`:
```python
import json
import numpy as np
from PIL import Image
from synthgen.tiling import tile_and_write, finalize_index


def test_tile_and_write(tmp_path):
    img = np.full((1024, 1024, 3), 180, np.uint8)
    mask = np.zeros((1024, 1024), np.uint8)
    mask[500:520, :] = 1  # 橫跨, 保證部分 tile 有前景
    items = tile_and_write(img, mask, "samp", str(tmp_path), 512, 256, 0.1,
                           np.random.default_rng(0))
    assert len(items) > 0
    # 檔案存在且命名合規
    for it in items:
        assert (tmp_path / "images" / it["tile"]).exists()
        assert (tmp_path / "masks" / it["tile"]).exists()
        assert it["tile"].startswith("samp__y")
    # 至少一個前景 tile, 且其 mask 為 {0,1}
    fg_items = [it for it in items if it["has_fg"]]
    assert fg_items
    m = np.array(Image.open(tmp_path / "masks" / fg_items[0]["tile"]))
    assert set(np.unique(m)).issubset({0, 1})


def test_finalize_index(tmp_path):
    (tmp_path / "images").mkdir(); (tmp_path / "masks").mkdir()
    items = [{"tile": "a__y00000_x00000.png", "stem": "a", "y": 0, "x": 0,
              "has_fg": True, "tile_std": 10.0, "fg_pixels": 50}]
    finalize_index(str(tmp_path), items, {"target_class": "crack"}, seed=42)
    idx = json.load(open(tmp_path / "tile_index.json"))
    assert idx["summary"]["target_class"] == "crack"
    assert idx["items"] == items
    split = json.load(open(tmp_path / "nofold_all_train.json"))
    assert split["group_by"] == "stem"
    assert "a" in split["groups"]
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_tiling.py -v`
Expected: FAIL,`ModuleNotFoundError`。

- [ ] **Step 3: 實作**

`src/synthgen/tiling.py`:
```python
import json
import os
import numpy as np
from PIL import Image
from crackseg_common.data_utils import tile_image


def tile_and_write(img, mask, stem, out_root, tile_size, stride,
                   keep_negative_ratio, rng):
    """切 img/mask 成 tile 並寫檔, 回傳保留 tile 的 item dict list。"""
    img_dir = os.path.join(out_root, "images")
    msk_dir = os.path.join(out_root, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    img_tiles, coords, _ = tile_image(img, tile_size=tile_size, stride=stride)
    msk_tiles, _, _ = tile_image(mask, tile_size=tile_size, stride=stride)

    items = []
    for (it_img, (y, x), it_msk) in zip(img_tiles, coords, msk_tiles):
        fg_px = int((it_msk > 0).sum())
        has_fg = fg_px > 0
        if not has_fg and rng.uniform() >= keep_negative_ratio:
            continue  # 零前景 tile 依比例抽樣保留
        name = f"{stem}__y{y:05d}_x{x:05d}.png"
        Image.fromarray(it_img).save(os.path.join(img_dir, name))
        Image.fromarray((it_msk > 0).astype(np.uint8)).save(os.path.join(msk_dir, name))
        items.append({
            "tile": name, "stem": stem, "y": int(y), "x": int(x),
            "has_fg": has_fg, "tile_std": float(it_img.astype(np.float32).std()),
            "fg_pixels": fg_px,
        })
    return items


def finalize_index(out_root, items, summary_extra, seed):
    """寫 tile_index.json 與 nofold_all_train.json。"""
    n_fg = sum(1 for it in items if it["has_fg"])
    summary = {
        "tile_size": None, "stride": None, "seed": seed,
        "total_tiles": len(items), "kept_foreground": n_fg,
        "kept_background": len(items) - n_fg,
    }
    summary.update(summary_extra)
    with open(os.path.join(out_root, "tile_index.json"), "w") as fh:
        json.dump({"summary": summary, "items": items}, fh, ensure_ascii=False, indent=2)

    groups = sorted({it["stem"] for it in items})
    split = {
        "tiles_root": os.path.abspath(out_root),
        "group_by": "stem", "n_splits": 1, "seed": seed,
        "groups": groups,
        "folds": [{
            "fold": 0, "val_groups": [], "train_groups": groups,
            "n_train_tiles": len(items), "n_val_tiles": 0,
        }],
    }
    with open(os.path.join(out_root, "nofold_all_train.json"), "w") as fh:
        json.dump(split, fh, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: 跑測試確認通過**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_tiling.py -v`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
cd /home/zzz90/research
git add data_generation/src/synthgen/tiling.py data_generation/tests/test_tiling.py
git commit -m "feat(synthgen): tiling + tile_index/split writer (TileSegDataset format)"
```

---

## Task 8: config 載入 + synthesize 主流程

**Files:**
- Create: `research/data_generation/src/synthgen/config.py`
- Create: `research/data_generation/configs/synth_v0.yaml`
- Create: `research/data_generation/scripts/synthesize.py`
- Test: `research/data_generation/tests/test_config.py`

- [ ] **Step 1: 寫 config 失敗測試**

`tests/test_config.py`:
```python
from synthgen.config import load_config


def test_load_config_has_required_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "seed: 42\n"
        "tile_size: 512\nstride: 256\nslice_size: 1024\n"
        "ratio: {crack: 8, craq: 1}\n"
        "target_tiles: {crack: 800, craq: 100}\n"
        "base: {manifest: data/base_manifest.json, allow_reuse: true, max_reuse: 20}\n"
        "crack: {n_curves: [2,6], taper_alpha: 2.0, taper_sigma: 0.5, branch_p: [0.3,0.5], target_fg: [0.05,2.0]}\n"
        "craq: {cell_px: [25,60], jitter: 0.3, edge_w: 1, break_p: 0.1, target_fg: [0.5,12.0]}\n"
        "appearance: {profile_dir: data/appearance, min_contrast: 12, erosion: 2, blur_sigma: 2}\n"
        "keep_negative_ratio: 0.1\nmax_regen_fail: 50\n"
    )
    cfg = load_config(str(p))
    assert cfg["seed"] == 42
    assert cfg["target_tiles"]["crack"] == 800
    assert cfg["crack"]["taper_alpha"] == 2.0
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL,`ModuleNotFoundError`。

- [ ] **Step 3: 實作 config.py**

`src/synthgen/config.py`:
```python
import yaml


def load_config(path):
    """讀 YAML 設定, 回傳 dict。"""
    with open(path) as fh:
        return yaml.safe_load(fh)
```

- [ ] **Step 4: 跑測試確認通過**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_config.py -v`
Expected: 1 passed。

- [ ] **Step 5: 建 configs/synth_v0.yaml**

`configs/synth_v0.yaml`:
```yaml
seed: 42
tile_size: 512
stride: 256
slice_size: 1024
ratio: {crack: 8, craq: 1}
target_tiles: {crack: 800, craq: 100}
output:
  crack: /home/zzz90/research/_data/synth_crack_v0
  craq: /home/zzz90/research/_data/synth_craq_v0
base:
  slices_dir: /home/zzz90/research/_data/image_1024_slices
  manifest: data/base_manifest.json
  allow_reuse: true
  max_reuse: 20
crack: {n_curves: [2, 6], taper_alpha: 2.0, taper_sigma: 0.5, branch_p: [0.3, 0.5], target_fg: [0.05, 2.0]}
craq:  {cell_px: [25, 60], jitter: 0.3, edge_w: 1, break_p: 0.1, target_fg: [0.5, 9.0]}
appearance: {profile_dir: data/appearance, min_contrast: 12, erosion: 2, blur_sigma: 2}
keep_negative_ratio: 0.1
max_regen_fail: 50
```

- [ ] **Step 6: 實作 synthesize.py**

`scripts/synthesize.py`:
```python
"""主合成流程: 生成 crack/craq 分型別 binary tile 資料集到 _data/synth_*_v0。

用法: python scripts/synthesize.py --config configs/synth_v0.yaml [--overwrite]
"""
import argparse
import os
import sys
import json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from synthgen.config import load_config
from synthgen.appearance import load_profile
from synthgen.compose import compose
from synthgen.tiling import tile_and_write, finalize_index
from synthgen.geometry import crack as crack_eng
from synthgen.geometry import craquelure as craq_eng

ENGINES = {"crack": crack_eng.generate, "craq": craq_eng.generate}


def _load_base(slices_dir, name):
    bgr = cv2.imread(os.path.join(slices_dir, name))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def synth_type(typ, cfg, rng):
    out_root = cfg["output"][typ]
    if os.path.exists(out_root):
        raise FileExistsError(f"{out_root} 已存在, 加 --overwrite 或先移除")
    os.makedirs(out_root)

    manifest = json.load(open(os.path.join(ROOT, cfg["base"]["manifest"])))
    bases = manifest["clean"] or [s["name"] for s in
                                   sorted(manifest["scores"], key=lambda s: s["score"])[:30]]
    profile = load_profile(os.path.join(ROOT, cfg["appearance"]["profile_dir"],
                                        f"appearance_profile_{typ}.json"))
    slices_dir = cfg["base"]["slices_dir"]
    size = cfg["slice_size"]
    fg_lo, fg_hi = cfg[typ]["target_fg"]
    gen = ENGINES[typ]

    items, n_target, reuse = [], cfg["target_tiles"][typ], 0
    base_idx = 0
    sample_id = 0
    while sum(1 for it in items if it["has_fg"]) < n_target:
        name = bases[base_idx % len(bases)]
        if base_idx >= len(bases):
            reuse = base_idx // len(bases)
            if reuse > cfg["base"]["max_reuse"]:
                print(f"[warn] {typ}: base 重用倍數 {reuse} 超過上限 {cfg['base']['max_reuse']}")
        base_idx += 1
        base = _load_base(slices_dir, name)

        # 生成幾何, 退化(空/超範圍)就重抽
        geo = None
        for _ in range(cfg["max_regen_fail"]):
            cand = gen(size, cfg[typ], rng)
            fg = cand.mean() * 100
            if fg_lo <= fg <= fg_hi:
                geo = cand
                break
        if geo is None:
            continue

        img, msk = compose(base, geo, profile, cfg["appearance"], rng)
        stem = f"{os.path.splitext(name)[0]}__s{sample_id:04d}"
        sample_id += 1
        items += tile_and_write(img, msk, stem, out_root, cfg["tile_size"],
                                cfg["stride"], cfg["keep_negative_ratio"], rng)

    summary = {"target_class": typ, "tile_size": cfg["tile_size"],
               "stride": cfg["stride"], "n_bases": len(bases),
               "base_reuse_factor": reuse, "n_samples": sample_id}
    finalize_index(out_root, items, summary, cfg["seed"])
    return {"type": typ, "out": out_root, "n_tiles": len(items),
            "n_fg": sum(1 for it in items if it["has_fg"]),
            "n_samples": sample_id, "reuse": reuse}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synth_v0.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(os.path.join(ROOT, args.config))

    if args.overwrite:
        import shutil
        for typ in ("crack", "craq"):
            if os.path.exists(cfg["output"][typ]):
                shutil.rmtree(cfg["output"][typ])

    ss = np.random.SeedSequence(cfg["seed"])
    child = dict(zip(("crack", "craq"), ss.spawn(2)))
    results = []
    for typ in ("crack", "craq"):
        results.append(synth_type(typ, cfg, np.random.default_rng(child[typ])))
    for r in results:
        print(f"{r['type']}: tiles={r['n_tiles']} (fg={r['n_fg']}) "
              f"samples={r['n_samples']} reuse={r['reuse']} -> {r['out']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: 全測試通過**

Run: `cd /home/zzz90/research/data_generation && /home/zzz90/research/datagen_env/bin/python -m pytest`
Expected: 全部 passed(約 14 個測試)。

- [ ] **Step 8: Commit**

```bash
cd /home/zzz90/research
git add data_generation/src/synthgen/config.py data_generation/configs/synth_v0.yaml data_generation/scripts/synthesize.py data_generation/tests/test_config.py
git commit -m "feat(synthgen): config loader + synthesize main pipeline (8:1)"
```

---

## Task 9: 端到端煙霧測試(小規模 run)

**Files:**
- Create: `research/data_generation/runs/2026-06-02-synth-v0-smoke/manifest.md`
- Modify: `research/data_generation/EXPERIMENTS.md`

- [ ] **Step 1: 確認前置產物存在**

Run:
```bash
cd /home/zzz90/research/data_generation
ls data/base_manifest.json data/appearance/appearance_profile_crack.json data/appearance/appearance_profile_craq.json
```
Expected: 三個檔都存在(Task 4 Step 6、Task 5 Step 6 已產生)。若缺,先跑對應腳本。

- [ ] **Step 2: 建煙霧用小 config**

`configs/synth_smoke.yaml`(複製 synth_v0.yaml,只改 target_tiles 與輸出路徑):
```yaml
seed: 42
tile_size: 512
stride: 256
slice_size: 1024
ratio: {crack: 8, craq: 1}
target_tiles: {crack: 16, craq: 2}
output:
  crack: /home/zzz90/research/_data/synth_crack_smoke
  craq: /home/zzz90/research/_data/synth_craq_smoke
base:
  slices_dir: /home/zzz90/research/_data/image_1024_slices
  manifest: data/base_manifest.json
  allow_reuse: true
  max_reuse: 50
crack: {n_curves: [2, 6], taper_alpha: 2.0, taper_sigma: 0.5, branch_p: [0.3, 0.5], target_fg: [0.05, 2.0]}
craq:  {cell_px: [25, 60], jitter: 0.3, edge_w: 1, break_p: 0.1, target_fg: [0.5, 9.0]}
appearance: {profile_dir: data/appearance, min_contrast: 12, erosion: 2, blur_sigma: 2}
keep_negative_ratio: 0.1
max_regen_fail: 50
```

- [ ] **Step 3: 跑煙霧合成**

Run: `cd /home/zzz90/research/data_generation && /home/zzz90/research/datagen_env/bin/python scripts/synthesize.py --config configs/synth_smoke.yaml --overwrite`
Expected: 印出 `crack: tiles=... (fg>=16) ...` 與 `craq: tiles=... (fg>=2) ...`,無例外。

- [ ] **Step 4: 驗證輸出可被 TileSegDataset 讀取**

Run:
```bash
/home/zzz90/research/datagen_env/bin/python - << 'PY'
import json
from crackseg_common.dataset import TileSegDataset
root = "/home/zzz90/research/_data/synth_crack_smoke"
idx = json.load(open(f"{root}/tile_index.json"))
ds = TileSegDataset(root, idx["items"])
s = ds[0]
print("n_items", len(ds), "image", tuple(s["image"].shape), "mask", tuple(s["mask"].shape),
      "mask_max", int(s["mask"].max()))
assert s["image"].shape[0] == 3 and s["mask"].max() <= 1
print("TileSegDataset OK")
PY
```
Expected: 印出 `TileSegDataset OK`,image shape `(3,512,512)`,mask shape `(512,512)`,mask_max ≤ 1。

- [ ] **Step 5: 視覺抽檢輸出**

Run:
```bash
/home/zzz90/research/datagen_env/bin/python - << 'PY'
import glob, numpy as np, cv2, os
from PIL import Image
out="/tmp/synth_preview"; os.makedirs(out, exist_ok=True)
for typ in ("crack","craq"):
    root=f"/home/zzz90/research/_data/synth_{typ}_smoke"
    imgs=sorted(glob.glob(f"{root}/images/*.png"))
    # 找有前景的 tile
    for ip in imgs:
        mp=ip.replace("/images/","/masks/")
        m=np.array(Image.open(mp))
        if (m>0).sum()>0:
            img=np.array(Image.open(ip).convert("RGB"))
            ov=img.copy(); ov[m>0]=[255,0,0]
            comp=np.concatenate([img, np.stack([(m>0)*255]*3,-1).astype('uint8'), ov],axis=1)
            Image.fromarray(comp).save(f"{out}/{typ}.png"); print(typ,"->",f"{out}/{typ}.png", "fg%=%.2f"%(100*(m>0).mean())); break
PY
```
Expected: 產生 `/tmp/synth_preview/crack.png` 與 `craq.png`(影像|mask|overlay)。**人眼檢視**:crack 應為稀疏細長裂、craq 應為密集 cell 網,overlay 對齊。

- [ ] **Step 6: 寫 run manifest 與 EXPERIMENTS 列**

`runs/2026-06-02-synth-v0-smoke/manifest.md`:
```markdown
# Run: synth-v0-smoke

- 日期: 2026-06-02
- 目的: 端到端煙霧驗證合成 pipeline(小規模 crack 16 / craq 2)
- 指令: `python scripts/synthesize.py --config configs/synth_smoke.yaml --overwrite`
- config: configs/synth_smoke.yaml(seed 42, 8:1)
- 環境: venv /home/zzz90/research/datagen_env;crackseg_common editable
- 前置: data/base_manifest.json, data/appearance/appearance_profile_{crack,craq}.json
- 輸出: _data/synth_{crack,craq}_smoke(煙霧用,可刪)
- 結果: <填實際 tiles/fg/reuse 數字>;TileSegDataset 可讀 = 是;視覺抽檢 = <通過/問題>
```

在 `EXPERIMENTS.md` 表格補一列:
```markdown
| 2026-06-02 | synth-v0-smoke | 端到端煙霧驗證合成 pipeline | tiles=<..> | <通過/待查> | `runs/2026-06-02-synth-v0-smoke/` |
```

- [ ] **Step 7: Commit**

```bash
cd /home/zzz90/research
git add data_generation/configs/synth_smoke.yaml data_generation/runs/2026-06-02-synth-v0-smoke data_generation/EXPERIMENTS.md
git commit -m "test(synthgen): end-to-end smoke run + manifest"
```

---

## Task 10: 驗收驗證(統計比對 + preview grid)

**Files:**
- Create: `research/data_generation/src/synthgen/validate.py`
- Create: `research/data_generation/scripts/validate_synth.py`
- Test: `research/data_generation/tests/test_validate.py`

**介面:** `mask_stats(mask_dir) -> dict`(`fg_mean_pct`、`fg_pcts`、`n`);`compare(real_stats, synth_stats) -> dict`(`in_range:bool` 等)。

- [ ] **Step 1: 寫失敗測試**

`tests/test_validate.py`:
```python
import numpy as np
from PIL import Image
from synthgen.validate import mask_stats, compare


def test_mask_stats(tmp_path):
    md = tmp_path / "masks"; md.mkdir()
    m = np.zeros((100, 100), np.uint8); m[:, :2] = 1  # 2% fg
    Image.fromarray(m).save(md / "a.png")
    st = mask_stats(str(md))
    assert st["n"] == 1
    assert abs(st["fg_mean_pct"] - 2.0) < 0.5


def test_compare_in_range():
    real = {"fg_mean_pct": 2.86, "fg_pcts": [0.0, 9.0]}
    synth = {"fg_mean_pct": 3.0, "fg_pcts": [0.5, 8.0]}
    res = compare(real, synth)
    assert res["synth_mean_within_real_minmax"] is True
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_validate.py -v`
Expected: FAIL,`ModuleNotFoundError`。

- [ ] **Step 3: 實作 validate.py**

`src/synthgen/validate.py`:
```python
import os
import glob
import numpy as np
from PIL import Image


def mask_stats(mask_dir):
    """統計一個 masks 目錄的前景佔比分佈。"""
    pcts = []
    for mp in sorted(glob.glob(os.path.join(mask_dir, "*.png"))):
        m = np.array(Image.open(mp))
        if m.ndim == 3:
            m = m[..., 0]
        pcts.append(float((m > 0).mean() * 100))
    pcts = pcts or [0.0]
    return {"n": len([p for p in pcts]), "fg_mean_pct": float(np.mean(pcts)),
            "fg_min_pct": float(np.min(pcts)), "fg_max_pct": float(np.max(pcts)),
            "fg_pcts": pcts}


def compare(real_stats, synth_stats):
    """合成 mean 是否落在真實 [min,max] 範圍內。"""
    rlo, rhi = min(real_stats["fg_pcts"]), max(real_stats["fg_pcts"])
    sm = synth_stats["fg_mean_pct"]
    return {
        "real_mean_pct": real_stats["fg_mean_pct"],
        "synth_mean_pct": sm,
        "real_range_pct": [rlo, rhi],
        "synth_mean_within_real_minmax": bool(rlo <= sm <= rhi),
    }
```

- [ ] **Step 4: 跑測試確認通過**

Run: `/home/zzz90/research/datagen_env/bin/python -m pytest tests/test_validate.py -v`
Expected: 2 passed。

- [ ] **Step 5: 實作 validate_synth.py**

`scripts/validate_synth.py`:
```python
"""比對合成 vs 真實 mask 統計, 輸出 report 與 preview grid。

用法: python scripts/validate_synth.py --type crack --synth_root <dir> --out <dir>
"""
import argparse
import os
import sys
import glob
import json
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from synthgen.validate import mask_stats, compare

REAL = {"crack": "/home/zzz90/research/_data/labeled32_crack_v3/tiles_512/masks",
        "craq": "/home/zzz90/research/_data/labeled32_craq_v3/tiles_512/masks"}


def preview_grid(synth_root, real_img_dir, out_path, n=4):
    """各取合成/真實有前景 tile 拼 overlay grid。"""
    def pick(root, imgs_glob, masks_dir_from_img):
        rows = []
        for ip in sorted(glob.glob(imgs_glob)):
            mp = masks_dir_from_img(ip)
            if not os.path.exists(mp):
                continue
            m = np.array(Image.open(mp))
            if m.ndim == 3:
                m = m[..., 0]
            if (m > 0).sum() == 0:
                continue
            img = np.array(Image.open(ip).convert("RGB"))
            ov = img.copy(); ov[m > 0] = [255, 0, 0]
            rows.append(np.concatenate([img, ov], axis=1))
            if len(rows) >= n:
                break
        return rows
    synth_rows = pick(synth_root, f"{synth_root}/images/*.png",
                      lambda ip: ip.replace("/images/", "/masks/"))
    real_rows = pick(real_img_dir, f"{os.path.dirname(real_img_dir)}/images/*.png",
                     lambda ip: ip.replace("/images/", "/masks/"))
    rows = synth_rows + real_rows
    if rows:
        h = min(r.shape[0] for r in rows)
        grid = np.concatenate([r[:h] for r in rows], axis=0)
        Image.fromarray(grid).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["crack", "craq"], required=True)
    ap.add_argument("--synth_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    real = mask_stats(REAL[args.type])
    synth = mask_stats(os.path.join(args.synth_root, "masks"))
    cmp = compare(real, synth)
    report = {"type": args.type, "real": real, "synth": synth, "compare": cmp}
    # 移除冗長 list 再寫 report
    for d in (report["real"], report["synth"]):
        d.pop("fg_pcts", None)
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    preview_grid(args.synth_root, REAL[args.type].replace("/masks", "/images"),
                 os.path.join(args.out, f"preview_{args.type}.png"))
    print(json.dumps(cmp, ensure_ascii=False, indent=2))
    print(f"report + preview -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 對煙霧輸出跑驗證**

Run:
```bash
cd /home/zzz90/research/data_generation
/home/zzz90/research/datagen_env/bin/python scripts/validate_synth.py --type crack \
  --synth_root /home/zzz90/research/_data/synth_crack_smoke --out runs/2026-06-02-synth-v0-smoke/validate_crack
/home/zzz90/research/datagen_env/bin/python scripts/validate_synth.py --type craq \
  --synth_root /home/zzz90/research/_data/synth_craq_smoke --out runs/2026-06-02-synth-v0-smoke/validate_craq
```
Expected: 各印出 compare JSON(含 `synth_mean_within_real_minmax`)與 report/preview 路徑。**人眼檢視** preview png:合成與真實型態一致。

- [ ] **Step 7: Commit**

```bash
cd /home/zzz90/research
git add data_generation/src/synthgen/validate.py data_generation/scripts/validate_synth.py data_generation/tests/test_validate.py data_generation/runs/2026-06-02-synth-v0-smoke
git commit -m "feat(synthgen): synth-vs-real stat validation + preview grid"
```

---

## Task 11: 正式 v0 全量合成 run

**Files:**
- Create: `research/data_generation/runs/2026-06-02-synth-v0/manifest.md`
- Modify: `research/data_generation/EXPERIMENTS.md`、`README.md`

- [ ] **Step 1: 跑全量合成(8:1, crack 800 / craq 100)**

Run: `cd /home/zzz90/research/data_generation && /home/zzz90/research/datagen_env/bin/python scripts/synthesize.py --config configs/synth_v0.yaml --overwrite`
Expected: 印出 `crack: tiles=... (fg>=800)` 與 `craq: tiles=... (fg>=100)`,reuse 倍數合理(未超 max_reuse 警告或僅少量警告);`_data/synth_crack_v0`、`_data/synth_craq_v0` 產生。

- [ ] **Step 2: 跑驗收驗證**

Run:
```bash
cd /home/zzz90/research/data_generation
/home/zzz90/research/datagen_env/bin/python scripts/validate_synth.py --type crack \
  --synth_root /home/zzz90/research/_data/synth_crack_v0 --out runs/2026-06-02-synth-v0/validate_crack
/home/zzz90/research/datagen_env/bin/python scripts/validate_synth.py --type craq \
  --synth_root /home/zzz90/research/_data/synth_craq_v0 --out runs/2026-06-02-synth-v0/validate_craq
```
Expected: crack/craq 的 `synth_mean_within_real_minmax` 皆為 `true`(crack 真實 range ~[0,2.07]、craq ~[0,9.07]);preview 人眼通過。**若 false:** 回 Task 2/3 調 `target_fg` 或引擎參數,重跑 Step 1。

- [ ] **Step 3: 全測試回歸**

Run: `cd /home/zzz90/research/data_generation && /home/zzz90/research/datagen_env/bin/python -m pytest`
Expected: 全部 passed。

- [ ] **Step 4: 寫 run manifest**

`runs/2026-06-02-synth-v0/manifest.md`:
```markdown
# Run: synth-v0

- 日期: 2026-06-02
- 目的: v0 正式合成 crack:craq = 8:1 (crack 800 / craq 100 fg tiles)
- 指令: `python scripts/synthesize.py --config configs/synth_v0.yaml --overwrite`
- config: configs/synth_v0.yaml(seed 42)
- 環境: venv /home/zzz90/research/datagen_env;crackseg_common editable
- 輸出: _data/synth_crack_v0, _data/synth_craq_v0(TileSegDataset 格式)
- 驗收: TileSegDataset 可讀=<是>;fg 分佈 within real range crack=<>/craq=<>;視覺=<通過>
- 結果數字: crack tiles=<> reuse=<>;craq tiles=<> reuse=<>
- 備註: 下游 real vs real+synth A/B 為後續獨立實驗(本 run 不含)
```

- [ ] **Step 5: 更新 EXPERIMENTS 與 README 狀態**

`EXPERIMENTS.md` 補一列:
```markdown
| 2026-06-02 | synth-v0 | v0 程序式合成 crack:craq=8:1 | within-range crack=<>/craq=<> | <通過/待查> | `runs/2026-06-02-synth-v0/` |
```

`README.md` 的「狀態」更新:目前進度改為「v0 合成器完成,已產出 `_data/synth_{crack,craq}_v0`;下一步下游 A/B 實驗」。

- [ ] **Step 6: Commit**

```bash
cd /home/zzz90/research
git add data_generation/runs/2026-06-02-synth-v0 data_generation/EXPERIMENTS.md data_generation/README.md
git commit -m "feat(synthgen): v0 full synthesis run (8:1) + acceptance record"
```

---

## 驗收對照(spec §8)

- **mask 正確性(硬)**:Task 6 `test_mask_is_pre_blur_binary` + Task 9 Step 4 TileSegDataset 讀取。
- **單元測試(硬)**:Task 1-10 各引擎/模組測試;Task 11 Step 3 全回歸。
  - 引擎 fg% 落 target:Task 2/3 測試。craq 形成 cell:Task 3 `test_forms_cells`。tiling round-trip:Task 7。appearance 不動 mask:Task 6。
- **分佈吻合(軟)**:Task 10/11 `synth_mean_within_real_minmax` + report.json。
- **視覺抽檢(軟)**:Task 9 Step 5 / Task 10 Step 6 preview grid 人眼。
- **下游 A/B**:明確排除於 v0,列後續獨立實驗計畫。
