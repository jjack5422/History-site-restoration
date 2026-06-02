# Spec: 木質彩繪文物 crack/craquelure 程序式合成器 v0

- 日期：2026-06-02
- 專案：`research/data_generation`
- 對應規則：`research/_research/RULES.md`
- 對應計畫/決策：`_research/plans/2026-06-02-data-generation-craquelure-crack-synth.md`、`_research/decisions/2026-06-02-data-generation-purpose.md`
- 相關文獻：`_literature/topics/heritage-crack-craquelure-synthesis.md`(核心 [[2026_cuch-guillen_synthetic-craquelure]]、[[2017_zhu_cyclegan]])

## 1. 目標與範圍 (Goal & Scope)

為**木質彩繪文物**(影像含彩繪顏料、木紋、破損、髒汙等干擾)合成劣化資料,
v0 為**純程序式合成器**,產出兩個**分型別 binary** 資料集供下游 crack 偵測/分割訓練:
- `_data/synth_crack_v0/` 與 `_data/synth_craq_v0/`,tile 數比 **crack : craquelure = 8 : 1**(crack 樣本嚴重不足,大幅偏向 crack 補償)。
- 輸出格式對齊 `crackseg_common.TileSegDataset`:`<root>/tiles_512/{images,masks}/<name>.png`(mask 二值 {0,1})+ `tile_index.json`。

**In scope:** 雙引擎幾何生成(crack=Bézier、craquelure=Voronoi)、資料驅動外觀渲染、底圖篩選、切片與索引、合成器自體驗收(統計+視覺+單元測試)。

**Out of scope(後續階段):**
- GAN / CycleGAN sim-to-real 真實感增強(第二階段)。
- 下游 real vs real+synth 的 A/B 訓練驗證(獨立實驗計畫;此 spec 不驗收 4:1 是否真的提升下游指標)。
- 多類別共存於同圖的 5-class 合成(v0 採分型別)。
- inpaint 乾淨底圖(C 路線)僅作**並行評估**,不擋 v0 主線。

## 2. 背景事實 (Context, 2026-06-02 盤點)

真實資料於 `research/_data/`:
| 目錄 | 內容 | 數量 / 規格 |
|---|---|---|
| `image/` | 原始照片 | 29 jpg + 1 json |
| `image_1024_slices/` | 未標註切片(合成底圖池) | 581,1024×1024 |
| `labeled32_crack_v3/` | 已標 crack | 32 影像+32 masks(1024²,{0,1});tiles_512=136 對 |
| `labeled32_craq_v3/` | 已標 craquelure | 32 影像+32 masks(1024²,{0,1});tiles_512=228 對 |

- **失衡量化**(前景像素佔比):crack 平均 0.226%(max 2.07%)、craquelure 平均 2.858%(max 9.07%);craq ≈ 12.6× crack。
- **乾淨底圖稀少**(`scripts/scan_clean_slices.py` black-hat proxy 掃描 581 張):proxy median 10.3%;`<0.5%` 僅 6 張、`<1.5%` 累計 31 張、`<4%` 累計 107 張。且 proxy 判定最乾淨者仍帶低對比龜裂 → 真正乾淨底圖極少,故 v0 以 light 級(~31 張)小量起步,並並行評估 inpaint(C)。
- 下游介面:`crackseg_common.TileSegDataset` 讀 `tiles_root/{images,masks}/<name>` + `tile_index.json`,mask 為整數類別;5-class 全集 `["background","crack","loss","shrinkage","craquelure"]`,亦支援 binary。v0 產 binary {0,1}。

## 3. 架構 (Architecture)

可重用模組於 `data_generation/src/synthgen/`,入口腳本於 `scripts/`,設定於 `configs/`。

| 模組 | 職責 | 介面(概) | 依賴 |
|---|---|---|---|
| `base_selection.py` | black-hat proxy 評分篩乾淨底圖 | `score(path)->float`；`build_manifest(slices_dir, thresh)->json` | cv2, numpy |
| `appearance.py` | 擬合真實裂縫外觀 profile;渲染 mask 上底圖 | `fit_profile(img_dir,mask_dir)->npz`；`render(base, geo_mask, profile, cfg)->img` | numpy, cv2 |
| `geometry/crack.py` | Bézier+tapered+branching crack mask | `generate(size, params, rng)->mask` | numpy |
| `geometry/craquelure.py` | Voronoi cell 邊界 craquelure mask | `generate(size, params, rng)->mask` | numpy, scipy.spatial |
| `compose.py` | base+geo_mask→appearance→(img,mask) 1024 對 | `compose(base, geo_mask, type, cfg)->(img,mask)` | 上述 |
| `tiling.py` | 1024 對切 512 tiles + 寫檔 + 累積 index | `tile_and_write(img,mask,name,out_root)->[tile_names]` | numpy, cv2 |

入口腳本:
- `fit_appearance_profile.py` — 由 `labeled32_*` 產 `data/appearance/appearance_profile_{crack,craq}.npz`(一次性、快取)。
- `select_bases.py` — 由 `image_1024_slices` 產 `data/base_manifest.json`(一次性)。
- `synthesize.py` — 主流程,讀 `configs/synth_v0.yaml`,產出 `_data/synth_{crack,craq}_v0/` + run manifest。
- `validate_synth.py` — 合成 vs 真實統計比對 + preview grid → run 目錄。
- `inpaint.py`(選配)— C 路線評估,不在 v0 主流程。

落點(符合 RULES):合成**資料集**→ `_data/synth_*_v0/`(共用,下游直接指向);**run 記錄/log/preview**→ `data_generation/runs/<date>-synth-v0/`;可重用程式→ `src/`;一次性腳本→ `scripts/`;設定→ `configs/`。

## 4. 資料流 (Data flow)

```
(一次性) fit_appearance_profile.py:
   labeled32_{crack,craq}_v3 → appearance_profile_{crack,craq}.npz
       對每個 mask 前景像素，統計其相對局部鄰域(如 7x7,排除前景)的
       ΔL(明度差)與 Δ(a,b)(色彩差)分佈，per type 存均值/標準差/分位數。

(一次性) select_bases.py:
   image_1024_slices(581) → black-hat proxy 評分 → base_manifest.json
       門檻 clean_thresh(預設 0.015);記錄每張分數;符合者列為 base 池。

(主流程) synthesize.py (讀 config, 固定 seed, per-type 子 rng):
   for type, n_target in [(crack, target_tiles.crack), (craq, target_tiles.craq)]:
     while 已寫 tile 數 < n_target:
       base   = base 池抽樣(不足→同底圖換 geometry seed 重用，計重用倍數)
       geo    = geometry[type].generate(slice_size, cfg[type], rng)
       if geo 退化(fg% 不在 target_fg / 空) → 重抽(連續失敗計數)
       img,msk= compose(base, geo, type, cfg)         # img=外觀渲染後; msk=geo 二值(blur前)
       names  = tiling.tile_and_write(img, msk, name, out_root[type])
   寫 tile_index.json(+ 可選 group split) 與 runs/<date>-synth-v0/manifest.md

(驗收) validate_synth.py:
   合成統計 vs labeled32 統計 → report.md + preview grids → runs/<date>-synth-v0/
```

## 5. 關鍵演算法 (Algorithms)

**crack 引擎**(沿用 Cuch-Guillén 2026):
- cubic Bézier `B(t)`,端點 p0,p3 均勻取樣於影像域,內控制點 p1,p2 加高斯擾動產生彎曲;曲線取樣 80–180 點。
- tapered 粗細:沿曲線畫實心圓盤,半徑 `r(t) ~ N(α(1−|t−0.5|), σ_r²)`(兩端細中段粗)。
- 機率 `branch_p` 旋轉+縮放局部方向向量產生分支。
- **稀疏**:少量長曲線(`n_curves` 範圍),目標 fg% 落於真實 crack 分佈(均 0.226%、上限 ~2%)。

**craquelure 引擎**(Voronoi):
- 撒種子點(密度由 `cell_px` 控制,= 目標 cell 邊長像素),加 `jitter` 擾動位置。
- Delaunay/Voronoi 鑲嵌,取 cell 邊界為裂縫網;邊界以 `break_p` 隨機斷裂模擬不連續,線寬 `edge_w` 細。
- 目標 fg% 落於真實 craq 分佈(均 2.858%、上限 ~9%);形成島狀 cell 被裂縫包圍的結構。

**appearance 渲染**(資料驅動):
- geo_mask → erosion(`erosion` 調粗細)→ Gaussian blur(`blur_sigma` 軟化滲色)得 soft-alpha。
- 裂縫像素值 = `base + 抽樣自 profile 的 ΔL/Δcolor`(crack 偏深、craq 偏淺細);以 soft-alpha 混合。
- 強制 `min_contrast`:暗區裂縫對比不足時拉開,避免隱形。
- **GT mask = blur 前的二值 geo_mask**;外觀與 GT 解耦,保證像素級精確標註。

## 6. 設定 (`configs/synth_v0.yaml`,初值可調)

```yaml
seed: 42
output: {crack: _data/synth_crack_v0, craq: _data/synth_craq_v0}
ratio: {crack: 8, craq: 1}
target_tiles: {crack: 800, craq: 100}
tile_size: 512
slice_size: 1024
base:
  manifest: data/base_manifest.json
  clean_thresh: 0.015
  allow_reuse: true
  max_reuse: 20            # 重用倍數上限，超過發警告
crack: {n_curves: [2, 6], taper_alpha: 2.0, taper_sigma: 0.5, branch_p: [0.3, 0.5], target_fg: [0.05, 2.0]}
craq:  {cell_px: [25, 60], jitter: 0.3, edge_w: 1, break_p: 0.1, target_fg: [0.5, 9.0]}
appearance: {profile_dir: data/appearance, min_contrast: 12, erosion: 2, blur_sigma: 2}
keep_negative_ratio: 0.1   # 切片後保留的零前景背景 tile 比例上限
max_regen_fail: 50         # 單樣本連續退化重抽上限
```

每次 run 寫 `runs/<date>-synth-v0/manifest.md`:config 快照、seed、`base_manifest` 版本、git/env(venv、commit)、實際產出 tile 數、base 重用倍數、正/負 tile 比。完成後 `EXPERIMENTS.md` 補一列。

## 7. 錯誤處理 / 邊界 (Error handling)

- **base 不足**:同底圖換 geometry seed 重用,記重用倍數;超過 `max_reuse` 發警告(不靜默)。
- **退化 geometry**(空 mask / fg% 超出 `target_fg`):重抽;連續失敗達 `max_regen_fail` 跳過並計數記入 manifest。
- **暗區低對比**:appearance 強制 `min_contrast`。
- **零前景 tile**:依 `keep_negative_ratio` 保留部分背景負樣本,記錄正/負 tile 比;其餘丟棄。
- **輸出已存在**:報錯要求 `--overwrite`,不默默覆蓋既有 `_data` 資料集。
- **可重現**:單一 `seed` 衍生 per-type 子 rng;相同 config+seed+base_manifest → 相同輸出。

## 8. 驗收 (Acceptance — v0 輕量級)

硬性(必過):
1. **mask 正確性**:GT=geo 二值 mask;單元測試驗 compose 前後 mask 對齊不變。
2. **單元測試**:兩引擎輸出 fg% 落於 `target_fg`;craq 連通性形成 cell(島狀區塊數 > 閾值);tiling round-trip(切片+索引)正確;appearance 不改動 mask。

軟性(達標即 v0 完成):
3. **分佈吻合**:合成 vs `labeled32` 真實統計,比對 fg%、crack 線寬/長度/走向、craq cell 大小分佈,數字落在真實 range 內(寫入 `validate_synth` 的 `report.md`)。
4. **視覺抽檢**:synth vs real preview grid(各型別),人眼確認真實感與型態合理。

下游 A/B(real vs real+synth 對 crack F1/IoU/recall)列為**後續獨立實驗計畫**,非本 spec 驗收項。

## 9. 並行評估:C inpaint 路線(非阻擋)

`inpaint.py` 對選中底圖先以 OpenCV inpaint(或論文 MTM/AD)填掉既有裂縫得乾淨版;
在 `validate_synth` 中比對「篩乾淨底圖」vs「inpaint 乾淨底圖」的殘留劣化與合成品質。
若 inpaint 顯著改善且 artifact 可接受,列為 v1 預設底圖策略。

## 10. 待決 (Open questions)

- 真實木質文物影像作為 CycleGAN target 域 / 最終評估集的劃分(可用 581 未標 + 64 已標),待第二階段。
- 是否引入較新生成法(CUT / diffusion translation)取代 CycleGAN,待後續文獻。
- `target_tiles` 規模初值(crack 800 / craq 100,8:1)是否足夠驗證,可於首次 run 後依下游需求調整。

## 11. 環境

讀 PDF/影像分析暫用 `/tmp/pdfenv`(pymupdf/pillow/numpy/opencv/scipy)。
專案正式 venv 待 writing-plans 階段決定(預期需 numpy/scipy/opencv/pillow;與下游共用 `_lib/crackseg_common`,editable install)。
