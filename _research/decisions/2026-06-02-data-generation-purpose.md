# Decision: data_generation 專案目的與定位

- 日期：2026-06-02
- 對應計畫：`_research/plans/2026-06-02-data-generation-craquelure-crack-synth.md`

## 專案目的 (Purpose)
為**木質文物古蹟**(影像含木紋 wood grain、顏料 pigment、破損 damage、髒汙 soiling 等干擾因子)
建立**劣化資料合成 pipeline**,合成兩類劣化標的:
- **craquelure**(龜裂網,密集細網狀)
- **crack**(裂縫,稀疏長裂)

## 動機 (Why)
1. **真實標註資料稀少**:木質文物裂縫的公開像素級資料常僅數百張(文獻佐證:SMG-Net 256 原圖、Zhang 2024 約 501 張)。
2. **類別比例嚴重失衡**:相對 craquelure,**crack 樣本數嚴重不足**,使下游偵測器在 crack 類 recall 偏低。
3. 合成資料可量產對齊 ground-truth,直接餵給下游 crack 偵測/分割專案,並讓我們**主動控制 crack:craquelure 的比例**以對症失衡。

## 定位 (Positioning)
- 上游姊妹專案,服務 `crack_detection_SepSAM2` / `crack_detection_sam2` / `crack_detection_unet`。
- 成效以**下游偵測 F1/IoU(尤其 crack recall)的提升**驗證,而非合成本身的美觀。

## 技術路線(初步,待 brainstorming 定案)
- **幾何走程序式**:Bézier + tapered + branching(參考 Cuch-Guillén 2026),可控、自帶精確 mask;crack 用稀疏長裂+順木紋走向,craquelure 用密集網狀。
- **真實感走域轉換**:CycleGAN(Zhu 2017)做 synthetic→real-wood sim-to-real,只改外觀紋理、不動裂縫幾何。
- **失衡兩條腿**:合成端調比例 + 訓練端用 focal/Dice/masked hybrid loss(參考 imbalanced-loss 文獻)。
- 輸出標準:對齊三元組 (clean I, mask M, degraded Ĩ)。

## 木材域增量 (研究貢獻點)
既有合成法多針對繪畫/路面/混凝土;本專案需加入**木紋導向裂縫走向、節點(knot)裂、髒汙/破損干擾**的建模 — 此為相對文獻的新增部分。

## 文獻
見 `_literature/topics/heritage-crack-craquelure-synthesis.md` 與 `INDEX.md`。

## 真實資料現況 (2026-06-02 盤點，位於 `research/_data/`)
- `image/`:29 張原始木質文物照片(+1 json);`image_1024_slices/`:581 張 1024² 未標註切片。
- `labeled32_crack_v3/`:32 影像+32 masks(1024²),tiles_512 = 136 對。
- `labeled32_craq_v3/`:32 影像+32 masks(1024²),tiles_512 = 228 對。
- masks 二值 {0,1};tiles 附 group-by-stem 4-fold split(seed 42)。
- **失衡量化**:前景像素佔比 crack 平均 0.226% vs craquelure 2.858%(craq ≈ 12.6× crack)→ 證實 crack 訊號嚴重不足(像素級)。

## 已定：合成目標比例
**crack : craquelure = 8 : 1**(crack 樣本嚴重不足,合成配額大幅偏向 crack;2026-06-02 使用者確認,當日由 4:1 上調為 8:1)。

## 待決 (Open)
- 真實木質文物影像作為 CycleGAN target 域與最終評估集的劃分方式尚未確認(可用 581 未標註切片 + 64 已標)。
- 是否引入較新生成法(CUT / diffusion translation)取代 CycleGAN,待後續文獻。
