# data_generation — 木質文物 crack/craquelure 資料合成

> 專案卡。新人/AI 進來先讀這頁就懂全貌。

## 目標 (Goal)
為**木質文物古蹟**(影像含木紋 wood grain、顏料 pigment、破損 damage、髒汙 soiling 等干擾)
合成劣化資料,標的為兩類:**craquelure(龜裂網)** 與 **crack(裂縫)**。
動機:真實標註資料稀少,且 **crack 樣本相對 craquelure 嚴重不足、類別比例失衡**;
合成資料用來補足下游 crack 偵測/分割專案 [`crack_detection_SepSAM2` / `crack_detection_sam2` / `crack_detection_unet`] 的訓練資料。
目標指標:由下游偵測專案的 F1 / IoU 提升來驗證(尤其 crack 類的 recall)。

## 狀態 (Status)
- 目前進度:專案已從 `/home/zzz90/data_generation` 遷入 research/;核心文獻已歸檔;真實資料已盤點(見 Data)。
- 最近一次有效實驗:尚無(見 EXPERIMENTS.md)。
- 已知問題 / TODO:
  - 合成器尚未開工;需先做 brainstorming 定合成策略(程序式 Bézier vs GAN 風格遷移)。
  - 兩篇核心文獻聚焦繪畫 craquelure,需針對木材紋理/顏料干擾調整。
  - crack vs craquelure 合成比例目標待定(對齊下游失衡問題)。

## 環境 (Environment)
- venv:尚未建立(讀 PDF 暫用 `/tmp/pdfenv`,pymupdf)。
- 共用程式:`research/_lib/crackseg_common`(editable install);共用資料:`research/_data`。

## 進入點 (Entry points)
| 用途 | 指令 |
|---|---|
| 合成 | (待建) `python scripts/synthesize.py --config configs/<...>.yaml` |

## 資料 (Data)
原始資料樞紐:`research/_data/`(共用,勿複製進本專案)。
| 目錄 | 內容 | 數量 / 規格 |
|---|---|---|
| `_data/image/` | 原始木質文物照片 | 29 jpg(最大 67MB)+ 1 json |
| `_data/image_1024_slices/` | 未標註切片池 | 581 張,1024×1024 |
| `_data/labeled32_crack_v3/` | 已標 crack | 32 影像+32 masks(1024²);tiles_512:136 對 |
| `_data/labeled32_craq_v3/` | 已標 craquelure | 32 影像+32 masks(1024²);tiles_512:228 對 |
- masks 二值 {0,1};tiles 附 group-by-stem 4-fold split (seed 42)。
- **失衡量化**:前景像素佔比 crack 0.226% vs craquelure 2.858%(craq ≈ 12.6× crack)。
- **合成目標比例:crack : craquelure = 8 : 1**(crack 樣本嚴重不足,大幅偏向 crack 補償;2026-06-02 由 4:1 調整)。
- 輸出格式:對齊三元組 (clean I, mask M, degraded Ĩ),供下游分割訓練。

## 相關文獻
見 `_literature/INDEX.md`,主題綜述見 `_literature/topics/heritage-crack-craquelure-synthesis.md`。
核心:
- `2026_cuch-guillen_synthetic-craquelure` — 程序式 craquelure 合成 (Bézier + tapered/branching)。
- `2017_zhu_cyclegan` — 非配對影像轉換,可用於提升合成資料真實感。
