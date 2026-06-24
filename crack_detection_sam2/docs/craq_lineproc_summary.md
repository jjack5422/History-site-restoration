# craquelure 後處理 + 正確指標重評 結果總結（2026-05-31）

四個 craquelure 方法,同一份 craq val(labeled32_craq_v3,排除 fold2 平均),**每張圖 F1 取平均(per-image macro)** 的一致口徑。指標:vanilla pixel-F1 / tolerant-F1@3px / clDice。

## 結果（排除 fold2 平均）

**raw(無後處理)**
| method | vanilla | tolerant | clDice |
|--------|---------|----------|--------|
| denseseg | 0.532 | 0.724 | 0.656 |
| sepsam_yolo | 0.536 | 0.686 | 0.615 |
| promptsam2_oracle | 0.554 | **0.783** | **0.686** |
| promptsam2_yolo | 0.500 | 0.707 | 0.627 |

**post-proc(cc_filter min_area=64 + skeleton_centerline 3px)**
| method | vanilla | tolerant | clDice |
|--------|---------|----------|--------|
| denseseg | 0.517 | 0.702 | 0.641 |
| sepsam_yolo | 0.474 | 0.662 | 0.594 |
| promptsam2_oracle | 0.552 | 0.770 | 0.669 |
| promptsam2_yolo | 0.499 | 0.695 | 0.612 |

## 核心發現

1. **正確指標證實 Notion 的論點**:tolerant/clDice 遠高於 vanilla(例:denseseg 0.532→0.72→0.66)。細線預測「幾何上很接近 GT」,vanilla pixel-F1 對 1px 對位的懲罰過重、低估了所有方法。

2. **換正確指標後名次改變 — 兩個翻盤**:
   - vanilla 下 sepsam_yolo(0.536)≈ denseseg(0.532);但 **tolerant/clDice 下 denseseg 明顯領先 sepsam_yolo**(0.724 vs 0.686;0.656 vs 0.615)。
   - **promptsam2_yolo(訓練過 SAM2 + YOLO 點)在正確指標下反超 sepsam_yolo(frozen YOLO-only)**:vanilla 0.500 < 0.536,但 tolerant 0.707 > 0.686、clDice 0.627 > 0.615。→ 訓練過的 SAM2 產生「幾何對齊更好但 vanilla 過度懲罰」的線,正是先前推測的效果。
   - promptsam2_oracle 在所有指標最佳(但用 oracle 點,非公平部署條件)。

3. **後處理(CC 濾雜訊 + centerline)全面略降**:對 sepsam_yolo 最傷(vanilla 0.536→0.474)。skeleton_centerline 把預測削細、GT 又非純 centerline,cc_filter 也誤刪細元件 → 這組後處理在此資料**淨負面**。真正改變結論的是**指標選擇**,不是後處理。

4. **口徑修正(誠實)**:dense-seg 舊報的 0.634 是訓練 `ConfusionMeter` 的 **micro(dataset 級彙總)**;本次四方法統一用 **macro(per-image 平均)**,故 denseseg 在此為 0.532(SAM2 三法 macro 與舊值吻合)。先前 dense-seg vs 其他的對照其實混用了 micro/macro,本次統一後才是公平比較。

## 結論
- **vanilla pixel-F1 不該當主指標**(Notion 早已指出);改 tolerant/clDice 後結論不同。
- 在一致 macro + 正確指標下:**dense-seg ≈ promptsam2(trained-SAM2)為第一梯隊,且 trained-SAM2+YOLO 略勝 frozen YOLO-only**;oracle 上限最高。
- 這組 CC+centerline 後處理無益,建議**只換指標、不套此後處理**(或改更輕的雜訊濾除參數再試)。

產物:`lineproc.py`、`dump_preds_{promptsam2,denseseg}.py`、`scripts/dump_preds_sepsam_yolo.py`(SepSAM2)、`eval_lineproc_craq.py`、`preds/{method}/fold{k}/*.png`。
