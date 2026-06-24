# craquelure 線條後處理 + 正確指標重評（3 方法）設計文件

- 日期: 2026-05-31
- 程式落點: `/home/zzz90/research/crack_detection_sam2`(post-proc/指標/dense-seg/PromptedSAM2 都在此;SepSAM-YOLO 的 dump 在 SepSAM2 專案、SepSAM2 env)
- 環境: `sam2_env`(主)+ `SepSAM2_env`(僅 dump YOLO 預測)
- 依據: Notion「標註規範與 Segmentation Pipeline 設計」的後處理流程 + 評估策略(clDice / tolerant IoU / skeleton-level P/R;vanilla IoU 對細線不公平)

## 1. 背景與動機

目前所有 craquelure 評估都用 **vanilla pixel-F1、無後處理**。Notion 明確指出:(a)細線任務 vanilla IoU 對 1px 對位極敏感(3px 線平移 1px → IoU~30%),應改用 clDice / tolerant / skeleton-level 指標;(b)Model 1 流程含後處理(skeletonize、connected components 雜訊抑制)。fold2 的 PromptedSAM2 預測滿屏 FP 碎塊正是未後處理的原始輸出。

本實驗:對三個 craquelure 方法套用相同後處理 + 改用對細線公平的指標重新對照,判定換口徑後名次/差距是否改變。

## 2. 目標與成功標準

- 產出三方法在 {raw, post-proc} × {vanilla-F1, tolerant-F1@3px, clDice} 的 4-fold 對照表(含排除 fold2)。
- **成功標準**:
  1. 量化後處理(CC 雜訊抑制)對各方法的提升;
  2. 判定改用 clDice/tolerant 後,PromptedSAM2 / dense-seg / SepSAM-YOLO 的相對名次是否改變(vanilla 對 SAM2 滿屏 FP 的過重懲罰,在 tolerant/clDice 下可能不同)。

## 3. 三方法與預測來源（各自 dump per-tile 預測 mask）

統一在同一份 craq val(`labeled32_craq_v3` 4-fold,fold 對齊)上產生二值 craquelure 預測,存 `preds/{method}/fold{k}/{stem}.png`:
- **promptsam2_oracle / promptsam2_yolo**(sam2_env):載 `outputs/promptsam2_craq_fold{k}/best.pt`,GT 點 / YOLO 點兩模式(沿用 `eval_promptsam2_craq.py` 的 predict 邏輯)。
- **denseseg**(sam2_env):載 dense-seg craquelure expert(`outputs/expert_craq_v3_clahe_fold{k}` 最佳),**輸入需套 CLAHE**(該 run 是 clahe 版),取 craquelure 類別二值 mask。沿用其訓練/predict 路徑。
- **sepsam_yolo**(SepSAM2 env):載 `sepsam_agent_craq_cv_fold{k}`,conf=0.05/iou=0.5 的 instance union mask。dump 成 png。

> 跨 env:dump 階段各用各的 env 把預測寫成 png;評估階段(sam2_env)只讀 png + GT,不在同進程混 env。

## 4. 共用模組 `lineproc.py`（post-proc + 指標）

- `cc_filter(mask_bool, min_area)` → 去除像素數 < min_area 的連通元件(**主力後處理:清雜訊 FP**)。`scipy.ndimage.label`。
- `skeleton_centerline(mask_bool, width=3)` → `skimage.morphology.skeletonize` 後 dilate 到 width px(對齊 Notion centerline;主要影響 vanilla)。
- `cldice(pred_bool, gt_bool)` → centerline Dice(Tprec = |skel(pred)∩gt| / |skel(pred)|,Tsens = |skel(gt)∩pred| / |skel(gt)|,clDice = 2·Tprec·Tsens/(Tprec+Tsens))。
- `tolerant_f1(pred_bool, gt_bool, tol=3)` → 容差比對:pred 命中「gt 膨脹 tol」算 TP,gt 命中「pred 膨脹 tol」算 recall,組 F1。
- `skeleton_pr(pred_bool, gt_bool, tol=2)` → skeleton 點層級 precision/recall(可選)。
- 邊界情況:pred 與 gt 皆空 → 各指標回 1.0;單邊空 → 0.0。

## 5. 評估流程 `eval_lineproc_craq.py`

對每個 method × fold,讀 `preds/{method}/fold{k}/*.png` 與對應 GT:
1. raw 版:直接算 vanilla-F1 / tolerant-F1@3 / clDice。
2. post-proc 版:`cc_filter(min_area)` (+ 可選 `skeleton_centerline`) 後再算三指標。
彙整 4-fold 平均 + 排除-fold2,輸出對照表(method × {raw,pp} × 3 指標)。
參數:`min_area`(預設 64,對齊 YOLO 標籤 min_area)、`tol=3`。

## 6. 元件（檔案）
- `lineproc.py`(新,sam2 專案)— post-proc + 指標,純函式、可單元測試。
- `dump_preds_*.py`:`dump_preds_promptsam2.py`(sam2_env)、`dump_preds_denseseg.py`(sam2_env)、`scripts/dump_preds_sepsam_yolo.py`(SepSAM2 env)。各輸出 `preds/{method}/fold{k}/{stem}.png`。
- `eval_lineproc_craq.py`(新,sam2 專案)— 統一讀 png + GT,raw/pp × 3 指標,對照表。
- `tests/test_lineproc.py`(新)— cldice/tolerant/cc_filter 在合成 mask 上的單元測試。

## 7. 風險與注意
- fold2(彩繪面板)的 FP 來自紋理混淆,CC 濾雜訊未必救得了(碎塊可能不小);如實看數據。
- dense-seg 是 CLAHE 版,dump 時務必對輸入套相同 CLAHE,否則 domain 不一致、低估 dense-seg。
- clDice/tolerant 對寬度不敏感 → skeleton_centerline 對它們影響小,主要修 vanilla;headline 看 clDice/tolerant。
- GT 線寬未知:不影響 clDice/tolerant 結論。
- 預測座標/解析度:dump 時統一存回原 tile 尺寸(512)的二值 png,GT 同尺寸,直接比對。

## 8. 範圍外（YAGNI）
- 不做拓樸判類(crack vs craquelure)— 本實驗單類 craquelure。
- 不含 crack(另案)。
- 不重訓任何模型 — 只後處理 + 重評既有模型輸出。
- 不調 min_area/tol 以外的後處理(先用 CC-filter + 可選 centerline 看 baseline)。
