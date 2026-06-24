# YOLO-prompted SAM2 decoder fine-tune（craquelure）結果總結（2026-05-31）

訓練 SAM2 的 prompt_encoder+mask_decoder(image encoder 凍結),GT 中軸點當 prompt,4-fold
(同 `labeled32_craq_v3` split)。eval 兩模式:oracle(GT 點)/ yolo-point(快取 fold-k craq agent 草稿點)。

## 結果(像素 F1)

| fold | oracle F1 | yolo-point F1 | dense-seg(+CLAHE) | SepSAM YOLO-only |
|------|-----------|---------------|-------------------|------------------|
| 0 | 0.575 | 0.566 | 0.634 | (≈) |
| 1 | 0.593 | 0.536 | 0.614 | |
| 2 | **0.323** | 0.039 | 0.040 | 0.0 |
| 3 | 0.495 | 0.397 | 0.655 | |
| **4-fold 平均** | 0.496 | 0.385 | — | — |
| **排除 fold2 平均** | **0.554** | **0.500** | **0.634** | **0.541** |

## 核心判定

1. **訓練 SAM2 decoder 確實讓 refine 有用(假說成立)**:SAM-only 從 frozen 的 ~0.10 跳到 trained 的 0.50–0.55。「把後面的 SAM2 訓練過」是有效的 —— 凍結 zero-shot 才是之前無用的主因。

2. **但仍贏不了現有方案**:trained-SAM2 + YOLO 點(0.500)**略低於** SepSAM YOLO-only(0.541),且明顯低於 dense-seg(0.634)。即使 decoder 訓練過,接 YOLO 點當 craquelure 專家**並沒有比直接用 YOLO mask 或 dense-seg 好**。

3. **recall 天花板代價(oracle→yolo)≈ 0.054**(排除 fold2:0.554→0.500),在難 fold 更大(fold3 0.495→0.397)。oracle 本身(0.554)也低於 dense-seg(0.634)→ **point-prompt 形式在此尺度本質上弱於 dense seg**,即使給完美點。

4. **fold2(彩繪面板)最有意思**:trained-SAM2 **oracle=0.323**,遠勝 dense-seg(0.04)與 YOLO(0)。代表「只要給對點,訓練過的 SAM2 decoder 連彩繪面板的 craquelure 都分得出來」;fold2 崩潰的根因是**定位(localization)**,不是分割能力。YOLO/dense-seg 在 fold2 是定位失敗,但有 oracle 點時 SAM2 能補上。

## 結論與建議
- **craquelure 專家仍以 dense-seg+CLAHE(0.634)為最佳**;trained-SAM2 點 prompt 雖然證明「訓練 SAM2 有效」,但整體 0.50–0.55,沒超過。
- 有價值的後續:**fold2 的 oracle 0.323 暗示 point-prompt 路線的瓶頸在定位**。若把 prompt 來源換成更好的定位(更高 recall 的偵測 / dense grid 點 / 跨方法融合),trained-SAM2 decoder 有頭。但那會往 dense-seg 收斂。
- frozen-SAM2 CMC(SepSAM2 原樣)對 craquelure 確定無用,應丟棄。

## 產物
- `model_prompted_sam2.py`、`train_promptsam2_craq.py`、`eval_promptsam2_craq.py`、`gt_points.py`
- `outputs/promptsam2_craq_fold{0..3}/best.pt`
- YOLO 點快取 `…/labeled32_craq_v3/tiles_512/craqfold{k}/yolo_points.json`(由 SepSAM2 `scripts/cache_yolo_points.py` 產)
