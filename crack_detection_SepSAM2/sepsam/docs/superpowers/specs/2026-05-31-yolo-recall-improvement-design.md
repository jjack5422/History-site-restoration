# SepSAM2 YOLO 召回提升 — 設計文件

- 日期: 2026-05-31
- 專案: `/home/zzz90/research/crack_detection_SepSAM2/sepsam`
- 環境: venv `/home/zzz90/research/SepSAM2_env`

## 1. 背景與動機

heritage 砌體裂紋 domain 下,CMC pipeline 的 SAM2 精修已被校準停用
(`configs/cmc_heritage.yaml`:SAM-only F1=0.066 / P=0.035,CMC 採用 SAM 0/32,
CMC ≡ YOLO-only)。因此提升整體裂紋分割品質的**唯一有效槓桿是 YOLO agent 本身**,
且改善會直接反映到 CMC 輸出。

4-fold panel-grouped CV(`sepsam_agent_heritage_cv_fold0-3`)實測召回偏低:
- conf=0.25:4-fold 平均 mask F1=0.171,mask recall=0.140
- max-F1 conf(ultralytics 自動):4-fold 平均 mask F1=0.215

主要誤差來源是 **recall 不足**(漏抓密集、低對比的細裂縫),precision 相對較高。

**fold2 是 domain-shift outlier**(val panel KJTHT-SC-L-A4-4 是彩繪/人物畫面板,
與其他風化表面 panel 視覺差異大,mask F1≈0.003)。本設計的槓桿**預期無法修復 fold2**,
報告時 fold2 單獨標註。

## 2. 目標與成功標準

- 主指標:4-fold 平均 **mask recall** 明顯上升(基準 0.140 @conf0.25)。
- 護欄:4-fold 平均 mask **F1 ≥ 基準最佳 0.215**,precision 不崩(設地板,見 Stage 1)。
- 次要:box recall/F1 同步觀察。
- 範圍:先在 4-fold CV 驗證手法,再套用到生產 agent(`sepsam_agent_heritage_ft-2`),
  在 `datasets/heritage_1_31test`(32 張)重評。

## 3. 整體策略

採「**分階段量化**」:免費槓桿(不重訓)先量化成基準,再逐個加入需重訓的槓桿並做 A/B,
確保每個槓桿的 recall/F1 增量可歸因(避免無效重訓)。

SAM2 在 heritage 維持停用,本設計不動 CMC 的 SAM2 決策參數;僅在 Stage 4 因 CLAHE
需要同步推論端前處理。

## 4. 分階段設計

### Stage 0 — Baseline(已有,補記)
記錄各 fold 與 4-fold 平均的 box & mask P/R/F1:
- conf=0.25 操作點
- ultralytics 自動最佳 F1 操作點

現有數值(來源:`runs/segment/runs/sepsam_agent_heritage_cv_fold{0..3}` 的 best.pt 重驗):

| 口徑 | mask F1 (4-fold) | mask recall |
|------|------------------|-------------|
| max-F1 conf | 0.215 | (各 fold P/R 已記錄) |
| conf=0.25 | 0.171 | 0.140 |

### Stage 1 — 免費槓桿掃描(不重訓)
在各 fold val 上用 `YOLO.val` 掃描:
- `conf` ∈ {0.25, 0.10, 0.05, 0.02}
- NMS `iou` ∈ {0.7, 0.5, 0.4}
- `max_det` ∈ {300, 1000}

每組合記錄各 fold 的 box+mask P/R/F1。**選操作點**:在 precision 地板(例如 mask P ≥ 0.15,
最終地板值依掃描結果定)的前提下最大化 mask recall。

產出:
- sweep 腳本(新增,例如 `scripts/sweep_cv_recall.py`)
- 結果表(csv + markdown 摘要)
- 選定的操作點(conf / iou / max_det),供後續所有 stage 的評估沿用

### Stage 2 — CLAHE 重訓
CLAHE 以**確定性前處理烘進 tile**(train 與 inference 一致):
- 對每張 heritage 原圖,在 LAB 色空間對 L 通道套 `cv2.createCLAHE`
  (`clipLimit≈2.0–3.0`,`tileGridSize=(8,8)`),轉回 RGB 後再切 tile。
- 輸出 `datasets/heritage_ft_cv_clahe`,**沿用既有 `folds.json` 的 panel 分組**
  (與非 CLAHE 版完全相同的 train/val 切分 → 乾淨 A/B)。
- 新腳本(例如 `scripts/build_heritage_cv_clahe.py`,或 `build_heritage_cv.py` 加 `--clahe` 旗標)。

用 `finetune_heritage_cv.py` 同超參(epochs=100, imgsz=512, batch=16, patience=30,
其餘 aug/optim 同現有預設)在 CLAHE tile 上重訓 fold0-3,套 Stage-1 操作點評估,
比較 vs Stage 0 baseline 的 recall/F1 增量。

### Stage 3 — 密集面板過採樣重訓
在 Stage 2 較佳的 tile 集(CLAHE 或原圖)上:
- 依每個 panel 的裂縫實例數過採樣 train list(複製 tile 條目)。
- 倍率公式:`repeat = clip(round(panel_instances / median_panel_instances), 1, 3)`(上限 3x)。
- 以 ultralytics 讀 train list 的方式實作(複製條目),val list 不變。
- 新腳本或在 split 產生階段加旗標。

重訓 fold0-3,套相同操作點評估,比較增量。

### Stage 4 — 套到生產 agent
取贏的組合(Stage 1 操作點 + CLAHE + 過採樣):
- 重訓生產 heritage agent(對應 `sepsam_agent_heritage_ft-2` 的全量訓練流程),用 CLAHE 全量資料。
- 在 `datasets/heritage_1_31test` 重評,對照 cmc_heritage 現有的 YOLO-only F1=0.434。
- **一致性關鍵改動**:CLAHE 烘進訓練後,CMC 推論端必須同步套相同 CLAHE。
  - 在 `scripts/infer.py` / `scripts/eval*.py` 餵 `agent.predict(rgb, ...)` 前,
    或 `src/agent.py` 的輸入處,加入相同的 CLAHE 前處理函式(共用一份實作避免 train/infer 不一致)。
- 更新 `configs/cmc_heritage.yaml`:`YOLO_CONF_1` 改為 Stage-1 選定值;
  記錄 CLAHE 參數於 config 註解。

## 5. 元件與介面

- **CLAHE 前處理函式**(共用):`clahe_rgb(img_rgb, clip, grid) -> img_rgb`,
  train tile 建構與 CMC 推論端共用同一份,確保一致。
- **sweep 腳本**:輸入 ckpt + data yaml + 參數網格,輸出 P/R/F1 表。
- **CLAHE tile 建構**:輸入原圖 + masks,輸出 CLAHE tile + 沿用 folds.json。
- **過採樣**:輸入 fold train list + 實例統計,輸出加權後 train list。

## 6. 評估與報告
- 各 fold box & mask P/R/F1 + 4-fold 平均,fold2 單獨標註(且報「排除 fold2」的 3-fold 平均)。
- 每個 stage 對照前一階段的增量,確認每個槓桿是否有效。

## 7. 風險與注意
- fold2 domain-shift:本設計不期待改善,避免被它拉低平均而誤判槓桿無效 → 看 3-fold(排除 fold2)。
- 過採樣可能讓模型偏向密集 panel 外觀,在稀疏 panel 上反而降 precision → 用 A/B 守住 F1 護欄。
- 降 conf 必然降 precision:用 precision 地板約束。
- CLAHE train/infer 不一致是最容易踩的坑 → 共用一份 CLAHE 實作,Stage 4 務必同步推論端。
- ultralytics 8.4.57 對 conf=0.0 的陷阱(繞過 NMS),掃描下限取 0.02 不取 0.0。

## 8. 範圍外(YAGNI)
- 不重新啟用 / 重新校準 heritage 的 SAM2 精修(已證實無益)。
- 不改 YOLOv8 結構(P2 head 等),不加 imgsz(本輪未選)。
- 不做隨機 CLAHE 增強版本。
