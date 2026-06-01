# SepSAM2 復現結果報告

對應 `SepSAM_復現規格書.md` 的 D.5 / M6 / M7。

## 摘要

| 項目 | 結果 |
|---|---|
| D.5 SAM_THRESH 校準（SAM2、valid 100） | `SAM_THRESH=0.80`、`CONFLICTION_RATIO=0.50`，mean F1=0.7297 |
| M6 完整 CMC（test 112） | P=0.6881、R=0.8092、F1=0.7151、IoU=0.5819 |
| M6 完整 CMC（valid 200） | P=0.7188、R=0.8472、F1=0.7471、IoU=0.6108 |
| M7 CMC ablation | YOLO-only 與 CMC 幾近持平（valid F1 +0.002），SAM-only 落後 7 點 |
| M7 SEA ablation | SEA 過擬合漲幅 0.060 vs no-SEA 0.106，下游 CMC F1 +0.009 |

## 1. 環境與設定

| 項目 | 值 |
|---|---|
| venv | `/home/zzz90/research/SepSAM2_env` |
| Python / torch | 3.12.3 / 2.11.0+cu128 |
| ultralytics | 8.4.57（已 patch C2f_SEA） |
| 大模型後端 | SAM2.1 Hiera Base+ (`weights/sam2.1_hiera_base_plus.pt`) |
| Agent | YOLOv8n-Seg + SEA，imgsz=416、batch=16、200 epochs |
| 訓練資料 | `datasets/crack_seg`（Roboflow crack-bphdr v2）3717 train / 200 val / 112 test |
| GPU | RTX 5060 Laptop |

## 2. 規格陷阱：`YOLO_CONF_1: 0.0`

`configs/cmc.yaml` 預設 `YOLO_CONF_1=0.0`（論文 Table 3）。在 ultralytics 8.4.57 中，`model.predict(conf=0.0)` 不會套用內建 NMS 信心門檻，會接收最多 `max_det=300` 個低信心錨點：

- 1604 (valid): conf=0.0 收 282 個 instance、`pred_fg=168900/173056`（97.6% 飽和）、F1=0.030
- 1604 (valid): conf=0.25 收 1 個 instance、`pred_fg=2333`、F1=0.828

這讓 CMC 的草稿與提示點完全失準，校準會挑出 mean F1≈0.10 的偽最佳值（`SAM_THRESH=0.95`、`CONFLICTION_RATIO=1.0`）。修正為 `YOLO_CONF_1: 0.25` 後校準才合理。

該偏離已寫入 `configs/cmc.yaml` 的註解。

## 3. D.5 SAM_THRESH 校準

`scripts/calibrate_sam_thresh.py` 在 valid 100 張上掃描 `SAM_THRESH ∈ {0.60,…,0.95}` 並可同時掃 `CONFLICTION_RATIO`。`scripts/_fine_sweep.py` 為更細的網格（thr step 0.025、conf step 0.10）。

SAM2 score 在此 domain 的分布範圍 `[0.006, 0.883]`，遠低於 SAM v1 常見的 > 0.85 → 論文預設 `SAM_THRESH=0.85` 在 SAM2 後端會拒絕近乎全部 SAM 結果。

Fine sweep top-10（valid 100）：

```
thr=0.800  conf=0.50  F1=0.7297  IoU=0.5878   ← 採用
thr=0.825  conf=0.50  F1=0.7289  IoU=0.5866
thr=0.825  conf=0.60  F1=0.7289  IoU=0.5866
...
Reference: YOLO-only F1=0.7277  SAM-only F1=0.6274
```

採用值已寫回 `configs/cmc.yaml`：

```yaml
SAM_THRESH: 0.80           # D.5 校準：valid 100 張，best F1=0.7297（原論文值 0.85）
CONFLICTION_RATIO: 0.50    # D.5 校準：原論文值 1.50；此資料集 SAM2 改進小，需更嚴的衝突門檻
YOLO_CONF_1: 0.25          # 取代論文 0.0（見第 2 節）
```

## 4. M6 評估

`scripts/eval.py` 跑完整 CMC、像素級 P/R/F1/IoU。

| 集合 | n | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|
| valid | 200 | 0.7188 | 0.8472 | 0.7471 | 0.6108 |
| test  | 112 | 0.6881 | 0.8092 | 0.7151 | 0.5819 |

替代說明：論文 Table 6/7 用的 CFD/PChun/VT mask 資料集未在本機（規格書註：Self-collected 381 張未公開）。本次以 `crack_seg` test/valid 替代，量級與論文 CFD 報告區間相近。

GT 取得：`scripts/yolo_seg_to_masks.py` 把 Roboflow YOLO polygon 標籤 fillPoly 成單通道 mask，輸出 `datasets/crack_seg/{valid,test}/masks/`。valid 200 張、test 112 張，valid 有 1 張無標籤（黑 mask）。

## 5. 質化判讀（`_steps.png`）

校準後重跑 4 張代表性 test 影像，存於 `outputs/m5_cmc_v2/`：

| 影像 | sam_score | conflict | decision | 觀察 |
|---|---|---|---|---|
| 1616 | 0.504 | 0.000 | yolo | SAM raw 完全空白；score < 0.80 → fallback |
| 1675 | 0.648 | 0.851 | yolo | SAM 過擴成 stripe；conflict > 0.5 → 拒絕 |
| 1686 | 0.275 | 8.980 | yolo | SAM 大範圍誤判；conflict 巨大 → 拒絕。**舊 broken config 誤判為 sam，校準後正確修掉** |
| 1706 | 0.434 | 0.316 | yolo | borderline；conflict 通過但 score 未過 → 保守拒絕 |

CMC 在每張上以正確理由拒絕 SAM；同個資料集 SAM2 score 偏低使 CMC 變得相對保守。

## 6. M7 Ablation：有/無 CMC（重現 Fig. 15）

`scripts/m7_ablation_cmc.py` 同時記錄 YOLO-only / SAM-only / CMC 的指標。

| 集合 | 配置 | P | R | F1 | IoU |
|---|---|---|---|---|---|
| valid (200) | YOLO-only | 0.7103 | 0.8529 | 0.7453 | 0.6079 |
| valid (200) | SAM-only  | 0.6325 | 0.8179 | 0.6579 | 0.5319 |
| valid (200) | **CMC**   | **0.7186** | 0.8473 | **0.7472** | **0.6108** |
| test (112)  | YOLO-only | 0.6896 | 0.8220 | 0.7203 | 0.5852 |
| test (112)  | SAM-only  | 0.5949 | 0.7911 | 0.6293 | 0.5062 |
| test (112)  | CMC       | 0.6912 | 0.8109 | 0.7181 | 0.5863 |

- CMC 採用 SAM 的比例 13.4–13.5%
- SAM-only 落後 YOLO-only 7-9 個 F1 點 → SAM2 在此分布傾向過擴
- CMC 相對 YOLO-only 提升極小：論文 Fig. 15 的明顯增益在此資料集未重現。歸因：
  1. crack-bphdr GT 較粗實，YOLO Agent 已可貼合
  2. SAM2 score 分布偏低，CMC 選擇變保守
  3. 預期在 CFD / PChun（細裂、邊界精細）domain 上 CMC 提升較大

## 7. M7 Ablation：有/無 SEA（重現 Fig. 13）

訓練 vanilla `yolov8n-seg.yaml`（同 epochs/imgsz/batch/data）做 baseline，runs/`sepsam_agent_v8n_noSEA_200ep`。`scripts/m7_ablation_sea.py` 同時比對 val 曲線與下游 CMC F1。

| 指標 | SEA | no-SEA |
|---|---|---|
| mAP50(M) peak | 0.7267 @ep109 | 0.7411 @ep126 |
| mAP50(M) final | 0.7090 | 0.7026 |
| mAP50-95(M) final | 0.2581 | 0.2534 |
| val/seg_loss 最低 | 1.1654 @ep74 | 1.1676 @ep66 |
| val/seg_loss 最終 | 1.2258 | 1.2739 |
| **val/seg_loss 最終−最低漲幅** | **0.060** | **0.106** |
| **最終 val−train seg_loss gap** | **0.285** | **0.333** |

no-SEA 的 mAP50(M) 峰值較高但末值更低，val_loss 過了最低點後漲幅是 SEA 的 1.76 倍，符合論文「SEA 主要功能是抑制過擬合」的敘述。

下游 CMC（test 112，相同 SAM2 + 同 CMC 超參數）：

| Agent | P | R | F1 | IoU |
|---|---|---|---|---|
| SEA    | 0.6902 | 0.8107 | 0.7174 | 0.5852 |
| no-SEA | 0.6745 | 0.8149 | 0.7084 | 0.5730 |
| **Δ (SEA - no-SEA)** | +0.0157 | -0.0042 | **+0.0090** | **+0.0122** |

方向與 Fig. 13 一致，但量級小。論文建議 ablation 跑 ≥300 epochs（≥500 epochs 最佳）才能完整展現過擬合差距；本次 200 epochs 已可看到趨勢。

## 8. 產出檔案

新增腳本：

| 路徑 | 用途 |
|---|---|
| `scripts/yolo_seg_to_masks.py` | YOLO polygon label → 二值 mask PNG |
| `scripts/_fine_sweep.py` | D.5 細粒度 thr×conf 掃描 + YOLO-only/SAM-only 參考線 |
| `scripts/m7_ablation_cmc.py` | YOLO-only vs SAM-only vs CMC 對照 |
| `scripts/m7_ablation_sea.py` | SEA vs no-SEA val 曲線 + 下游 CMC 對照 |

修改：

| 路徑 | 內容 |
|---|---|
| `configs/cmc.yaml` | `SAM_THRESH=0.80`、`CONFLICTION_RATIO=0.50`、`YOLO_CONF_1=0.25`，並補上來由註解 |

訓練輸出：

| 路徑 | 內容 |
|---|---|
| `runs/segment/runs/sepsam_agent_v8n_200ep/` | SEA Agent（已有） |
| `runs/segment/runs/sepsam_agent_v8n_noSEA_200ep/` | no-SEA baseline（本次新增） |

質化圖：

| 路徑 | 內容 |
|---|---|
| `outputs/m5_cmc/`    | 校準前 4 張 _steps/_mask/_overlay（保留作對照） |
| `outputs/m5_cmc_v2/` | 校準後 4 張 _steps/_mask/_overlay |

衍生資料：

| 路徑 | 內容 |
|---|---|
| `datasets/crack_seg/valid/masks/` | 由 YOLO polygon 轉的 valid GT mask（200 張，1 張空白） |
| `datasets/crack_seg/test/masks/`  | 由 YOLO polygon 轉的 test GT mask（112 張） |

## 9. 復現指令

```bash
VENV=/home/zzz90/research/SepSAM2_env/bin/python
cd /home/zzz90/research/crack_detection_SepSAM2/sepsam

# 0. GT mask 生成（如未產生）
$VENV scripts/yolo_seg_to_masks.py \
    --images datasets/crack_seg/valid/images \
    --labels datasets/crack_seg/valid/labels \
    --out    datasets/crack_seg/valid/masks
$VENV scripts/yolo_seg_to_masks.py \
    --images datasets/crack_seg/test/images \
    --labels datasets/crack_seg/test/labels \
    --out    datasets/crack_seg/test/masks

# 1. D.5 校準
$VENV scripts/calibrate_sam_thresh.py \
    --images datasets/crack_seg/valid/images \
    --masks  datasets/crack_seg/valid/masks \
    --limit 100 --sweep-conflict
$VENV scripts/_fine_sweep.py \
    --images datasets/crack_seg/valid/images \
    --masks  datasets/crack_seg/valid/masks --limit 100

# 2. M6 評估
$VENV scripts/eval.py --images datasets/crack_seg/valid/images --masks datasets/crack_seg/valid/masks
$VENV scripts/eval.py --images datasets/crack_seg/test/images  --masks datasets/crack_seg/test/masks

# 3. 質化圖
$VENV scripts/infer.py --source datasets/crack_seg/test/images/<stem>.jpg \
    --out outputs/m5_cmc_v2 --dump-steps

# 4. M7 ablation
$VENV scripts/m7_ablation_cmc.py --images datasets/crack_seg/test/images --masks datasets/crack_seg/test/masks
$VENV scripts/m7_ablation_sea.py \
    --sea_run  runs/segment/runs/sepsam_agent_v8n_200ep \
    --base_run runs/segment/runs/sepsam_agent_v8n_noSEA_200ep \
    --images   datasets/crack_seg/test/images \
    --masks    datasets/crack_seg/test/masks
```

## 10. 後續可選工作

- 取得 CFD（118 張）/ PChun（98 張）/ VT（1096 val）對照論文 Table 6/7 原始量級
- 把 ablation epochs 拉到 500（spec 建議），更完整展現 SEA 過擬合抑制
- 依裂紋寬度分桶（`width>3`、`width>20`）的指標，需先對 GT 跑距離轉換；目前 `src/metrics.py` 已留 TODO
- 細掃 `POINTS_DIVISOR`（目前固定 50），看 SAM 提示點數對 SAM2 後端的影響
