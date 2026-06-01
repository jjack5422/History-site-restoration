# SepSAM craquelure 專家 — 4-fold CV 實驗 設計文件

- 日期: 2026-05-31
- 執行專案(pipeline 程式): `/home/zzz90/research/crack_detection_SepSAM2/sepsam`
- 資料來源: `/home/zzz90/research/crack_detection_sam2/data/labeled32_craq_v3/tiles_512`
- 環境: venv `/home/zzz90/research/SepSAM2_env`(ultralytics + SAM2);資料/對照數字來自 `crack_detection_sam2`(venv `sam2_env`)

## 1. 背景與動機

最終目標是兩個獨立專家(crack / craquelure)。craquelure 目前已有 **dense-seg 專家**(`crack_detection_sam2` 的 FPN-on-SAM2-encoder),per-fold craqF1 ≈ 0.634 / 0.614 / 0.04(fold2 崩) / 0.655。本實驗評估 **SepSAM(YOLOv8-Seg SEA agent + SAM2 CMC)能否當 craquelure 專家**。

核心假說:之前在 **crack**(細線、稀疏)上 SAM2 點 prompt 精修有害(SAM-only F1≈0.07,CMC≡YOLO-only)。但 **craquelure 是成片網狀「區域型」紋理**,SAM2 的點 prompt 本就適合區域物件 → 對 craquelure,CMC 的 SAM2 refine **可能反而有正貢獻**。本實驗驗證此假說,並看 SepSAM 能否追上 dense-seg 的 0.63。

## 2. 目標與成功標準

- 產出 SepSAM craquelure 的 4-fold **YOLO-only 與 CMC** 之 P/R/F1/IoU。
- 與 dense-seg craquelure expert 在**同一份 split、同 fold** 上對照。
- **成功標準**:
  - 主問題:SepSAM(YOLO-only 或 CMC)4-fold 平均 craquelure F1 是否接近/超過 dense-seg expert(~0.63,排除 fold2)。
  - 假說檢驗:CMC F1 是否 > YOLO-only F1(SAM2 refine 對 craquelure 是否有正貢獻)— 這是本實驗最有價值的判定,無論輸贏都記錄。

## 3. 資料與對照(已核實)

- 資料: `crack_detection_sam2/data/labeled32_craq_v3/tiles_512/{images,masks}` + `group_split_stem.json`。
- **dense-seg expert 用的就是這份 tiles_root 與 split**(已確認 `expert_craq_v3_fold0_small/args.json`)→ 沿用同一份 split,fold 索引對齊,可直接比 0.63。
- **mask 格式: 512×512,值 {0,1}(binary,1=craquelure)。**→ 轉 YOLO-seg 前需以 `>0` 二值化(不可用預設 `>127`,否則標籤全空)。
- split 結構: 頂層 `folds`,每個 fold 物件含 `val_groups`(panel 名)、`train`/`val`(mask 檔名清單,`*.png`)。fold0 例:val_groups=[KJTHT-SC-M-2RB1-4],n_val_tiles=72,n_train_tiles=156(craquelure 的 panel 數/ tile 數與 crack 不同,照單全收)。

## 4. 分階段設計

### Stage 0 — 資料準備(YOLO-seg 標籤 + 各 fold data yaml)
- **遮罩二值化 + 轉 YOLO-seg 標籤**:把 masks(0/1)轉成 YOLO-seg polygon 標籤(cls 0 = craquelure)。用 `scripts/binary_mask_to_yolo_seg.py`,但該腳本預設 `>127` 門檻;因 mask 是 0/1,需先把 mask ×255 寫到暫存目錄再轉,或在轉換腳本加 `--mask_thresh 0`。標籤輸出到 `data/labeled32_craq_v3/tiles_512/labels_craq/`。
- **由 split 寫各 fold 清單**:新腳本 `scripts/build_craq_cv.py` 讀 `group_split_stem.json`,把每個 fold 的 `train`/`val`(png 檔名)對應到 image 路徑,寫 `fold{k}/train.txt`、`fold{k}/val.txt`(絕對路徑),並產 `configs/data_craq_cv_fold{k}.yaml`(`nc:1`、`names:[craquelure]`、`path/train/val`)。labels 由 ultralytics 以 images→labels 規則尋得(故 labels 目錄需與 images 同層、名為 `labels`,或在 yaml 對齊;實作時確認 ultralytics 的 image/label 對應規則)。

### Stage 1 — 訓 SEA craquelure agent(4 fold)
- 架構 `configs/yolov8n-seg-sea.yaml`(需 `sea_setup.py` 已註冊 C2f_SEA),從 COCO 預訓 yolov8n-seg 權重起訓(craquelure≠crack,不沿用 crack agent)。
- imgsz=512、batch=16、~150 epochs、aug 對齊 `finetune_heritage_cv.py`(fliplr/flipud/degrees15/scale0.1/mosaic/mixup/copy_paste/erasing、AdamW、cos_lr、single_cls)。
- 輸出 `runs/segment/runs/sepsam_agent_craq_cv_fold{k}/weights/best.pt`。複用 `finetune_heritage_cv.py` 的 CV 迴圈(或仿作 `scripts/train_craq_cv.py`),以 `data_craq_cv_fold{k}.yaml` 訓練。

### Stage 2 — 校準 CMC(craquelure 專用 config)
- 新 domain 的 SAM2 score 分布與門檻會變。用 `scripts/calibrate_sam_thresh.py` 在 craquelure val 子集掃 `SAM_THRESH` / `CONFLICTION_RATIO` / `YOLO_CONF`,輸出 `configs/cmc_craq.yaml`。
- 注意 ultralytics conf=0.0 陷阱(下限取 0.02)。
- 網狀 craquelure 的中軸(medial axis)取點會是網格狀;`POINTS_DIVISOR` 可能需調(觀察點數/分布)。
- `agent_ckpt` 指向各 fold 的 craq agent(校準與 eval 逐 fold 對應其 val)。

### Stage 3 — CMC 4-fold 評估與對照
- 每 fold 在其 val 上跑 CMC(仿 `eval_heritage_before_after.py` 的 `run_on`,全圖或依 tile 尺度;此處 tile 已是 512,單張即可不滑窗),輸出 **YOLO-only / SAM-only / CMC** 的 P/R/F1/IoU(用 `src/metrics.py` 的 `prf_iou`/`aggregate`)。
- 彙整 4-fold 平均與 per-fold,fold2 單列(且報「排除 fold2」3-fold 平均)。
- 對照表:SepSAM(YOLO-only / CMC) vs dense-seg expert(0.634/0.614/0.04/0.655)逐 fold。

## 5. 元件(檔案)
- `scripts/build_craq_cv.py`(新):split→各 fold train/val txt + data yaml;含 mask 0/1→YOLO-seg 標籤步驟(或呼叫 binary_mask_to_yolo_seg)。
- `scripts/train_craq_cv.py`(新或複用 finetune_heritage_cv.py):4-fold 訓 SEA craq agent。
- `configs/data_craq_cv_fold{0..3}.yaml`、`configs/cmc_craq.yaml`(新)。
- `scripts/eval_craq_cv.py`(新或複用 eval 模式):4-fold CMC eval + 對照表。

## 6. 風險與注意
- **fold2(彩繪面板 KJTHT-SC-L-A4-4)極可能照樣崩**(crack/craquelure dense-seg 都崩過)→ 看「排除 fold2」3-fold 平均。
- **0/1 mask 轉換陷阱**:>127 門檻會產生全空標籤,務必 >0。
- **SAM2 校準陷阱**:conf=0.0 NMS flood;SAM_THRESH 需重校(SAM2 score 分布與 v1 不同)。
- **中軸取點對網狀紋**:行為與 crack 細線不同,需觀察 SAM2 輸出是否合理(可能需要 box prompt 或負點,但本輪先用既有點 prompt 看 baseline)。
- 假說可能不成立(CMC ≤ YOLO-only):若如此,如實記錄「SAM2 對 craquelure 也無益」,並以 YOLO-only / dense-seg 作為 craquelure 專家候選。

## 7. 範圍外(YAGNI)
- 不在本輪改 CMC 的 prompt 機制(逐 instance / box / 負點)— 先測既有 SepSAM pipeline 的 baseline;若 CMC 有潛力再優化 prompt。
- 不動 crack 專家(ResUNet)— 另案。
- 不做 dense-seg expert 的再訓練 — 它只是對照基準。
