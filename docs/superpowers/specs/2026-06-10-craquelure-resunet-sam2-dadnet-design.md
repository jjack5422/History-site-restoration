# Craquelure 雙模型實驗設計 (ResUNet→SAM2 prompt 精修 + DADNet 復現)

- 日期: 2026-06-10
- 狀態: design (待轉 plan)
- 分支: feature/cvat-craq-crack-agent

## Context (為何做這個)

目前古蹟劣化分割聚焦 **craquelure**(crack 暫放)。要比較兩條 craquelure 分割路線:

- **A**: 用 ResUNet 當輕量「粗 expert」產生 craquelure mask,再用 **SAM2** 以該 mask(dense mask prompt)或自其取樣的點(point prompt)做精修。動機:單獨 ResUNet 在本資料上 craquelure/crack IoU 偏低(crack expert 最佳僅 IoU ~0.17),希望凍結 image encoder 的 SAM2 mask decoder 能把粗 mask 精修成更貼合的邊界,且**推論全自動**(prompt 由 ResUNet 產,不需人工點)。
- **B**: 復現 **DADNet**(Wu et al., Heritage Science 2024;筆記 `_literature/notes/2024_wu_dadnet.md`),作為獨立 craquelure baseline。

兩條共用同一份 craquelure 標註資料與單一 holdout 切分,最終可直接互比 IoU/F1。

預期產出:一張 craquelure 對照表(ResUNet / SAM2+mask / SAM2+points / DADNet)+ 各自權重與 run 記錄。

## 範圍與非目標

- 範圍: 只做 craquelure 二值分割。
- 非目標: crack 與其餘劣化類(本輪不做);多類聯訓;ResUNet 與 SAM2 端對端聯合訓練(採兩階段)。

## 資料 (共用 Stage 0)

- 標註來源: `_data/0-94test/SegmentationClass`(CVAT VOC palette PNG, 1024×1024)。
- 原圖: `_data/selected_slices/batch_1`(1024×1024 .jpg)。
- 只用**前 95 張已完成標註**的影像(其餘 stem 標註未完成)。craquelure 色 = `(102,255,102)`。
- palette→binary 重用 `crack_detection_sam2/scripts/build_binary_datasets.py` 的 `palette_mask_to_binary(rgb, target_rgb)` 與 `PALETTE` 字典。
- **切分**: 單一 holdout,依「來源影像 stem」分 ~80/20(group split,同圖 tile 不跨 train/val,避免洩漏)。固定 random seed,切分清單存檔。
- 兩種 tile 規格:
  - 給 A: **512×512 tile, stride 256**(專案標準),輸出 `images/ masks/ tile_index.json`,相容 `_lib/crackseg_common/dataset.py` 的 `TileSegDataset` / `load_tile_index`。
  - 給 B: **224×224 chunk**(依 DADNet 原論文;**非先前誤記的 24×24**)。
- 產物資料夾(暫定): `_data/craq_0-94_v1/tiles_512/` 與 `_data/craq_0-94_v1/chunks_224/`,各含 `images/ masks/ tile_index.json` 與 holdout 切分 json。

## 子專案 A: ResUNet craquelure expert → SAM2 prompt 精修

兩階段、離線傳遞 prompt(繞開 `unet_env` / `sam2_env` 套件互斥)。

### A1 — 訓練 ResUNet craquelure expert
- 重用 `crack_detection_unet/src/train.py` + `src/unet_model.py::build_resunet(encoder="resnet50", num_classes=2)`。venv: `unet_env`(torch 2.12 / smp 0.5)。
- Dataset: `TileSegDataset`(`_lib/crackseg_common/dataset.py`);augment: `_lib/crackseg_common/augment.py`。loss: BCE+Dice(沿用)。
- 輸出: `crack_detection_unet/runs/2026-06-10-craq-resunet50/{best.pt,last.pt,log.json}`。此即「小 craquelure expert」交付物。

### A2 — 離線產生 prompt(ResUNet craquelure 機率圖)
- 用 `crack_detection_unet/src/predict_full.py --save_prob` 對 A 的**每個 512 tile** 跑推論,存 craquelure 機率圖 `.npy`(取前景通道, float32, 512×512)。
- 存到 `_data/craq_0-94_v1/tiles_512/resunet_prob/{tile}.npy`,與 GT tile 對齊。train/val 皆產(SAM2 訓練/驗證都需要對應 prompt)。

### A3 — SAM2 prompt 精修(兩 prompt 模式)
- 模型: `crack_detection_sam2/model_prompted_sam2.py::PromptedSAM2Seg(variant="small", image_size=512)`。image encoder 凍結;mask decoder + prompt encoder 可訓(預設)。venv: `sam2_env`。
- 新訓練腳本 `crack_detection_sam2/train_craq_promptrefine.py`,旗標 `--prompt_mode {mask,points}`。Dataset 回傳 (image tile, craquelure GT, ResUNet prob)。
- **mask 模式**: ResUNet prob → resize 到 128×128 → 當 `forward(..., prev_mask=)`(接到 SAM2 `sam_prompt_encoder(masks=)`)。points 給「無點/padding」。
- **points 模式**: 從 ResUNet binary(prob>0.5)取 K 個高信心正點 + 從預測背景取 K 個負點(座標在 0..512),當 `point_coords/point_labels`;`prev_mask=None`。推論時同樣以 ResUNet 取點(全自動)。
- loss/metric/optim 重用 `train_prompt.py` 既有實作模式: `BinaryCEDiceLoss`、`evaluate()`(回 IoU/F1/P/R)、`param_groups(base_lr, encoder_lr_mult)`、cosine+warmup、AMP、grad clip。
- 輸出: `crack_detection_sam2/runs/2026-06-10-craq-sam2prompt-mask/` 與 `...-points/`,各含 `best.pt/last.pt/log.json/metrics.json`。

### A 邊界處理
- PromptedSAM2Seg 在 points 模式需可接受「空/padding 點」(mask 模式時)。實作時於 `decode()` 對 `point_coords` 傳單一 padding 點(label = -1)或確認 prompt_encoder 接受 N=0;plan 階段先寫測試驗證。
- ResUNet prob 當 mask prompt 時以 logit 化(或直接 prob)餵入,plan 階段比較哪種數值範圍較穩。

## 子專案 B: DADNet 復現

- **新 venv `dadnet_env`**(與 sam2/unet 隔離)。deps: `torch`(對齊可用 CUDA)、`timm`(ConvNeXt-T)、`natten`(Neighborhood Attention)、`albumentations`、`numpy`、`pillow`。
- **架構**(依論文): ConvNeXt-T(`timm` features_only)當 encoder → U-Net decoder;每條 skip connection 經 **Dual Attention** =
  - **Neighborhood Attention (NA)**, 鄰域 **k=7**;
  - **Biaxial Attention Block (BA-B)** = axial(row + col)attention `BA(X)=X+AA_row(X)+AA_col(X)`,外層仿 RFB 多分支不同 dilation,**rate 7**。
  - 從零實作(論文無公開 code)。
- **資料/超參**(依論文): 224×224 chunk、Adam(β=0.9)、lr 1e-4、batch 16、**CE loss**。單一 holdout 80/20(與 A 一致;論文為 80/10/10)。可選加測 Dice 以對齊 A。
- **風險/退路**: `natten` CUDA kernel 與安裝 torch 版本可能不合 → 退路:純 PyTorch `unfold` 版 neighborhood attention(較慢、免 kernel)。建環境時先跑最小 smoke 確認可用,失敗即切退路。
- 輸出: `crack_detection_dadnet/runs/2026-06-10-craq-dadnet/{best.pt,log.json,metrics.json}`。

## 評估與交付

- 統一指標: craquelure val **IoU / F1 / Precision / Recall**(像素級, 前景=craquelure)。同一 holdout val 集。
- 對照表(README 或 EXPERIMENTS.md 一節): ResUNet 單獨 / SAM2+mask / SAM2+points / DADNet。
- 每個 run 一資料夾,含: 完整指令(manifest)、config(args)、`metrics.json`、`log.json`。遵循 experiment-tracking 慣例。

## 重用清單(已查證)

- 資料/切分/augment: `_lib/crackseg_common/{dataset.py,augment.py,data_utils.py}`;palette→binary `crack_detection_sam2/scripts/build_binary_datasets.py`。
- ResUNet: `crack_detection_unet/src/{train.py,predict_full.py,unet_model.py}`(venv `unet_env`)。
- SAM2 prompt: `crack_detection_sam2/model_prompted_sam2.py`(forward 支援 `prev_mask` 與 point prompt);訓練模式參考 `crack_detection_sam2/train_prompt.py`(`BinaryCEDiceLoss`/`evaluate`/`param_groups`)。
- DADNet 參考: `_literature/notes/2024_wu_dadnet.md`。

## 開放問題(plan 前確認或實作中決定)

- A3 mask 模式餵 prob vs logit 的數值穩定性(實作中比較)。
- B 是否同時加跑 512 tile 版以與 A 完全對齊(預設先只跑 224)。
