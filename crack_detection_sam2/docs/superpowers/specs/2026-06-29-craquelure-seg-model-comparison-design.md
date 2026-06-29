# Design: craquelure 分割 4-model baseline 比較(DeepLabV3+ / SegFormer vs ResUNet / SAM2-refine)

- 日期: 2026-06-29
- 分支: `exp/crack-craq-dualexpert-refine`
- 狀態: design 已與使用者敲定,待寫實作計畫
- 關係: 取代本階段焦點;先前 `2026-06-29-crack-craq-dualexpert-refine-design.md`(crack+craq dual-expert
  in-dist ceiling)**延後為後續階段**,本實驗先只做 craquelure 的主流模型比較。

## 目標 / 研究問題

建新 venv,加入兩個主流語意分割模型 **DeepLabV3+** 與 **SegFormer**,與既有 **ResUNet**、
**SAM2-refine** 在 **craquelure 二值分割** 上做公平比較:

> 在本專案壁畫 craquelure 資料上,主流通用分割架構(DeepLabV3+/SegFormer)相較專案既有方案
> (ResUNet 單階段 / SAM2-refine 兩階段)的 **precision / recall / IoU / F1** 表現如何?

非目標(YAGNI / 後續): crack 類、dual-expert、refine 接 router、泛化 held-out。

## 評估協定(已敲定,公平性核心)

- **fold0 honest val**: 四個模型全部 train folds1-4 / eval fold0,**同一份 split、同一份資料**。
  (使用者原話「全資料訓練」在此具體化為「全 fold 訓練、fold0 驗證」以取得可比的誠實數字。)
- 資料: 由現行 `_data/multiclass_512_dataset`(1027 tile,canonical 0-94 GT,craquelure 色 102,255,102)
  **重建 `_data/multiclass_512_craq_bin` 到 1027**(現為舊 917),用其 `group_split_stem.json` fold0。
- 參照處理: **ResUNet 在此重建 craq_bin 同 split 重跑當 anchor**(與新模型同資料);
  **SAM2-refine 引用既有 fold0 數字並明確註記資料版/ split 差異**(不重跑)。

## 架構 / 元件(最小改動,共用既有訓練棧)

四模型皆 smp,共用 `crack_detection_unet/src/train.py` + `crackseg_common.CEDiceLoss` + 同 dataset/aug:

| 模型 | smp 建構 | encoder | 階段 | 本實驗動作 |
|---|---|---|---|---|
| ResUNet (anchor) | `smp.Unet` | resnet50 | 1-stage | 重跑 fold0 |
| DeepLabV3+ | `smp.DeepLabV3Plus` | resnet50 | 1-stage | 新跑 fold0 |
| SegFormer | `smp.Segformer` | mit_b0(8GB 起步,記憶體許可升 b2) | 1-stage | 新跑 fold0 |
| SAM2-refine | (既有兩階段) | hiera+ResUNet prob prompt | 2-stage | 引用既有數字 |

**程式改動**: 把 `crack_detection_unet/src/unet_model.py` 的 `build_resunet` 一般化為
`build_model(arch, encoder, ...)`(支援 unet/deeplabv3plus/segformer,三者皆 smp、forward 同為 logits NCHW、
backbone 皆 `encoder.` 前綴故 `param_groups` 不變);`train.py` 加 `--arch`(預設 unet 保舊行為)。
encoder 預設: deeplabv3plus=resnet50(與 ResUNet 對齊隔離架構變因)、segformer=mit_b0(其自帶 MiT encoder)。

## 環境

- **新 venv `segbaseline_env`**,鏡像 unet_env 可用堆疊: torch cu128(sm_120 必須)+ smp 0.5.0 +
  `pip install -e _lib`(crackseg_common)。理由: smp 已涵蓋三架構,複用同棧確保只有 architecture 是變因。
- 8GB GPU guards: batch size 視記憶體(預估 unet/deeplab bs4-8、segformer-b0 bs4);
  `NO_ALBUMENTATIONS_UPDATE=1` + `HF_HUB_OFFLINE=1`;GPU 工作序列化勿並行重 IO(避 cudaErrorUnknown)。

## 訓練設定(三模型對齊,只變架構)

- loss: `CEDiceLoss`(ce+dice,沿用 ResUNet 既有預設權重),class_weight_mode 同 ResUNet 既有設定。
- aug: 沿用 `train.py` 既有 train_transforms(含 CLAHE/HSV/翻轉)。
- optimizer: AdamW,encoder_lr_mult 0.1,base_lr/epoch 與 ResUNet 既有 fold0 run 對齊。
- image_size 512,num_classes 2,encoder_weights imagenet。
- 每模型 run: `crack_detection_unet/runs/<arch>-craqbin1027-fold0-2026-06-29/`,含 manifest/train.log/metrics.json/best.pt。

## 評估 / 成功定義

- 指標: craquelure(前景類)**precision / recall / IoU / F1**,micro pixel-pooled,fold0 val,thr 0.5
  (與 ResUNet 既有計量一致;refine 引用值用其原 thr 並註記)。
- 產出比較表(`crack_detection_sam2/runs/craq-modelcompare-fold0-2026-06-29/comparison.md` + .json):
  四列 ResUNet/DeepLabV3+/SegFormer/SAM2-refine × {P,R,IoU,F1,params,stage,資料版註}。
- 成功 = 得到四模型在 craquelure 同 split 的誠實對照數字,並能明確下結論:主流模型(DeepLabV3+/SegFormer)
  是否勝過/持平/輸給 ResUNet 與 SAM2-refine,以及各自 P/R 取捨型態(refine 歷史是高 recall 修剪器)。

## 產出位置(RULES)

- expert runs: `crack_detection_unet/runs/`;比較彙整 + 結論表: `crack_detection_sam2/runs/craq-modelcompare-fold0-2026-06-29/`。
- 計畫: `_research/plans/2026-06-29-craquelure-seg-model-comparison.md`(含檢查點)。
- 結論: `_research/decisions/2026-06-29-craquelure-model-comparison.md`。
- EXPERIMENTS.md 各補列。

## 風險 / 已知陷阱

- SegFormer-b0 與 resnet50-based 模型參數量差很多 → 表內列 params,比較時點出容量差異,別只看單一數字。
- SAM2-refine 引用值資料版不同 → 表內明確標 caveat,不可當同條件直接比;若使用者要嚴格,後續再同 split 重跑。
- 8GB OOM → 先 bs 小、必要時 grad accumulation;segformer 若 b0 仍緊則降 image_size 為 ablation 並註記。
- fold0 val 群含哪些 stem 決定難度(KJTHT 素面為主)→ 報告附 fold0 stem 清單。
