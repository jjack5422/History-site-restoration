# Decision: craquelure 分割 4-model 比較(DeepLabV3+ / SegFormer vs ResUNet / SAM2-refine)

- 日期: 2026-06-29
- 分支: `exp/crack-craq-dualexpert-refine`
- spec: `crack_detection_sam2/docs/superpowers/specs/2026-06-29-craquelure-seg-model-comparison-design.md`
- plan: `_research/plans/2026-06-29-craquelure-seg-model-comparison.md`
- 比較表: `crack_detection_sam2/runs/craq-modelcompare-fold0-2026-06-29/comparison.{md,json}`

## 協定
fold0 honest val(train folds1-4 / eval fold0),資料 `_data/multiclass_512_craq_bin`(1027 tile,由 canonical
`multiclass_512_dataset` craquelure 通道 target_id=4 派生)。三個本次訓練的模型**同 split、同 loss(CEDiceLoss
ce0.5+dice0.5)、同 aug、同 optimizer(AdamW lr3e-4/enc_lr_mult0.1)、80ep、seed42**,只變 architecture。
fold0 val_groups = KJTHT-SC-L-A4-4 / KJTHT-SC-M-2LB1-2 / KJTHT-SC-R-A4-3(素面)+ MGLST-DT-1L-A2-1(彩繪)。

## 結果(craquelure 前景類,fold0 val,thr 0.5;best.pt 以 craq IoU 選)

| model | params | stage | precision | recall | IoU | F1 |
|---|---|---|---|---|---|---|
| **ResUNet** (smp.Unet resnet50) | 32.5M | 1 | **0.6107** | 0.8084 | **0.5335** | **0.6958** |
| DeepLabV3+ (smp resnet50) | 26.7M | 1 | 0.4902 | 0.8261 | 0.4443 | 0.6153 |
| SegFormer (smp mit_b0) | 3.7M | 1 | 0.4400 | 0.8442 | 0.4070 | 0.5785 |
| SAM2-refine *(引用,非同條件)* | — | 2 | 0.6811 | 0.8309 | 0.5982 | 0.7486 |

SAM2-refine = `craq-refine-tversky28-aug-0-94gt-2026-06-22` fold0 @ep17,**資料版 craq_0-94_v1(非本表 craq_bin1027)、
all-data ResUNet prob 當 prompt 有洩漏偏樂觀、split 不同** → 僅供參考,不可當同條件直接比。

## 結論(對照預期:符合)

1. **主流通用分割模型在本 craquelure 任務上未勝過專案 ResUNet anchor。** 同 split 三模型 IoU 排序
   **ResUNet 0.534 ≫ DeepLabV3+ 0.444 > SegFormer 0.407**;DeepLabV3+ 落後 ResUNet **−0.089 IoU**、
   SegFormer **−0.126 IoU**。
2. **差距主要在 precision,不在 recall。** 三者 recall 都高(0.81–0.84,median_freq 類權重把模型推向高 recall),
   但 ResUNet precision 0.611 遠高於 DeepLabV3+ 0.490 / SegFormer 0.440 → 兩個主流模型**過度預測、content-FP 更多**
   (與 [[finding_craq_refiner_content_fp]] 同源:craquelure 在彩繪/裝飾紋上易 FP,呼應 fold0 val 含 MGLST 彩繪)。
3. **SegFormer-b0 最弱但容量最小(3.7M vs ~27–32M)**,容量差異是重要 confound;尚未測 mit_b2/b5,不能據此斷定
   transformer 架構本身較差。
4. **SAM2-refine 引用值(0.598)仍是檯面最高**,但因資料版/洩漏 caveat 不能與本表並列下強結論;若要嚴格,
   需在 craq_bin1027 同 split 重跑 refine(後續)。

## How to apply / 下一步
- **craquelure 單階段首選仍是 smp.Unet resnet50(ResUNet)**;DeepLabV3+/SegFormer 在此資料無優勢,主因 precision。
- 若要替主流模型翻盤:(a) SegFormer 升 mit_b2/b5(補容量);(b) 對齊 SAM2-refine 在 craq_bin1027 同 split 重跑做真正同條件比;
  (c) 針對 content-FP 加 hard-neg(彩繪負樣本)再比 precision。
- crack 類 + dual-expert refine ceiling 仍為延後階段(spec `2026-06-29-crack-craq-dualexpert-refine-design.md`)。

相關:[[project_craquelure_dual_model]] [[finding_craq_refiner_content_fp]] [[project_multiclass_512_dataset_711]]
