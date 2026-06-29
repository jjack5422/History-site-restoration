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

## 結果(craquelure 前景類,fold0 val,thr 0.5;**四模型皆同條件**:同 craq_bin1027 / 同 fold0 / leak-free)

| model | params | stage | precision | recall | IoU | F1 |
|---|---|---|---|---|---|---|
| **SAM2-refine** (mask-prompt tversky28-aug) | SAM2 small + ResUNet prompt | 2 | **0.6736** | 0.7499 | **0.5500** | **0.7097** |
| ResUNet (smp.Unet resnet50) | 32.5M | 1 | 0.6107 | **0.8084** | 0.5335 | 0.6958 |
| DeepLabV3+ (smp resnet50) | 26.7M | 1 | 0.4902 | 0.8261 | 0.4443 | 0.6153 |
| SegFormer (smp mit_b0) | 3.7M | 1 | 0.4400 | **0.8442** | 0.4070 | 0.5785 |

SAM2-refine 改為**同條件重跑** `craq-refine-tversky28-aug-craqbin1027-fold0-2026-06-29` @ep28:prompt = **leak-free
fold0 ResUNet**(`unet-craqbin1027-fold0` best.pt 對 craq_bin 全 1027 tile dump → `resunet_prob_fold0`,fold0 val
prob 未洩漏),eval `logits>0`(sigmoid 0.5)與其他三模型 argmax(0.5)同門檻。**舊引用 0.598(0-94 資料 + all-data
prob 洩漏)已汰**;誠實同條件值 **0.550 < 0.598**,證實舊值偏樂觀。

## 結論(對照預期:符合)

1. **SAM2-refine 同條件下 IoU 最高(0.550),小幅勝 ResUNet(+0.016)**,但靠 **precision**(0.674 vs 0.611)、
   **recall 反而較低**(0.750 vs 0.808)→ refine 是「修剪器」,壓 content-FP 換來淨 IoU 微升,代價是少抓一些。
   增益不大(+0.016),非數量級差距。
2. **主流通用分割模型(DeepLabV3+/SegFormer)明顯墊底,未勝任一專案方案。** IoU 排序
   **SAM2-refine 0.550 > ResUNet 0.534 ≫ DeepLabV3+ 0.444 > SegFormer 0.407**;DeepLabV3+ 落後 ResUNet
   **−0.089**、SegFormer **−0.126** IoU。
3. **差距主要在 precision,不在 recall。** 四者 recall 都不低(0.75–0.84),但 precision:refine 0.674 / ResUNet
   0.611 ≫ DeepLabV3+ 0.490 / SegFormer 0.440 → 兩個主流模型**過度預測、content-FP 更多**(與
   [[finding_craq_refiner_content_fp]] 同源;呼應 fold0 val 含 MGLST 彩繪)。
4. **SegFormer-b0 最弱但容量最小(3.7M vs ~27–32M)**,容量差異是重要 confound;尚未測 mit_b2/b5,不能據此斷定
   transformer 架構本身較差。

## How to apply / 下一步
- **craquelure 首選 = SAM2-refine(mask-prompt tversky28-aug)**,同條件下 IoU 最高(0.550),但只比 ResUNet 微升
  +0.016、且犧牲 recall;若重視高 recall 召回,**單階段 ResUNet(0.808 recall)更合適**。兩者都遠勝 DeepLabV3+/SegFormer。
- **DeepLabV3+/SegFormer 在此資料無優勢**,主因 precision(content-FP)。要替主流模型翻盤:(a) SegFormer 升 mit_b2/b5
  補容量;(b) 針對 content-FP 加 hard-neg(彩繪負樣本)再比 precision。
- crack 類 + dual-expert refine ceiling 仍為延後階段(spec `2026-06-29-crack-craq-dualexpert-refine-design.md`)。

相關:[[project_craquelure_dual_model]] [[finding_craq_refiner_content_fp]] [[project_multiclass_512_dataset_711]]
