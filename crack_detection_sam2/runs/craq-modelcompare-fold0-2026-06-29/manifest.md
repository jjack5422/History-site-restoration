# craq-modelcompare-fold0-2026-06-29

目的: 彙整 craquelure fold0 honest val 四模型比較。
來源 runs:
  crack_detection_unet/runs/unet-craqbin1027-fold0-2026-06-29 (ResUNet anchor)
  crack_detection_unet/runs/deeplabv3plus-craqbin1027-fold0-2026-06-29
  crack_detection_unet/runs/segformer-craqbin1027-fold0-2026-06-29
  SAM2-refine 引用 crack_detection_sam2/runs/craq-refine-tversky28-aug-0-94gt-2026-06-22 (資料版不同, caveat)
重跑指令:
  segbaseline_env/bin/python crack_detection_sam2/scripts/collect_craq_compare.py \
    --run "...=...unet-craqbin1027-fold0-2026-06-29" \
    --run "...=...deeplabv3plus-craqbin1027-fold0-2026-06-29" \
    --run "...=...segformer-craqbin1027-fold0-2026-06-29" \
    --cite "SAM2-refine(2-stage, CITED)=0.5982,0.6811,0.8309,0.7486,<caveat>" \
    --out crack_detection_sam2/runs/craq-modelcompare-fold0-2026-06-29
結論: _research/decisions/2026-06-29-craquelure-model-comparison.md
  ResUNet 0.534 IoU ≫ DeepLabV3+ 0.444 > SegFormer 0.407; 差距在 precision (主流模型 content-FP 多)。
