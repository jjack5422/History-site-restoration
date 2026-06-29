# craquelure fold0 model comparison

| model | precision | recall | IoU | F1 | run | note |
|---|---|---|---|---|---|---|
| ResUNet(smp.Unet resnet50, 1-stage, 32.5M) | 0.6107 | 0.8084 | 0.5335 | 0.6958 | crack_detection_unet/runs/unet-craqbin1027-fold0-2026-06-29 | fold0 honest val, best ep48 |
| DeepLabV3+(smp resnet50, 1-stage, 26.7M) | 0.4902 | 0.8261 | 0.4443 | 0.6153 | crack_detection_unet/runs/deeplabv3plus-craqbin1027-fold0-2026-06-29 | fold0 honest val, best ep49 |
| SegFormer(smp mit_b0, 1-stage, 3.7M) | 0.4400 | 0.8442 | 0.4070 | 0.5785 | crack_detection_unet/runs/segformer-craqbin1027-fold0-2026-06-29 | fold0 honest val, best ep68 |
| SAM2-refine(2-stage, CITED) | 0.6811 | 0.8309 | 0.5982 | 0.7486 | (cited) | CITED craq_0-94_v1 0-94GT fold0 @ep17; all-data prompt prob 洩漏偏樂觀; 資料版/split 與本表 craq_bin1027 不同, 非同條件直接比 |
