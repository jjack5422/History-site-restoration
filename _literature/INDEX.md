# Literature Index

> 文獻庫總表。每讀一篇就補一列（最新在上）。筆記在 `notes/`，PDF 在 `papers/`。
> 由 `literature-review` skill 維護。

| 題目 | 年 | 標籤 | 一句話結論 | 筆記 |
|------|----|------|------------|------|
| DADNet: dual-attention detection network for crack segmentation on tomb murals | 2024 | heritage-crack, craquelure, unet, convnext, attention | ConvNeXt-T U-Net + skip 插 Neighborhood(k=7)與 Biaxial/axial(dilation 7)雙注意力,224x224 壁畫 crack/craquelure 二值分割勝 Swin-Unet/UCTransNet 等 | [note](notes/2024_wu_dadnet.md) |
| Synthetic Craquelure Generation for Unsupervised Painting Restoration | 2026 | craquelure, synthetic-data, segformer, lora | 程序式 Bézier+tapered+branching 合成 craquelure 三元組,免標註訓練裂縫偵測,勝過 SOTA | [note](notes/2026_cuch-guillen_synthetic-craquelure.md) |
| Unpaired Image-to-Image Translation (CycleGAN) | 2017 | gan, image-translation, unpaired, cycle-consistency | 非配對域互轉,cycle-consistency loss;可做合成→真實的 sim-to-real 真實感增強 | [note](notes/2017_zhu_cyclegan.md) |
| SMG-Net: fine-grained crack segmentation in ancient wooden structures | 2025 | wood-heritage, crack-seg, unet, lightweight | 木構裂縫域資料極稀少(256 原圖),SMG-Net 木構 mIoU 81.12 勝 Swin-Unet/SegFormer | 見 [topic](topics/heritage-crack-craquelure-synthesis.md) |
