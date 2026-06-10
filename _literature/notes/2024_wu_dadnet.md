---
title: "DADNet: dual-attention detection network for crack segmentation on tomb murals"
authors: Meng Wu, Ruochang Chai, Yongqin Zhang, Zhiyong Lu
year: 2024
venue: Heritage Science 12:369
link: https://doi.org/10.1186/s40494-024-01474-0
pdf: papers/DADNet.pdf
tags: [heritage-crack, craquelure, crack-seg, unet, convnext, attention, neighborhood-attention, axial-attention]
read_date: 2026-06-10
---

# DADNet: dual-attention detection network for crack segmentation on tomb murals

## TL;DR（一句話結論）
ConvNeXt-T backbone 的 U-Net,在 skip connection 插入雙注意力(Neighborhood Attention 抓局部細節 + Biaxial/axial Attention 抓全域語意),在自建 1000 張 224x224 唐墓壁畫 TTMural 上做 crack+craquelure 二值分割,贏過 U-Net / DeepLabV3+ / DCSAU-Net / Swin-Unet / UCTransNet。

## 問題 (Problem)
墓室壁畫先勾線再上色,細密的 sketch 線條與細裂(crack ending)/龜裂(craquelure)混在複雜壁畫背景裡難以區分;且壁畫資料量極少,易過擬合。要在小樣本下精準分割 crack/craquelure。

## 方法 (Method)
- **Backbone**: ConvNeXt-T(實驗中 T 版最佳;小樣本下深/寬的版本反而過擬合)。U-Net encoder-decoder 結構。
- **Dual Attention 插在 skip connection**:
  - **Neighborhood Attention (NA)**: 每個 pixel 只對 k×k 鄰域做 local self-attention,計算量 ~ k²·H·W(遠低於 full self-attention)。**k=7 最佳**(太小細節不足,太大引入無關資訊)。
  - **Biaxial Attention Block (BA-B)**: axial attention(分別沿 row / col 做注意力再相加回原圖 `BA(X)=X+AA_row(X)+AA_col(X)`),外層仿 RFB(Receptive Field Block)用多分支不同 dilation 的卷積擴大感受野。**dilation rate=7 最佳**。NA 補局部、BA-B 補全域多尺度,互補。
- **輸入解析度 224×224**;loss = cross-entropy;Adam(β=0.9),batch 16,lr 1e-4,RTX 3090,PyTorch。
- **資料 TTMural**: 壁畫切成 21790 個 224×224 chunk,剔除無 crack/craquelure 的塊後隨機取 1000 張,LabelMe 標 crack+craquelure,80/10/10 split。**無公開程式碼/資料(需向作者索取)**。

## 關鍵結果 (Key Results)
- TTMural 上 DADNet MIoU/F1 為各 baseline 最高(U-Net、DeepLabV3+、DCSAU-Net、Swin-Unet、UCTransNet);論文以表格與視覺圖佐證,小龜裂與 crack ending 區域明顯較少 under-segmentation。
- 複雜度:Params 與 FLOPs 低於純 transformer 的 Swin-Unet/UCTransNet;FPS 27.97(U-Net 140.56 最快但精度低)。
- Ablation:baseline → +ConvNeXt → +BA-B → +NA 逐步提升,三者累加最佳。
- (註:論文正文未列出精確 MIoU 數字表格內容於本次抽取的文字中,表 2/5 數值在圖表內;結論為 DADNet 全面領先。)

## 限制與假設 (Limitations)
- 單一壁畫(章懷太子墓 polo mural)來源,domain 窄;作者自承小樣本易過擬合,future work 要擴資料。
- 把 crack 與 craquelure 合併成單一前景類(二值),未分兩類細分。
- 無公開 code/權重,需自行復現;NA 需 `natten` 類 CUDA kernel,環境相依性高。
- 輸入固定 224×224 chunk。

## 與我研究的關聯 (Relevance to my work)
直接對應古蹟劣化 craquelure 分割。可當本專案 craquelure 的另一條 baseline(對照現有 dense-seg SAM2 / ResUNet / SepSAM2)。ConvNeXt-T + NA(k=7)+ BA-B(dilation 7)的雙注意力 skip 設計,正好補我們資料上「細龜裂被 under-segment」的痛點。**使用者原以為切 24×24,實為 224×224**,小 crop 對細結構有利的直覺方向對、數字記錯。

## 可行動結論 (Actionable conclusion)
- [ ] 新建獨立 venv(ConvNeXt 用 timm、NA 用 natten,與 sam2_env/unet_env 隔離)復現 DADNet。
- [ ] 用本專案 craquelure 二值資料(0-94test + batch_1)訓練,當 craquelure expert 的對照組;先用 224×224 chunk(可另試 512 tile 與專案其他模型對齊)。
- [ ] 關鍵超參:NA k=7、BA-B dilation 7、ConvNeXt-T、Adam lr 1e-4 batch 16 CE loss。
- [ ] natten 安裝若與現有 torch 版本衝突,退路是用純 PyTorch 實作 unfold 版 neighborhood attention(慢但免 CUDA kernel)。
- [ ] 關聯 [[2026_cuch-guillen_synthetic-craquelure]](合成 craquelure 可補小樣本)。
