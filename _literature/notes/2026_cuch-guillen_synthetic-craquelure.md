---
title: Synthetic Craquelure Generation for Unsupervised Painting Restoration
authors: Jana Cuch-Guillén, Antonio Agudo, Raül Pérez-Gonzalo
year: 2026
venue: arXiv preprint (cs.CV)
link: https://arxiv.org/abs/2602.12742
pdf: papers/2026_cuch-guillen_synthetic-craquelure.pdf
tags: [craquelure, crack-detection, synthetic-data, painting-restoration, segformer, lora, procedural-generation]
read_date: 2026-06-02
---

# Synthetic Craquelure Generation for Unsupervised Painting Restoration

## TL;DR(一句話結論)
用**程序式合成 craquelure**(Bézier 曲線 + tapered/branching 幾何)在乾淨 WikiArt 繪畫上造出對齊三元組 (I, M, Ĩ),
訓練一個 SegFormer+LoRA 精修器來分離真裂縫與筆觸,達成**完全免人工標註**的裂縫偵測與虛擬修復,零樣本下勝過 SOTA 照片修復模型。

## 問題 (Problem)
舊畫的 craquelure(龜裂)極細、不規則、低對比,難與筆觸/老化亮漆區分。
深度法受限於**像素級標註稀少**與**極端類別失衡**(裂縫像素極少);transformer 還會 oversmooth 細結構。
需要一個不靠人工標註的 domain-specific 策略。

## 方法 (Method)
三階段 pipeline:
1. **古典偵測**:top-hat 形態學 + size-based 去噪 → 過度涵蓋的候選裂縫 mask(over-inclusive)。
2. **學習式精修**:SegFormer MiT-B0 + **LoRA**(r=8, α=16, dropout 0.1;只訓 LoRA + seg head)。
   - **Detector-guided 4-channel 輸入**:RGB + 二值偵測 mask 當空間先驗。
   - **Masked hybrid loss**:權重 w = m + α(1−m), α=0.01,聚焦偵測啟動區;`L = L_CE(p̃,y;w) + λ L_Dice(p,y)`, λ=2。
   - **Detector-guided logit adjustment**:`p̃ = p + γm`, γ=1,軟性偏置往偵測處預測。
   - 只用合成資料訓練;AdamW lr=2e-4, batch=8。
3. **修復 inpainting**:Modified Trimmed Mean (MTM, 由外向內填) 與 Anisotropic Diffusion (AD, λ=0.25, K=127, 20 steps),僅作用於裂縫像素。

**合成資料生成(本專案最關鍵的部分)**:
- 對齊三元組 (clean I, binary mask M, damaged Ĩ),底圖用 WikiArt。
- 裂縫軌跡 = cubic **Bézier 曲線**;端點均勻取樣,內控制點加高斯擾動 (σ_p=8px) 產生彎曲。曲線取樣 80–180 點。
- **Tapered 幾何**:沿曲線畫實心圓盤,半徑 r(t) ~ N(α(1−|t−0.5|), σ_r²),α=2.0px、σ_r=0.5px → 兩端細、中段粗。
- 每張圖 80–150 條曲線;以機率 p_br∈[0.3,0.5] 旋轉+縮放局部方向向量**分支 branching**。
- 後處理:2×2 形態學侵蝕調厚度 + 5×5 高斯模糊 (σ=2) 模擬顏料滲色;閾值 50 二值化。
- damaged Ĩ:把 M 內像素換成裂縫專屬灰值,其餘不動(模擬掉漆/暗裂)。最後縮放(如 598×375)。

## 關鍵結果 (Key Results)
測試集為 4 張人工標註的真實裂縫畫作。Detection 用 Acc/F1,Restoration 用 SSIM。
- 提出的 Learning-based Segmentation + AD vs 傳統 Grassfire baseline [4]:
  - Detection MEAN:Acc 85.19 / F1 61.36 vs baseline(Grassfire+AD)Acc 74.11 / F1 51.36。
  - Restoration MEAN SSIM:64.87 vs 56.03。
- 即:精修器在四張畫平均把 detection F1 由 ~51 提到 ~61,SSIM 由 56 提到 65。
- Ablation(Table III)顯示 Guided Logit / LoRA / Mask Loss 三組件皆有貢獻(含 IoU/Dice/MCC/PSNR/LPIPS/VIF)。

## 限制與假設 (Limitations)
- 測試集僅 **4 張**真實畫作,規模極小,泛化結論偏弱。
- Domain 是**繪畫**(WikiArt 風格 + 老化亮漆),非木質文物;沒有木紋/雕刻紋理這類干擾。
- 合成劣化模型簡化:裂縫=灰值填充,未建模真實掉漆的色彩/深度變化、髒汙、生物劣化。
- 仰賴古典 top-hat 偵測當先驗,低對比/複雜光照下先驗品質會限制上限。

## 與我研究的關聯 (Relevance to my work)
**直接對應** [[data-generation-project]] 的核心:程序式合成 craquelure 來解決標註稀少 + 類別失衡。
- 可直接借用的合成原語:Bézier 軌跡、tapered 半徑剖面、branching、erosion+blur 後處理、(I,M,Ĩ) 三元組格式。
- 與下游 [[crack-detection-projects]] 對齊:detector-guided 4-channel 輸入、masked hybrid loss、logit adjustment 都是對付 crack 極端失衡的具體 trick,可移植到我們的偵測器。
- **不適用/需改**:木質文物有木紋、雕刻、顏料層、髒汙,要把背景域從繪畫換成木材,並把「裂縫=灰值」換成更貼近木材劣化(順紋裂、節點裂)的外觀模型。

## 可行動結論 (Actionable conclusion)
- [ ] 復現其 Bézier+tapered+branching 合成器當 data_generation 的 v0 baseline 合成原語。
- [ ] 把 (I, M, Ĩ) 三元組格式定為本專案輸出標準,供 crack_detection_* 直接吃。
- [ ] 區分 **crack vs craquelure** 兩類:craquelure 用密集網狀(80–150 條/圖),crack 用稀疏長裂、順木紋方向 — 以此直接調整兩類的合成比例,對症解決失衡。
- [ ] 評估把 masked hybrid loss + detector-guided logit 移植到下游 SepSAM2/unet 偵測器。
- [ ] 針對木材域:設計木紋導向的裂縫走向先驗、加入髒汙/破損干擾的合成,prep 後與 CycleGAN([[2017_zhu_cyclegan]])做真實感增強的對照實驗。
