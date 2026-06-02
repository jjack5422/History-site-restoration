---
title: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)
authors: Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
year: 2017
venue: ICCV 2017 (arXiv:1703.10593, v7 2020)
link: https://arxiv.org/abs/1703.10593
pdf: papers/2017_zhu_cyclegan.pdf
tags: [gan, image-translation, unpaired, cycle-consistency, domain-adaptation, style-transfer]
read_date: 2026-06-02
---

# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)

## TL;DR(一句話結論)
在**沒有配對資料**的情況下學兩個域 X↔Y 的互轉:用兩個生成器 G、F 加兩個判別器 D_X、D_Y,
靠 **cycle-consistency loss**(F(G(x))≈x 且 G(F(y))≈y)約束,讓非配對影像轉換可行。經典奠基論文。

## 問題 (Problem)
影像轉影像(image-to-image translation)在**有配對**訓練資料時(pix2pix)已能做好,
但多數任務拿不到配對資料(標註昂貴、有些任務輸出根本沒有良好定義)。
目標:只給兩個**非配對**影像集合,學會域間互轉。

## 方法 (Method)
- 兩個映射 G: X→Y、F: Y→X,各配一個 adversarial discriminator D_Y、D_X。
- **Adversarial loss**:讓 G(X) 的分佈無法與 Y 區分(F 與 D_X 同理)。
- **Cycle-consistency loss**(核心):forward x→G(x)→F(G(x))≈x;backward y→F(y)→G(F(y))≈y,用 L1。
- 總目標 = 兩個 adversarial loss + λ·cycle loss(論文 λ=10);選配 identity loss 穩定顏色。
- 生成器用 residual blocks,判別器用 70×70 PatchGAN;以 LSGAN(least-squares)取代 log loss 穩定訓練。
- 不依賴任務專屬的相似度函數,也不假設輸入輸出在同一低維嵌入 → 通用。

## 關鍵結果 (Key Results)
- 多任務質性成果:Monet↔photo、zebra↔horse、夏↔冬、collection style transfer、photo enhancement 等。
- Cityscapes labels↔photo 上以 AMT 真偽測試與 FCN-score 量化,顯著優於 CoGAN、SimGAN、pixel-loss、BiGAN 等非配對 baseline(具體數字見論文 Table;此處僅讀到方法與前兩頁,未逐欄抄錄)。
- Ablation:去掉 cycle loss 或只留單向都明顯變差(mode collapse/不穩)。

## 限制與假設 (Limitations)
- 幾何形變大的轉換(如貓↔狗、需大形狀改變)常失敗,偏向紋理/顏色層級的改變。
- 受訓練集分佈限制(如只看過家馬,斑馬轉換會帶上騎師等偽影)。
- 與配對監督(pix2pix)仍有差距;非配對本質有歧義。

## 與我研究的關聯 (Relevance to my work)
對 [[data-generation-project]] 是**真實感增強 / 域轉換**的工具,而非合成幾何本身的來源:
- 程序式合成(見 [[2026_cuch-guillen_synthetic-craquelure]])造出的裂縫/龜裂在外觀上偏「假」;
  可用 CycleGAN 把 synthetic-degraded 域轉到 real-wood-degraded 域,提升真實感(sim-to-real)。
- 非配對特性正好契合:我們有「合成劣化影像」與「真實木質文物劣化影像」兩個**非配對**集合。
- 也可反向當資料增強:把乾淨木材影像轉成帶劣化風格,擴充訓練分佈。

## 可行動結論 (Actionable conclusion)
- [ ] 規劃 sim-to-real 實驗:程序式合成 → CycleGAN 轉真實木材劣化域,對照「純程序式」對下游 crack 偵測 F1/IoU 的影響。
- [ ] 注意其幾何形變弱點:**裂縫的形狀/拓樸應由程序式合成決定,CycleGAN 只負責外觀/紋理真實感**,避免它改動裂縫幾何。
- [ ] 收集非配對的真實木質文物劣化影像集當 target 域(資料可行性待確認)。
- [ ] 評估較新替代法(如 contrastive CUT、diffusion-based translation)是否比 CycleGAN 更適合,作為後續文獻搜尋項。
