# Topic: 文物 crack / craquelure 資料合成與失衡

> 主題綜述。服務 [[data-generation-project]]:為木質文物合成 crack/craquelure 劣化資料,解決標註稀少 + crack 對 craquelure 嚴重失衡。
> 更新:2026-06-02。標註資料來源深度:**[全文]** 精讀 / **[fetch]** 抓網頁正文 / **[snippet]** 僅搜尋摘要,數字未核實。

## 1. 核心方法:合成劣化資料

### 程序式 (procedural) 合成
- **[全文] Synthetic Craquelure Generation (Cuch-Guillén 2026, arXiv:2602.12742)** → [note](../notes/2026_cuch-guillen_synthetic-craquelure.md)
  - Bézier 軌跡 + tapered 半徑剖面 + branching;乾淨 WikiArt 圖上造 (I, M, Ĩ) 三元組;免標註訓練。**本專案 v0 合成原語的首選參考。**
- **[snippet] CutMix Crack Synthesis (Concrete crack seg, Sensors/PMC 2022)** — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9823640/
  - 用 CutMix 把裂縫貼到無裂背景做資料合成 + temporal fusion。低成本擴增 positive 樣本的思路,可借。
- **[snippet] CrackModel: Dataset Synthesis for Data Scarcity (MDPI Buildings 15(7):1053, 2025)** — https://doi.org/10.3390/buildings15071053
  - 針對 crack 偵測資料稀少提出資料集合成法。MDPI 抓取 403,僅摘要;待 fetch 細節。

### GAN / 生成式
- **[全文] CycleGAN (Zhu 2017, arXiv:1703.10593)** → [note](../notes/2017_zhu_cyclegan.md)
  - 非配對域互轉;對本專案是 **sim-to-real 真實感增強**工具(合成劣化→真實木材劣化域),不負責裂縫幾何。
- **[snippet] Generative AI-Driven Data Augmentation for Crack Detection (MDPI Electronics 13(19):3905, 2024)** — https://www.mdpi.com/2079-9292/13/19/3905
- **[snippet] Multi-stage GANs for pavement crack images (Eng. App. of AI, 2023)** — https://www.sciencedirect.com/science/article/abs/pii/S0952197623019516
  - 多階段 GAN 穩定生成高品質裂縫影像;StarGANv2 + ReMix 保形狀。
- **[snippet] GAN-based CNN for pavement crack segmentation (2023)** — https://www.sciencedirect.com/science/article/pii/S2590123023003948

## 2. 類別失衡 (crack 像素極少)
- **[snippet] Crack segmentation of imbalanced data: the role of loss functions (Engineering Structures, 2023)** — https://www.sciencedirect.com/science/article/pii/S0141029623014037
  - 系統比較 loss(joint loss / focal 等)對裂縫失衡的影響。**選 loss 的直接參考。**
- 對照 [[2026_cuch-guillen_synthetic-craquelure]] 的 masked hybrid loss + Dice + detector-guided logit,也是對付失衡的手段。

## 3. 木質文物 (wood / timber heritage) 域
- **[fetch] SMG-Net (Fang et al., PLoS One 20(11), 2025)** — https://pmc.ncbi.nlm.nih.gov/articles/PMC12629464/ ;code: github.com/HuiZhenxing
  - 閩南古建木構裂縫分割。**域內資料極稀少**:僅 256 張原圖 → 擴增切塊到 2,400 (256×256)。
  - SMG-Net (improved U-Net + SACP/MRFE/GSSFusion):木構 mIoU 81.12 / F1 85.13;勝 Swin-Unet (78.89/83.91)、SegFormer (72.31/81.91)、U-Net 變體 (mIoU 63–66)。
  - 佐證:木構裂縫域真實資料嚴重不足 → 合成資料有明確需求。
- **[snippet] Deep-Learning Crack ID for Wooden Components in Ancient Chinese Timber Structures (Zhang, Struct. Control Health Monit., Wiley 2024)** — https://doi.org/10.1155/2024/9999255
  - 古中式木構件裂縫;資料集約 501 張(450 train / 51 val)。Wiley 付費牆 (402),僅摘要。
- **[snippet] DADNet: dual-attention crack seg on tomb murals (2024)** — researchgate 385176759
- **[snippet] Deep-Learning Crack Detection on Cultural Heritage Surfaces (MDPI Appl. Sci. 15(14):7898, 2025)** — https://www.mdpi.com/2076-3417/15/14/7898

## 4. 對本專案的啟示 (synthesis)
1. **資料稀少是全域共識**:木構裂縫公開資料常僅數百張(SMG-Net 256、Zhang 501)→ 合成是合理且必要路線。
2. **路線分工**:幾何走**程序式**(Bézier+tapered+branching,可控、可標)、真實感走**GAN/域轉換**(CycleGAN sim-to-real,不動幾何)。這對應本專案兩篇核心 PDF 的天然分工。
3. **失衡兩條腿一起上**:(a) 合成時刻意調 crack:craquelure 的條數/密度比例;(b) 訓練端用 focal/Dice/masked hybrid loss + detector-guided 先驗。
4. **木材域特化待補**:現有合成法都針對繪畫/路面/混凝土;需加入木紋導向裂縫走向、節點、髒汙/破損干擾 — 這是本專案的研究增量。
5. **待辦文獻**:CrackModel、imbalanced-loss、Zhang 2024 三篇值得之後取得全文精讀(目前 MDPI/Wiley 擋抓)。
