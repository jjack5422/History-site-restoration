# Plan: data_generation — 木質文物 crack/craquelure 資料合成

- 日期：2026-06-02
- 專案：data_generation（新增於 research/，crack_detection_* 的上游姊妹專案）
- 對應規則：_research/RULES.md

## 目標 (Goal)
為**木質文物古蹟**(具木紋 wood grain、顏料 pigment、破損 damage、髒汙 soiling 等干擾)的劣化
建立**資料合成 pipeline**,產生劣化標的:
- **craquelure**(龜裂網)
- **crack**(裂縫)

動機:真實標註資料量本就稀少,且 **crack 樣本數相對 craquelure 嚴重不足、類別比例失衡**。
合成資料用來補足下游 crack 偵測/分割專案(crack_detection_SepSAM2 / sam2 / unet)的訓練資料。
完成定義:專案歸位 + 兩篇核心文獻歸檔成可檢索筆記 + 網路補充文獻盤點 + 專案目的落檔。

## 背景 / 已知 (Context)
- doc/ 原有兩篇 PDF:
  - Synthetic Craquelure Generation for Unsupervised Painting Restoration (arXiv:2602.12742, 2026)
    — Bézier 曲線 + tapered/branching 幾何在乾淨 WikiArt 影像上合成 craquelure 三元組 (I, M, Ĩ)。
  - CycleGAN (Zhu et al., arXiv:1703.10593, 2017) — 非配對影像轉換,cycle-consistency loss。
- 既有 crack 偵測專案皆在 research/ 下,共用 _literature / _lib / _data。

## 步驟 (Steps)
1. 寫本計畫 → `_research/plans/`(本檔)。
2. 遷移:scaffold `research/data_generation/`;兩篇 PDF 移到 `_literature/papers/`;清掉空舊目錄 → `research-workspace`。
3. 精讀兩篇 PDF,寫 `_literature/notes/` 筆記 + 登錄 `INDEX.md` → `literature-review`。
4. 上網找木質文物劣化合成 / crack 生成 / 類別失衡資料增強文獻,擇要歸檔 → `literature-review`。
5. 記錄專案目的:`README.md` + `_research/decisions/` + memory。

## 檢查點 (Checkpoints — 用來驗收，不可省)
- [ ] CP1: `research/data_generation/` 具標準目錄與 README/EXPERIMENTS;舊 `/home/zzz90/data_generation` 已清空。
- [ ] CP2: 兩篇 PDF 在 `_literature/papers/` 且檔名合規;`notes/` 各有一份含「可行動結論」的筆記;`INDEX.md` 兩列已登錄。
- [ ] CP3: 至少找到 3 篇網路文獻(木質劣化/crack 合成/失衡增強),擇要登錄筆記或 topic 綜述。
- [ ] CP4: 專案目的在 README、decisions、memory 三處一致記錄。

## 產出位置 (Where outputs go)
- 專案:`research/data_generation/`(README/EXPERIMENTS + src/scripts/configs/data/runs/results/docs)
- 文獻:`_literature/papers/`、`_literature/notes/`、`_literature/topics/`、`INDEX.md`
- 結論:`_research/decisions/2026-06-02-data-generation-purpose.md`

## 風險 / 待決 (Risks / Open questions)
- 兩篇文獻聚焦繪畫(painting)craquelure,木質文物的木紋/顏料干擾與此不同 — 合成器需針對木材紋理調整(待後續實驗計畫)。
- crack vs craquelure 的合成策略與比例目標、是否引入 GAN(CycleGAN 風格遷移補真實感)— 待後續 brainstorming/實驗計畫拍板。
- 環境:讀 PDF 暫用 /tmp/pdfenv(pymupdf);專案 venv 之後另定。
