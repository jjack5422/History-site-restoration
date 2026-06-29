# Design: crack/craquelure 雙 expert ResUNet→SAM2-refine 分割(in-distribution ceiling)

- 日期: 2026-06-29
- 分支: `exp/crack-craq-dualexpert-refine`
- 狀態: **延後**。使用者 2026-06-29 改先做 craquelure 的主流模型比較，見
  `2026-06-29-craquelure-seg-model-comparison-design.md`；本 crack+craq dual-expert ceiling 為後續階段。

## 目標 / 研究問題

用既有的 **ResUNet→SAM2-refine** 兩階段架構，在現行 `_data/multiclass_512_dataset`（1027 tile）
**全資料訓練、不設 holdout、容許洩漏**，回答一個問題：

> 在本專案壁畫資料上，能不能「穩定」把 **crack** 與 **craquelure** 兩類分割出來？
> 各類的 in-distribution 上限是多少、兩類重疊（混淆）有多嚴重？

這是刻意的 **in-distribution ceiling 量測**：先不管泛化，若連記憶都做不到穩定分離，問題就是根本性的；
若做得到，泛化才是下一個問題。對照歷史泛化負結果（held-out crack/craq IoU 偏低、根因標註覆蓋）。

非目標（YAGNI，留後續）:
- 泛化 / leak-free held-out 數字（本實驗明確不做）。
- 把分割接成 region router 自動派工、CVAT 自動 prelabel。
- joint softmax / patch 分類 / 拓樸分離（皆已試，見既有 finding）。

## 架構

兩條獨立 **binary expert**，各自 ResUNet（stage-1）→ SAM2 refine（stage-2）：

```
multiclass_512_dataset (1027 tile, canonical 0-94 GT, crack=1 craquelure=4)
        │  build_binary_datasets (per-class 二值化)
        ├── crack_bin  ── ResUNet crack expert ── dump prob ── SAM2 crack refine  ── crack mask
        └── craq_bin   ── ResUNet craq  expert ── dump prob ── SAM2 craq  refine  ── craquelure mask
                                                                         │
                                       合併（重疊像素 argmax 機率）→ crack/craquelure 分割
```

選雙 expert 而非單一多類模型的理由：SAM2 refine 是 **binary mask-prompt** 機制；joint 3-class softmax
過去（2026-06-25）per-class IoU 未勝雙 expert（crack 0.181<0.222、craq 0.495<0.537）。見
`_research/decisions/2026-06-25-joint-vs-experts-crackcraq.md`。

## 元件

| 元件 | 角色 | 現況 |
|---|---|---|
| `crack_bin` / `craq_bin` 二值集 | per-class 訓練資料 | 舊版 917、且 crack_bin 缺 resunet_prob → **重建到 1027** |
| ResUNet crack expert | stage-1 crack | 舊 917 all-data 版 → **重訓** |
| ResUNet craq expert | stage-1 craquelure | 舊 917 all-data 版 → **重訓** |
| dump prob | 出 per-tile 機率給 refine 當 prompt | crack 從未做 → **新建 crack prob** |
| SAM2 crack refine | stage-2 crack | **不存在 → 新訓** |
| SAM2 craq refine | stage-2 craquelure | tversky28-aug 配方既有 → **重訓** |
| 合併 + 評估 | 兩 mask 合併、per-class 指標、confusion | **新腳本** |

## 資料流 / 關鍵設定

- 二值化: crack=color(255,24,3)→1、craquelure=(102,255,102)→1；ignore=255 保留排除於 loss。
- split: **全資料進 train、val=[]**（沿用 `nofold_all_train` / `allval_split.json` 形式），取 last.pt
  或 all-train best；**明確記錄此為 in-dist、洩漏，數字偏樂觀**。
- refine 配方沿用既有勝出設定: recall 加權 Tversky β0.8/α0.2、`--aug`、thr 掃描對齊 R≈0.80。
- 大圖推論 guards: `MAX_IMAGE_PIXELS=None`、單張跑避開 `cudaErrorUnknown`、GPU 工作序列化勿並行重 IO、
  `NO_ALBUMENTATIONS_UPDATE=1` + `HF_HUB_OFFLINE=1`。
- 共用程式走 `_lib/crackseg_common`，禁止 sys.path hack（RULES §4.5）。

## 評估 / 成功定義

- 主指標: **per-class crack / craquelure 的 IoU、F1（P/R 分開列）**，在全資料（in-dist）。
- 重疊/混淆: 兩二值 mask 的像素重疊比例；合併後 crack↔craquelure 互相誤判量。
- 對照: 各類 expert-only vs expert+refine（refine 是否加值，如 craquelure 歷史 0.538→0.598）。
- demo 大圖質化 overlay（青=crack、藍=craquelure），看線狀 vs 網狀是否在視覺上分得開。
- **成功 = 「穩定」**: craquelure in-dist IoU 維持 ~0.58+；crack 給出明確 in-dist 上限數字
  （誠實預期偏低 ~0.2-0.3 量級，全資料記憶或拉高），並量化兩類重疊是否可接受。
  結論二擇一明確寫出: (a) 雙 expert 能穩定分離 → 下一步攻泛化；(b) 連 in-dist 都不穩 → 哪一類是瓶頸、為什麼。

## 產出位置（RULES）

- runs: `crack_detection_unet/runs/`（兩 expert）+ `crack_detection_sam2/runs/`（兩 refine + 合併評估），
  各 run 有 manifest/log/metrics.json。
- 計畫: `_research/plans/2026-06-29-crack-craq-dualexpert-refine.md`（含檢查點）。
- 結論: `_research/decisions/2026-06-29-crack-craq-dualexpert-ceiling.md`。
- EXPERIMENTS.md 各補列。

## 風險 / 已知陷阱

- crack 稀疏 → expert recall 低、refine 空 prompt 無從修（prompt-bounded 天花板，見 finding）。
- 全資料無 holdout → 不可拿來宣稱泛化；報告務必標明 in-dist/leak。
- 8GB GPU OOM / cudaErrorUnknown → 工作序列化、大圖單張。
- 兩 expert 獨立 → 重疊像素需 argmax 機制決定歸屬，否則 crack∩craq 雙標。
