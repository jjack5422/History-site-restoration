# RULES.md — 研究工作區規則書

> 這是 AI 與你共同遵守的規則。任何複雜任務開始前，AI 必須先讀本檔。
> 違反字面規則 = 違反規則精神。產出沒有落在規定位置 = 沒完成。

## 1. 目錄結構（位置明確）

```
research/
├─ _literature/                文獻庫（跨專案共用）
│  ├─ INDEX.md                 總表，每篇一列
│  ├─ papers/                  PDF 原檔  <year>_<firstauthor>_<slug>.pdf
│  ├─ notes/                   每篇一份筆記  <year>_<firstauthor>_<slug>.md
│  └─ topics/                  主題綜述  <topic-slug>.md
├─ _research/                  研究總管（跨專案共用）
│  ├─ RULES.md                 本檔
│  ├─ plans/                   任務計畫  YYYY-MM-DD-<task-slug>.md
│  └─ decisions/               決策/結論日誌  YYYY-MM-DD-<topic>.md
└─ <project>/                  各專案自包
   ├─ README.md                專案卡：目標/狀態/venv/進入點
   ├─ EXPERIMENTS.md           實驗索引表，每次 run 一列
   ├─ data/                    輸入資料（不改原始檔；衍生檔放子資料夾）
   ├─ configs/                 設定檔（YAML/JSON），實驗以此為準
   ├─ scripts/                 一次性腳本、工具
   ├─ src/                     可重用模組（model/dataset/metrics…）
   ├─ runs/                    每次實驗一個資料夾，見 §4
   ├─ results/                 整理過、要進論文的圖表/表格
   └─ docs/                    專案文件、訓練筆記
```

## 2. 命名規則

- 日期一律 `YYYY-MM-DD`（絕對日期，不用「今天/上週」）。
- slug 用小寫連字號：`crack-seg-baseline`，不用空格或底線。
- 實驗 run 目錄：`runs/<YYYY-MM-DD>-<slug>/`，slug 要能一眼看出在試什麼。
- 文獻檔名與筆記檔名一致，方便對應。

## 3. 鐵則（解決「沒計畫、結果沒法檢查」）

1. **規則先於執行**：任務複雜（≥3 步、跨檔案、或要跑實驗）就先寫計畫到 `_research/plans/`，定好檢查點，再動手。
2. **位置先於產出**：寫任何檔案前，先確認它該落在上面哪個資料夾。沒有歸屬就先問。
3. **快照先於跑實驗**：跑訓練/評估前，先在 `runs/` 建目錄並寫 manifest（指令、config、資料/環境狀態）。沒 manifest 的結果視為不可信。
4. **檢查點對照驗收**：跑完要對照計畫裡的預期結果，明說「符合 / 不符合 / 待查」，不可只貼數字就當完成。
5. **不堆檔**：根目錄與專案根目錄不放臨時檔；臨時檔放 `scripts/` 或 run 目錄內。

## 4. 實驗 run 目錄規範

每個 `runs/<date>-<slug>/` 至少包含：
- `manifest.md`  — 目的、指令、config 路徑+關鍵超參、資料版本、環境（venv/commit）
- `train.log` / `eval.log` — 原始輸出
- `metrics.json` — 結構化指標（之後可彙整）
- 產出的權重/預測放這裡或在 manifest 註明外部路徑

跑完在 `EXPERIMENTS.md` 補一列：日期 | slug | 目的 | 關鍵指標 | 結論 | run 路徑。

## 5. 分工（哪個 skill 管什麼）

| 需求 | 用哪個 skill |
|---|---|
| 建/整理專案目錄、套用本規範 | `research-workspace` |
| 讀論文、出摘要與可行動結論 | `literature-review` |
| 跑實驗前後的記錄與數據存放 | `experiment-tracking` |
| 複雜任務先出計畫、定規則、分派 | `research-orchestrator` |

## 6. 語言

摘要/結論/筆記用**中英混合**：術語、模型名、指標名保留英文，敘述用繁體中文。
