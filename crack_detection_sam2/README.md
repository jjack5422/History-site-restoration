# crack_detection_sam2

> 專案卡。進來先讀這頁。注意 §資料與 §架構：`labeled32_*` 已退休，只保留在歷史 run/log/spec 中供重現。

## 目標 (Goal)
在現行 512 tile 資料上做牆面缺陷 dense segmentation / SAM2 refine，重點是 **craquelure**（龜裂/裂紋網）與 **crack**（裂縫）兩類 expert / joint 路線比較。

目前 canonical 資料線：
- `_data/multiclass_512_dataset/` — 現行 expanded multiclass dataset，1027 tiles；mask id: `0 bg / 1 crack / 2 loss / 3 shrinkage / 4 craquelure / 5 flaking / 6 stain / 255 ignore`。
- `_data/multiclass_512_craq_bin/` — 從 multiclass dataset 派生的 craquelure binary 訓練集（craquelure id 4 -> foreground）。
- `_data/multiclass_512_crack_bin/` — 從 multiclass dataset 派生的 crack binary 訓練集（crack id 1 -> foreground）。
- `_data/craq_512_dataset_711_0-94/` — 2026-06-22 canonical 0-94 craquelure binary baseline，保留給 711-tile baseline / 舊比較。

`_data/labeled32_crack_v3/`、`_data/labeled32_craq_v3/` 是早期 32 張 seed dataset，已於 2026-06-25 從 `_data/` 移除；只應出現在歷史文件、舊 run manifest、或可重現性說明中。

## 狀態 (Status)
- 目前同條件 craquelure fold0 比較（1027-tile craq binary）：SAM2-refine IoU **0.550** > ResUNet **0.534** > DeepLabV3+ **0.444** > SegFormer **0.407**。見 `EXPERIMENTS.md` 2026-06-29。
- 早期 `labeled32` 4-fold 家族 A-E 與 fold2 退化問題是 legacy baseline，不再代表目前資料版本。
- demo / deployment 類流程優先使用 multiclass-derived binary datasets；需要舊 711 baseline 時才用 `_data/craq_512_dataset_711_0-94/`。

## 環境 (Environment)
- venv：`/home/zzz90/research/sam2_env`（`source sam2_env/bin/activate`）

## Lab Safety
- 禁止使用 `sudo pip install`。
- 禁止把套件安裝到 system Python、base conda、或實驗室共用環境。
- 所有安裝都必須在專案 venv 或 Docker image 內執行；優先使用明確路徑如 `/home/zzz90/research/sam2_env/bin/python -m pip ...`。

## 進入點 (Entry points) — 一律從專案根目錄執行
| 路線 | 訓練 | 模型檔 |
|---|---|---|
| dense-seg ResUNet | `python train.py ...` | `model_seg.py` |
| full-FPN | `python train.py ... (full_fpn)` | `model_seg_full_fpn.py` |
| learnable prompt | `python train_prompt.py ...` | `model_prompt_seg.py` |
| prompted frozen SAM2 | `python train_promptsam2_craq.py ...` | `model_prompted_sam2.py` |
| decoder-only SAM2 | （待補 train 入口） | `model_decoder_seg.py` |

預測/評估：`predict_full.py`、`predict_prompt_overlay.py`、`eval_lineproc_craq.py`、`dump_preds_denseseg.py`。

## 架構：共用資料/訓練工具在 `_lib/crackseg_common`
共用 dataset / augment / losses / metrics / data_utils 已抽到：

- `_lib/crackseg_common/`

本專案與 `crack_detection_unet` 目前都應使用 `from crackseg_common...`。早期文件或舊 run 若提到 `from dataset import ...`、`SAM2_ROOT`、或 `crack_detection_sam2/data/labeled32_*`，視為 legacy 記錄。

SAM2 專案根目錄仍保留模型與 entry scripts；不要做大搬移，除非同步更新 `crack_detection_sam2`、`crack_detection_unet`、scripts、tests 的 import 與 run manifest。

## 已移除 (Removed)
`gt_points.py`（判定無用）連同 import 它的 promptsam2 路線檔一併移除：
`dump_preds_promptsam2.py`、`eval_promptsam2_craq.py`、`train_promptsam2_craq.py`、`tests/test_gt_points.py`。
模型定義 `model_prompted_sam2.py` 與 `tests/test_prompted_sam2_forward.py` 保留（不依賴 gt_points）；
promptsam2 的歷史實驗結果仍記錄在 `EXPERIMENTS.md` 家族 D（`runs/promptsam2_craq_fold*`）。

## 資料與產出位置
- `_data/`：shared data hub。現行資料清單見 `_data/README.md`。
- `data/`：legacy project-local data 位置；不要新增新資料到這裡。
- `runs/`（原 `outputs/`）：每個實驗一資料夾，含 `args.json` `log.json` `train.log` `best.pt`。小檔進 git，`*.pt` 不進。
- `preds/`、`checkpoints/`：gitignored。
- `docs/`：`TRAINING_NOTES.md` 及各路線 summary。

## Notion 索引 (架構筆記)
| 頁面 | 內容 | 對應檔 |
|---|---|---|
| [SAM2PromptSeg 架構詳圖](https://app.notion.com/p/376283855a37811bba40f825afc815df) | learnable-prompt 路線 code-faithful 詳圖：forward 逐步張量 trace、frozen/trainable、param_groups | `model_prompt_seg.py` |
| [Model Architectures: SAM2+FPN / SAM2+Prompt / ResUNet](https://app.notion.com/p/36e283855a37817eb9d5ef97cfb67a84) | 三種模型架構 overview 對照（FPN head / prompt decoder / ResUNet） | `model_seg.py`, `model_prompt_seg.py` |
