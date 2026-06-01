# crack_detection_sam2

> 專案卡。進來先讀這頁。注意 §架構：根目錄 .py 是被多方共用的扁平套件，勿隨意搬動。

## 目標 (Goal)
在 `labeled32` tiles 上做牆面缺陷 dense segmentation，並比較多種模型路線。資料有兩套前景類別：
- `data/labeled32_craq_v3/` — **craquelure**（craq，龜裂/裂紋網）
- `data/labeled32_crack_v3/` — **crack**（裂縫，也供 `crack_detection_unet` 使用）

目前已完成的成套實驗為 **craquelure**，比較 4 種模型家族（見 `EXPERIMENTS.md`）。

## 狀態 (Status)
- craquelure best ≈ IoU 0.48 / F1 0.65（expert / fullfpn 的 fold3）。
- **fold2 在所有家族都退化**（IoU ~0.02-0.1）→ 該 fold 切分或資料有問題，待查。
- prompt / promptsam2 路線 recall 很高但 precision 低（~0.37）→ 過度預測。
- TODO：診斷 fold2、prompt 路線的低 precision。

## 環境 (Environment)
- venv：`/home/zzz90/research/sam2_env`（`source sam2_env/bin/activate`）

## 進入點 (Entry points) — 一律從專案根目錄執行
| 路線 | 訓練 | 模型檔 |
|---|---|---|
| dense-seg ResUNet | `python train.py ...` | `model_seg.py` |
| full-FPN | `python train.py ... (full_fpn)` | `model_seg_full_fpn.py` |
| learnable prompt | `python train_prompt.py ...` | `model_prompt_seg.py` |
| prompted frozen SAM2 | `python train_promptsam2_craq.py ...` | `model_prompted_sam2.py` |

預測/評估：`predict_full.py`、`predict_prompt_overlay.py`、`eval_lineproc_craq.py`、`dump_preds_denseseg.py`。

## 架構：根目錄 .py 是共用扁平套件（重要）
根目錄的 model/dataset/augment/metrics/losses/… 以**扁平 import**（`from dataset import …`）被以下三方共用，且都靠 `sys.path` 指向**本專案根目錄**：
- 本專案 entry scripts、`tests/`（`sys.path.insert("..")`）、`scripts/`（`PROJECT_ROOT`）
- **`crack_detection_unet`** 透過 `SAM2_ROOT = .../crack_detection_sam2` 借用 `augment`/`dataset`/`metrics`

→ **不要把根目錄 .py 搬進 src/**，否則上述每一方的 import 都會壞，需同步改十幾處路徑。這是刻意保留的佈局。

## 已移除 (Removed)
`gt_points.py`（判定無用）連同 import 它的 promptsam2 路線檔一併移除：
`dump_preds_promptsam2.py`、`eval_promptsam2_craq.py`、`train_promptsam2_craq.py`、`tests/test_gt_points.py`。
模型定義 `model_prompted_sam2.py` 與 `tests/test_prompted_sam2_forward.py` 保留（不依賴 gt_points）；
promptsam2 的歷史實驗結果仍記錄在 `EXPERIMENTS.md` 家族 D（`runs/promptsam2_craq_fold*`）。

## 資料與產出位置
- `data/`（188M，gitignored）：tiles、masks、splits。
- `runs/`（原 `outputs/`）：每個實驗一資料夾，含 `args.json` `log.json` `train.log` `best.pt`。小檔進 git，`*.pt` 不進。
- `preds/`、`checkpoints/`：gitignored。
- `docs/`：`TRAINING_NOTES.md` 及各路線 summary。
