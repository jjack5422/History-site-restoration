# crack_detection_unet

> ResUNet crack 二分類分割 baseline。專案卡：進來先讀這頁。

## 目標 (Goal)
用 ResUNet（encoder=resnet50, imagenet 預訓練）對 `expert_crack_v3` 的 512 tiles 做 background/crack 二類 dense segmentation，作為 crack-seg 的 UNet baseline，與 SAM2 路線比較。

## 狀態 (Status)
- 已完成 9 次 run（見 `EXPERIMENTS.md`）：plain 5-fold（fold0-3 + final）、CLAHE 4-fold。
- crack 類指標偏低：best 為 fold3 ≈ crack IoU 0.17 / F1 0.29；CLAHE 沒有明顯改善。與 dense-seg crack 難收斂的已知現象一致。
- TODO：診斷 crack recall 偏低、類別不平衡處理；考慮 loss/取樣調整。

## 環境 (Environment)
- venv：`/home/zzz90/research/unet_env`（啟用：`source unet_env/bin/activate`）

## Lab Safety
- 禁止使用 `sudo pip install`。
- 禁止把套件安裝到 system Python、base conda、或實驗室共用環境。
- 所有安裝都必須在專案 venv 或 Docker image 內執行；優先使用明確路徑如 `/home/zzz90/research/unet_env/bin/python -m pip ...`。

## 進入點 (Entry points)
| 用途 | 指令 |
|---|---|
| 訓練 | `python src/train.py --tiles_root <...> --split <...> --fold N --output_dir runs/<date>-<slug>` |
| 預測/評估 | `python src/predict_full.py --image_dir <...> --mask_dir <...> --out_dir <...>` |

## 資料 (Data) — 外部依賴
本專案**不自包資料與部分程式**：
- 資料來自 `crack_detection_sam2/data/labeled32_crack_v3/tiles_512/`。
- `src/train.py`、`src/predict_full.py` 以絕對路徑 `sys.path.insert` 借用 `crack_detection_sam2` 的 `augment` / `dataset` / `metrics` 模組（見檔頭 `SAM2_ROOT`）。
- 因為是絕對路徑，程式位置可搬動但**不要改動 crack_detection_sam2 的路徑**，否則 import 會壞。

## 相關文獻
見 `_literature/topics/crack-segmentation.md`（如尚未建立可用 literature-review skill 補）。
