# Decision: 共用程式抽套件 + 獨立資料樞紐

- 日期：2026-06-02
- 對應計畫：`_research/plans/2026-06-02-shared-lib-and-data-hub.md`

## 決定
根除跨專案 `sys.path.insert(別人的根目錄)` 耦合（業界做法：shared library + editable install + 共用資料樞紐）。

## 做了什麼
- **`research/_lib/crackseg_common/`**：把 `dataset/augment/losses/metrics/data_utils` 從 crack_detection_sam2 根目錄抽出成套件；`sam2_env`、`unet_env` 各 `pip install -e _lib`。全 repo import 改為 `from crackseg_common.X import …`，移除 unet 的 `SAM2_ROOT` hack。
- **`research/_data/`**：`crack_detection_sam2/data` 移到頂層；sam2 config、unet train、SepSAM2 4 scripts、sam2 scripts 的 `PROJECT_ROOT/"data"` 全部改指 `_data`。
- RULES §1 加入 `_lib`/`_data`；§4.5 新增「禁止跨專案 sys.path import」規則。

## 驗收（全部符合）
- 兩 venv `import crackseg_common` OK；裸名 import grep=0；unet/sam2 `--help` exit 0；model/test import-clean。
- `_data` 含全部資料集、舊路徑 grep=0、PIL 開圖 smoke OK。

## 例外
- `eval_crack_type.py` 跨 expert 評估仍需 unet_model，保留 `UNET_ROOT`（已在檔內註記）。

## 後續
- 新 model 專案：venv 跑 `pip install -e ../_lib`，資料指向 `../_data`，不得撈他人資料夾。
- editable install 可逆：`pip uninstall crackseg_common`。
