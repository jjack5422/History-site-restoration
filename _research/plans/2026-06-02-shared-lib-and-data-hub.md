# Plan: 抽共用套件 (_lib) + 獨立資料樞紐 (_data)

- 日期：2026-06-02
- 專案：cross-project（crack_detection_sam2 / unet / SepSAM2）
- 對應規則：_research/RULES.md
- 起因：根除「一個專案用 sys.path 撈另一個大目錄的散檔/資料」的脆弱耦合（曾因搬 unet 造成 UNET_ROOT regression）。目標：每個專案的 .py 只 import「已安裝的套件名」，資料走獨立樞紐。

## 目標 (Goal)
1. 共用程式抽成 editable-install 套件 `_lib/crackseg_common/`，消除跨專案 `sys.path.insert(別人的根目錄)`。
2. 共用資料移到頂層 `_data/`，新 model 專案只需指向 `_data`，不必撈別人的 `data/`。
完成定義：見各 Phase 檢查點全部「符合」。

## 背景 / 已知 (discovery)
- 共用程式閉包（無相互 import 的葉模組）：`augment.py dataset.py losses.py metrics.py data_utils.py`（都在 crack_detection_sam2 根目錄）。
- 消費者：sam2 自身 root .py + scripts + tests；crack_detection_unet/src（借 augment/dataset/losses/metrics）。SepSAM2 不借 sam2 程式。
- `eval_crack_type.py` 另 import unet 的 `unet_model`（跨 expert 評估，本質跨專案）→ 列為**可接受例外**，保留指向 unet/src 的路徑並註記。
- 指向 `crack_detection_sam2/data` 的活躍參照：`sam2/configs/default.yaml`、`unet/src/train.py`、`SepSAM2/scripts/{voc_mask_to_binary,m2_medial_axis,m1_sam2_smoke,eval_sepsam_cv_generic}.py`。
- venvs：`sam2_env` `unet_env` `SepSAM2_env`。Phase A 影響 sam2_env+unet_env；Phase B 影響三者的資料路徑。

## Phase 0 — 規則 (先做，零風險)
- 在 RULES.md 新增：禁止跨專案 `sys.path` import 別人程式；共用程式走 `_lib/` 套件、共用資料走 `_data/`；登記這兩個新樞紐。

## Phase A — 共用程式套件 `_lib/crackseg_common/`
步驟：
1. 建 `_lib/crackseg_common/`：`pyproject.toml`(name=crackseg_common) + `__init__.py`；用 `git mv` 把 5 個模組從 sam2 根目錄移進來。
2. 兩個 venv 各做 `pip install -e _lib`（sam2_env、unet_env）。
3. 更新所有 import 點：`from dataset import X` → `from crackseg_common.dataset import X`（augment/losses/metrics/data_utils 同理）。涵蓋 sam2 root .py、sam2 scripts、unet/src。
4. 移除不再需要的 sys.path hack：unet/src 的 `SAM2_ROOT` 區塊；sam2 scripts 若只為這些模組而插 PROJECT_ROOT 則移除（若仍 import 其他 sam2 本地檔則保留並註記）。
5. 驗證（見檢查點）。

檢查點 A（可觀察）：
- [ ] CP-A1：`pip show crackseg_common` 在 sam2_env 與 unet_env 都成功；`_lib/crackseg_common/` 含 5 模組 + pyproject。
- [ ] CP-A2：全 repo `grep -rE "^(from|import) (augment|dataset|losses|metrics|data_utils)\b"`（裸名）回傳 0 筆（全部改成 crackseg_common.*）。
- [ ] CP-A3：unet/src 內無 `SAM2_ROOT`／無指向 sam2 根目錄的 sys.path；`unet_env/bin/python crack_detection_unet/src/train.py --help` exit 0。
- [ ] CP-A4：`sam2_env/bin/python crack_detection_sam2/{train.py,predict_full.py} --help` exit 0；`sam2_env/bin/python -m pytest crack_detection_sam2/tests/test_lineproc.py crack_detection_sam2/tests/test_prompted_sam2_forward.py` 通過或 import-clean。
- [ ] CP-A5：除 `eval_crack_type.py`→unet_model（已註記例外）外，無任何專案 sys.path 撈他人程式。

## Phase B — 獨立資料樞紐 `_data/`
步驟：
1. `mv crack_detection_sam2/data research/_data`（data 已 gitignore，純磁碟搬移；同檔系統瞬間完成）。
2. 更新 6 處活躍參照：絕對路徑 `…/crack_detection_sam2/data` → `…/research/_data`（sam2 config、unet train.py、SepSAM2 4 scripts，含 docstring 範例）。
3. 更新 RULES.md（登記 _data）與 heritage memory（資料路徑變更：selected_slices / 1-31test 等）。
4. 驗證（見檢查點）。

檢查點 B（可觀察）：
- [ ] CP-B1：`research/_data/` 含 labeled32_crack_v3、labeled32_craq_v3、1-31test、selected_slices、splits.json；舊 `crack_detection_sam2/data` 不存在。
- [ ] CP-B2：全 repo `grep -r "crack_detection_sam2/data"`（排除歷史 args.json/log）回傳 0 筆。
- [ ] CP-B3：smoke 測試——用更新後的某 config/路徑，python 檢查一個 split 檔 + 一張對應 image/mask 路徑確實存在且可開啟。
- [ ] CP-B4：RULES 與 heritage memory 的資料路徑已更新。

## 產出位置 (Where outputs go)
- 套件：`research/_lib/crackseg_common/`
- 資料：`research/_data/`
- 規則更新：`_research/RULES.md`
- 完成結論：`_research/decisions/2026-06-02-shared-lib-and-data-hub.md`

## 風險 / 待決 (Risks / Open questions)
- 動到 3 個專案的 import 與資料路徑，接觸面大 → 每個 CP 逐一 `--help`/pytest/smoke 驗證後才進下一步。
- editable install 會寫進各 venv 的 site-packages（可逆：`pip uninstall crackseg_common`）。
- 待你拍板：(a) `data_utils` 要不要一起進 lib（建議要，scripts 才能完全免 sys.path）；(b) Phase A 與 B 要一起做還是分兩次 commit/驗收（建議分兩段，各自可獨立回溯）。
