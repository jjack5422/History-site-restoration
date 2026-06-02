# Experiments — crack_detection_SepSAM2

> SepSAM = YOLOv8-Seg+SEA agent 產生提示 + 凍結 SAM2，經 CMC 互校。
> 詳細結果已寫在 `runs/*.md` 與 `docs/SepSAM_復現結果報告.md`；本檔為索引與頭條結論。
> venv：`SepSAM2_env`。資料對齊 dense-seg 的 craquelure/crack split（fold 一致）。

## 結果索引
| 主題 | 檔案 |
|---|---|
| craquelure 4-fold CV（SAM2 refine 有用嗎？） | `runs/craq_sepsam_summary.md` |
| YOLO 召回提升（推論參數槓桿） | `runs/recall_improvement_summary.md` |
| heritage CV（CLAHE / 過採樣 sweep） | `runs/cv_recall_clahe.md` `cv_recall_os.md` `cv_recall_sweep.md` `heritage_cv_summary.json` |
| 完整復現報告 | `docs/SepSAM_復現結果報告.md` |
| 原始 log | `logs/*.log` |

## 頭條結論
- **CMC ≡ YOLO-only**：SAM2 點 prompt 精修對 craquelure 與 crack 都**無正貢獻**（SAM-only F1 僅 0.08-0.13，CMC 幾乎全退回 YOLO，每 fold 採用 0-1 個）。沿中軸正向點 + 凍結 SAM2 在此 domain 不 work。與 [[finding_yolo_prompt_vs_dense_seg]] 一致。
- **唯一有效槓桿是推論參數**（conf=0.25→0.10, iou=0.7→0.5）：crack recall 0.140→0.215（+54% 相對）、mask F1 0.171→0.219，零重訓。CLAHE 與密集面板過採樣幾乎無益。
- **fold2 再次退化**（F1=0.000）——跨所有專案一致的 OOD val-split 現象（見 heritage memory：craq fold2 = L-A4-4 紋理 OOD）。
- CV 操作點 conf=0.10 **不轉移到生產 agent**（ft-2 上反而更差），生產需重掃 conf×iou。

## 跑實驗
從專案根執行（scripts 內 `from src.X import`，`sys.path` 以本目錄為 REPO）：
`SepSAM2_env/bin/python scripts/eval.py --config configs/<...>.yaml --images <...> --masks <...>`
