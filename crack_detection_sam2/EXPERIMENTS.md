# Experiments — crack_detection_sam2 (craquelure)

> 前景類別 = **craquelure**。指標為 val；best = 80 epoch 內前景 IoU 最高者，final = 末 epoch。
> 詳情見各 `runs/<name>/log.json`。legacy run（原 `outputs/`，約 2026-05），沿用原名。
> 觀察：**fold2 在所有家族都退化**；prompt 系列 recall 高 precision 低。

## A. dense-seg ResUNet (`model_seg.py`)
| run | best IoU / F1 (@ep) | final IoU / F1 | 註 |
|---|---|---|---|
| expert_craq_v3_fold3_small | 0.482 / 0.651 (@67) | 0.478 / 0.647 | 最佳之一 |
| expert_craq_v3_clahe_fold3_small | 0.487 / 0.655 (@49) | 0.482 / 0.651 | 最佳之一 |
| expert_craq_v3_clahe_fold0_small | 0.464 / 0.634 (@19) | 0.423 / 0.594 | |
| expert_craq_v3_fold1_small | 0.432 / 0.603 (@55) | 0.427 / 0.598 | |
| expert_craq_v3_clahe_fold1_small | 0.443 / 0.614 (@55) | 0.430 / 0.601 | |
| expert_craq_v3_fold0_small | 0.407 / 0.579 (@19) | 0.382 / 0.553 | |
| expert_craq_v3_fold2_small | 0.017 / 0.033 (@15) | 0.009 / 0.017 | 退化 |
| expert_craq_v3_clahe_fold2_small | 0.022 / 0.044 (@40) | 0.015 / 0.029 | 退化 |
| expert_craq_v3_final_small | 無 val（全資料 50ep） | — | 最終權重 |

## B. full-FPN (`model_seg_full_fpn.py`)
| run | best IoU / F1 (@ep) | final IoU / F1 | 註 |
|---|---|---|---|
| fullfpn_craq_v3_fold3_small_retry | 0.483 / 0.651 (@39) | 0.472 / 0.642 | 最佳 |
| fullfpn_craq_v3_clahe_fold3_small | 0.470 / 0.639 (@64) | 0.461 / 0.631 | |
| fullfpn_craq_v3_fold1_small_retry | 0.464 / 0.634 (@42) | 0.453 / 0.623 | |
| fullfpn_craq_v3_clahe_fold1_small | 0.457 / 0.627 (@62) | 0.446 / 0.617 | |
| fullfpn_craq_v3_fold0_small_retry | 0.424 / 0.596 (@41) | 0.401 / 0.573 | |
| fullfpn_craq_v3_clahe_fold0_small | 0.434 / 0.606 (@45) | 0.406 / 0.578 | |
| fullfpn_craq_v3_fold2_small_retry | 0.033 / 0.064 (@51) | 0.030 / 0.059 | 退化 |
| fullfpn_craq_v3_clahe_fold2_small | 0.022 / 0.044 (@15) | 0.012 / 0.023 | 退化 |
| fullfpn_craq_fold0-3_small | 無 log.json（舊版，僅 .log） | — | 見 train.log |

## C. learnable prompt (`model_prompt_seg.py`)
| run | best IoU / F1 (@ep) | final IoU / F1 | 註 |
|---|---|---|---|
| prompt_craq_fold0_small | 0.434 / 0.605 (@25) | 0.355 / 0.524 | recall 高 precision 低 |
| prompt_craq_fold3_small | 0.371 / 0.541 (@61) | 0.319 / 0.484 | |
| prompt_craq_fold1_small | 0.368 / 0.538 (@40) | 0.306 / 0.469 | |
| prompt_craq_fold2_small | 0.030 / 0.058 (@74) | 0.018 / 0.035 | 退化 |

## D. prompted frozen SAM2 (`model_prompted_sam2.py`)
> 訓練/評估腳本 `train_promptsam2_craq.py`/`eval_promptsam2_craq.py` 已移除（依賴已刪的 gt_points）；以下為歷史結果。

| run | best IoU / F1 (@ep) | final IoU / F1 | 註 |
|---|---|---|---|
| promptsam2_craq_fold1 | 0.457 / 0.627 (@42) | 0.395 / 0.566 | |
| promptsam2_craq_fold3 | 0.430 / 0.601 (@53) | 0.396 / 0.567 | |
| promptsam2_craq_fold0 | 0.428 / 0.599 (@32) | 0.368 / 0.538 | |
| promptsam2_craq_fold2 | 0.108 / 0.195 (@59) | 0.088 / 0.161 | 退化（但比其他家族 fold2 好） |

## E. decoder-only no-prompt (`model_decoder_seg.py`, `SAM2DecoderSeg`)
> 捨棄 prompt encoder，只借 SAM2 預訓練 mask decoder 當 seg head（sparse=空、dense=learnable 全域 param、image_pe 借 pe_layer）。`train_decoder.py`，`--dense_mode learnable --num_queries 0`，trainable 4.48M。run 目錄：`runs/2026-06-06-decoder-craq-4fold/`。
>
> 2-epoch smoke（fold0）：train loss 1.13→0.75、val IoU 0.177→0.264（確認可訓）。

| run | best IoU / F1 (@ep) | final IoU / F1 | 註 |
|---|---|---|---|
| decoder_craq_fold0_small | 0.432 / 0.604 (@14) | 0.366 / 0.536 | 家族 E 最佳 |
| decoder_craq_fold1_small | 0.379 / 0.550 (@51) | 0.361 / 0.531 | |
| decoder_craq_fold2_small | 0.019 / 0.037 (@15) | 0.011 / 0.021 | 退化 |
| decoder_craq_fold3_small | 0.382 / 0.553 (@59) | 0.356 / 0.525 | |
> mean best IoU = 0.303（含 fold2）/ **0.398**（排除退化的 fold2）。non-fold2 與 prompt 家族 C（0.37-0.43）同級、略低於 dense-seg/full-FPN（0.43-0.49）。fold2 仍退化，與其他家族一致。

## 其他產出（非訓練 run）
- **2026-06-08-prelabel-selected**：對 `_data/selected_slices` 284 張 1024-tile 跑 pre-label（同 final experts, tile512/stride256）→ 打包成 CVAT 'Segmentation mask 1.1' zip `runs/2026-06-08-prelabel-selected/cvat_segmask.zip`。284/284 產出且尺寸/色彩驗證通過。打包腳本 `scripts/package_cvat_segmask.py`。
- **2026-06-08-prelabel-image**：對 `_data/image` 30 張全解析度原圖跑 pre-label（final experts: dense-seg craq + ResUNet-50 crack, tile512/stride256）→ CVAT VOC palette。30/30 產出且尺寸/色彩驗證通過；只含 crack+craq 兩類，餘三類待人工補。run 路徑 `runs/2026-06-08-prelabel-image/`（manifest + merged/voc_palette）。

- **2026-06-08-cvat-agent**：把 final experts 包成 app.cvat.ai 本機輔助標註 AI agent（只產 craquelure+crack 兩類 mask）。專屬 venv `cvat_agent_env` + native function `crack_detection_sam2/cvat_agent/craq_crack_func.py` + register/run 腳本。離線 env/function smoke 皆通過；線上因免費方案鎖背景標註，改走 `cvat-cli task auto-annotate` 寫回 task 2282525（8 frames，附加模式）→ crack 731 + craquelure 585 mask，符合。run 路徑 `runs/2026-06-08-cvat-agent/manifest.md`。

`runs/` 內另有 eval/可視化目錄：`eval_crack_type_4fold_clahe`、`eval_postproc_thr0.5_erode1`、`prompt_craq_fold*_overlay`、`promptsam2_craq_fold2_viz`、`pre_label_v3*`，以及未歸位的 `pre_label_v3_run.log`、`promptsam2_craq.log`（跨 fold 合併 log）。

## 小結
- 四家族 best 都落在 IoU 0.43-0.49 區間，差異不大；dense-seg / full-FPN 略穩。
- **fold2 普遍退化**是跨家族一致現象 → 優先查 fold2 切分。
- prompt 系列 final 明顯掉（過擬合/不穩），應取 best.pt。
