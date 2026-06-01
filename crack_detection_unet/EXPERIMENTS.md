# Experiments — crack_detection_unet

> 每跑完一次實驗補一列。指標為 **crack 類** val（background 一律 ~0.99，不列）。
> best = 80 epoch 內 crack IoU 最高的那個 epoch；final = 最後一個 epoch。
> 下表 9 筆為遷移前的 legacy run（原 `outputs/`，約 2026-05-25/26），沿用原名、無 date 前綴。

| 日期 | run (slug) | 目的 | best crack IoU / F1 (@ep) | final crack IoU / F1 | 結論 | run 路徑 |
|------|-----------|------|---------------------------|----------------------|------|----------|
| ~2026-05-26 | expert_crack_v3_clahe_fold3_resnet50 | CLAHE 5-fold | 0.1710 / 0.2920 (@53) | 0.1576 / 0.2723 | 全體最佳，但仍低 | `runs/expert_crack_v3_clahe_fold3_resnet50/` |
| ~2026-05-26 | expert_crack_v3_fold3_resnet50 | plain 5-fold | 0.1717 / 0.2931 (@46) | 0.1305 / 0.2309 | 與 clahe_fold3 相當 | `runs/expert_crack_v3_fold3_resnet50/` |
| ~2026-05-26 | expert_crack_v3_clahe_fold1_resnet50 | CLAHE 5-fold | 0.0720 / 0.1344 (@15) | 0.0000 / 0.0000 | final 退化崩潰 | `runs/expert_crack_v3_clahe_fold1_resnet50/` |
| ~2026-05-26 | expert_crack_v3_clahe_fold2_resnet50 | CLAHE 5-fold | 0.0635 / 0.1194 (@54) | 0.0584 / 0.1104 | 低 | `runs/expert_crack_v3_clahe_fold2_resnet50/` |
| ~2026-05-26 | expert_crack_v3_fold2_resnet50 | plain 5-fold | 0.0612 / 0.1154 (@39) | 0.0605 / 0.1141 | 低 | `runs/expert_crack_v3_fold2_resnet50/` |
| ~2026-05-26 | expert_crack_v3_fold1_resnet50 | plain 5-fold | 0.0446 / 0.0854 (@8) | 0.0027 / 0.0055 | final 近崩潰 | `runs/expert_crack_v3_fold1_resnet50/` |
| ~2026-05-25 | expert_crack_v3_clahe_fold0_resnet50 | CLAHE 5-fold | 0.0069 / 0.0137 (@25) | 0.0000 / 0.0000 | 崩潰 | `runs/expert_crack_v3_clahe_fold0_resnet50/` |
| ~2026-05-25 | expert_crack_v3_fold0_resnet50 | plain 5-fold | 0.0053 / 0.0105 (@24) | 0.0001 / 0.0002 | 崩潰 | `runs/expert_crack_v3_fold0_resnet50/` |
| ~2026-05-26 | expert_crack_v3_final_resnet50 | 全資料最終模型 (50ep) | 無 val（全資料訓練，無切分） | — | 產出最終權重 | `runs/expert_crack_v3_final_resnet50/` |

## 觀察
- crack 類整體偏低（best ≈ IoU 0.17 / F1 0.29），且多個 fold final epoch 退化至近 0 → 訓練不穩、類別極度不平衡。
- CLAHE 與 plain 差異不大；fold3 兩者皆最佳，fold0 兩者皆崩潰 → 受 fold 切分影響大。
- 下一步建議：早停取 best.pt 而非 last.pt；檢視 crack recall；調整 class weight / loss。
