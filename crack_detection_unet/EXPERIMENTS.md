# Experiments — crack_detection_unet

> 每跑完一次實驗補一列。指標為 **crack 類** val（background 一律 ~0.99，不列）。
> best = 80 epoch 內 crack IoU 最高的那個 epoch；final = 最後一個 epoch。
> 下表 9 筆為遷移前的 legacy run（原 `outputs/`，約 2026-05-25/26），沿用原名、無 date 前綴。

| 日期 | run (slug) | 目的 | best crack IoU / F1 (@ep) | final crack IoU / F1 | 結論 | run 路徑 |
|------|-----------|------|---------------------------|----------------------|------|----------|
| ~2026-05-26 | expert_crack_v3_clahe_fold3_resnet50 | CLAHE 5-fold | 0.1710 / 0.2920 (@53) | 0.1576 / 0.2723 | 全體最佳，但仍低 | `runs/expert_crack_v3_clahe_fold3_resnet50/` |
| ~2026-05-26 | expert_crack_v3_clahe_fold1_resnet50 | CLAHE 5-fold | 0.0720 / 0.1344 (@15) | 0.0000 / 0.0000 | final 退化崩潰 | `runs/expert_crack_v3_clahe_fold1_resnet50/` |
| ~2026-05-26 | expert_crack_v3_clahe_fold2_resnet50 | CLAHE 5-fold | 0.0635 / 0.1194 (@54) | 0.0584 / 0.1104 | 低 | `runs/expert_crack_v3_clahe_fold2_resnet50/` |
| ~2026-05-25 | expert_crack_v3_clahe_fold0_resnet50 | CLAHE 5-fold | 0.0069 / 0.0137 (@25) | 0.0000 / 0.0000 | 崩潰 | `runs/expert_crack_v3_clahe_fold0_resnet50/` |
| ~2026-05-26 | expert_crack_v3_final_resnet50 | 全資料最終模型 (50ep) | 無 val（全資料訓練，無切分） | — | 產出最終權重 | `runs/expert_crack_v3_final_resnet50/` |
| 2026-06-15 | craq-resunet50-alldata-2026-06-15 | **craquelure**(非crack) 全資料重訓 +176 batch4 白色 craquelure tile, 80ep 無val取last.pt | 無 val | 無 val | 修白色 craquelure 盲點:batch4 GT 處 craq prob 中位數 **0.001→0.836**、frac>0.25 **1%→92%**(train,非泛化)。供 refine 重訓 | `runs/craq-resunet50-alldata-2026-06-15/` |
| 2026-06-22 | craq-resunet50-alldata-0-94gt-2026-06-22 | craquelure 全資料重訓,**改用 canonical 0-94 GT**(乾淨 711,無半成品 R2),80ep 無val取last.pt | 無 val | 無 val | 取代建在過時 _seg95 的舊權重(craq +9.5%、3 tile 補回前景);loss→0.336;供 0-94 GT refine 重訓 | `runs/craq-resunet50-alldata-0-94gt-2026-06-22/` |
| 2026-06-22 | craq-resunet50-demoholdout-0-94gt-2026-06-22 | 0-94 GT,**leak-free demo-holdout**(held-out L-1RB1-1/A4-8/R-A4-3,train356/val355),80ep | held-out IoU **0.222** / Dice 0.364 | P0.668 R**0.250** | 誠實跨站點泛化:recall 崩(漏75%);held-out craq 70% 來自 A4-8 dense batch4(全沒看過)。供 leak-free refine | `runs/craq-resunet50-demoholdout-0-94gt-2026-06-22/` |
| 2026-06-24 | craq-resunet50-multiclass917-alldata-bestpt | craquelure expert,multiclass_512_dataset(917)衍生二值集,全資料 best.pt(val=train) | craq IoU **0.5365** / F1 0.6983 (@66) | P0.569 / R0.903 | 全集偏樂觀;供 refine + demo 推論;refine 接手把 0.5365→0.5930 | `runs/craq-resunet50-multiclass917-alldata-bestpt-2026-06-24/` |
| 2026-06-24 | crack-resunet50-multiclass917-alldata-bestpt | **crack** expert,同資料(crack 0.42% px 極稀疏),全資料 best.pt(val=train) | crack IoU **0.2216** / F1 0.3627 (@76) | P0.225 / R0.942 | crack 難:過度預測(R高P低),與過往一致;best.pt 避開 final 崩潰;供 demo 推論 | `runs/crack-resunet50-multiclass917-alldata-bestpt-2026-06-24/` |

> plain（非 CLAHE）4 fold 權重已於 2026-06-10 刪除（只留 CLAHE 版）。當時數據：plain 與 CLAHE 各 fold 相當（fold3 plain 0.1717 vs clahe 0.1710；fold0 兩者皆崩潰）。

## 觀察
- crack 類整體偏低（best ≈ IoU 0.17 / F1 0.29），且多個 fold final epoch 退化至近 0 → 訓練不穩、類別極度不平衡。
- CLAHE 與 plain 差異不大；fold3 兩者皆最佳，fold0 兩者皆崩潰 → 受 fold 切分影響大。
- 下一步建議：早停取 best.pt 而非 last.pt；檢視 crack recall；調整 class weight / loss。

## joint 3-class softmax(bg/crack/craquelure)— 2026-06-25
| 日期 | slug | 目的 | crack IoU | craq IoU | 結論 | run |
|---|---|---|--:|--:|---|---|
| 2026-06-25 | joint-crackcraq-resunet50-allval | option B:互斥 softmax 分離 crack/craq | 0.181 | 0.495 | **不勝雙 expert**(crack −0.041、craq −0.043);overlap=0、端點對,但 content-FP 改判 crack | `runs/joint-crackcraq-resunet50-allval-2026-06-25/` |

結論詳:`_research/decisions/2026-06-25-joint-vs-experts-crackcraq.md`。demo `crack_detection_sam2/runs/predict-demo5-joint-crackcraq-2026-06-25/`。

| 2026-06-29 | unet-craqbin1027-fold0 | craquelure 4-model 比較 anchor(smp.Unet resnet50);craq_bin **1027** fold0 honest val | IoU 0.5335 / P 0.6107 / R 0.8084 / F1 0.6958 @ep48 | 三模型最強單階段;高 recall 高 precision | `runs/unet-craqbin1027-fold0-2026-06-29/` |
| 2026-06-29 | deeplabv3plus-craqbin1027-fold0 | 同上 split/配方,`--arch deeplabv3plus` resnet50 | IoU 0.4443 / P 0.4902 / R 0.8261 / F1 0.6153 @ep49 | 輸 ResUNet −0.089 IoU,主因 precision 低(content-FP 多) | `runs/deeplabv3plus-craqbin1027-fold0-2026-06-29/` |
| 2026-06-29 | segformer-craqbin1027-fold0 | 同上 split/配方,`--arch segformer mit_b0`(3.7M) | IoU 0.4070 / P 0.4400 / R 0.8442 / F1 0.5785 @ep68 | 最弱但容量最小(confound);未測 b2/b5 | `runs/segformer-craqbin1027-fold0-2026-06-29/` |

craquelure 4-model 比較結論:`_research/decisions/2026-06-29-craquelure-model-comparison.md`;彙整表 `crack_detection_sam2/runs/craq-modelcompare-fold0-2026-06-29/`。**ResUNet 0.534 ≫ DeepLabV3+ 0.444 > SegFormer 0.407**,差距在 precision(主流通用模型未勝專案 ResUNet)。
