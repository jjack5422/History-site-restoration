# Experiments — crack_detection_sam2 (craquelure)

> 前景類別 = **craquelure**。指標為 fold0 val 前景 IoU / F1。
> **2026-06-10 清理**：legacy 探索家族 A–E 的權重與 per-epoch log 已刪除（只省 KB，但精簡目錄）；下表為保留的結論數字，原 run 目錄不再存在。

## 現役（保留）
| 路線 | 結果 | run |
|---|---|---|
| **ResUNet→SAM2 mask 精修(refine)** | best **IoU 0.574** @ep22；recall 加權 Tversky β0.8 @thr0.25 → R0.802/P0.637/IoU0.550，**全面壓制 ResUNet(0.804/0.620/0.538)** | `runs/craq-sam2prompt-tversky28-2026-06-10/`（**推薦預設 @ thr0.25**） |
| **refine 重訓 (新 prob + batch4 白色 craquelure)** 2026-06-15 | best **IoU 0.584** @ep23 / F1 0.738 / P0.661 / R0.834（≥ 舊 0.574，不退化）；讀全資料 ResUNet 重生 prob，fold0 train +176 batch4 tile。**端到端 A4-8 重測上方白色 craquelure 已抓到**（`runs/predict-A4-8-newprob-2026-06-15/`） | `runs/craq-refine-tversky28-newprob-2026-06-15/` |
| **refine + 溫和增強 (`--aug`: HSV/亮對比 + flips/rot90)** 2026-06-15 | best **IoU 0.5925** @ep32 / F1 0.744 / P0.672 / R0.834（vs 上行 0.584，**+0.008 不退化**）。跨底材 R-A4-3 右區覆蓋 3.63%→**3.86%**（綠/暗底填更完整）。**漏標 OOD 區(左區)仍救不了**→aug 是小補強，標註覆蓋才是主槓桿 | `runs/craq-refine-tversky28-aug-2026-06-15/`（**新推薦預設 @thr0.25**） |
| **crack∪craquelure 合併單類 pipeline** 2026-06-15 | 新 dataset `crackcraq_0-94_v1`(0-94 181 label union + batch4 3 panel,1266 tile)。ResUNet val IoU **0.440**(F1 0.611/P0.495/R0.797);refine val IoU **0.423**(≈ResUNet,**未勝出**,best@ep10 後漂移)→ 合併類下兩階段 refine 沒加值(對比純 craquelure refine 0.574>0.538) | ResUNet `crack_detection_unet/runs/crackcraq-resunet50-fold0-2026-06-15/`;refine `runs/crackcraq-refine-fold0-2026-06-15/` |
| **prompt-ceiling ablation（負結果）** 2026-06-15 | 解「ResUNet 漏→refine 救不回」:**(2)prompt-dropout refine 無效**(覆蓋≈baseline,凍結 encoder 補不出 recall;val 0.5924≈aug);**(3)ridge∪ResUNet prompt 不可全自動**(ridge 把彩繪線當裂紋,門神 union 多出 +8% 幾乎全 FP)。baseline+resu 仍最佳;主槓桿仍是標註覆蓋 | `runs/ablation-prompt-ceiling-2026-06-15/`（4 圖 panel+diff crop） |
| SAM2 mask-prompt(對稱 loss) | IoU 0.574 / F1 0.730 / P0.745 / R0.714 @ep22 | log: `runs/craq-sam2prompt-mask-2026-06-10/`（僅 log+manifest） |
| SAM2 points-prompt | IoU 0.519 / F1 0.684 @ep10 | log: `runs/craq-sam2prompt-points-2026-06-10/` |
| Tversky β0.7 | R0.800/P0.628/IoU0.542 @thr0.15（margin 較薄，被 β0.8 取代） | log: `runs/craq-sam2prompt-tversky37-2026-06-10/` |
| 全資料最終 dense-seg 模型 | 無 val（全資料 50ep） | `runs/expert_craq_v3_final_small/`（最終權重，保留） |

### 負結果（已驗證無效，僅留結論）
- **test-time CLAHE**：ResUNet 訓練有 CLAHE 增強但推論沒套；補上後角落不增反減 → 無效。
- **mask prompt 128→256**：recall 0.714→0.710 沒回來；dense prompt embedding 恆 32×32（凍結 image encoder 特徵網格），prompt 細節進 decoder 前已被池化。log: `runs/craq-sam2prompt-mask256-2026-06-10/`。

## Legacy 探索家族 A–E（runs 已刪 2026-06-10，留結論）
| 家族 | 模型 | best IoU 區間（非 fold2） | 最佳 fold |
|---|---|---|---|
| A dense-seg | `model_seg.py` | 0.43–0.49 | fold3 clahe 0.487 |
| B full-FPN | `model_seg_full_fpn.py` | 0.43–0.48 | fold3 0.483 |
| C learnable prompt | `model_prompt_seg.py` | 0.37–0.43 | fold0 0.434（recall 高 precision 低） |
| D prompted frozen SAM2 | `model_prompted_sam2.py` | 0.43–0.46 | fold1 0.457 |
| E decoder-only | `model_decoder_seg.py` | 0.38–0.43（mean 0.398 排除 fold2） | fold0 0.432 |

## 小結
- 四家族 best 都落在 IoU 0.43-0.49，差異不大；dense-seg / full-FPN 略穩。**全部被 refine(0.574) 取代。**
- **fold2 普遍退化**是跨家族一致現象 → 若重做優先查 fold2 切分。
- prompt 系列 final 明顯掉（過擬合）→ 取 best.pt。
- 其他舊產出（eval/overlay/viz 目錄、pre_label 之外的散落 log）已於 2026-06-10 清理；`pre_label_v3*`（CVAT 預標）保留。

## 2026-06-22 — 0-94 canonical GT 重訓鏈(ResUNet→prob→refine)
| 日期 | run | 配方 | best craq IoU / F1 / P / R | 結論 | 路徑 |
|---|---|---|---|---|---|
| 2026-06-22 | craq-refine-tversky28-aug-0-94gt | mask-prompt + Tversky β0.8 + aug, fold0, 讀 0-94 GT 新 prob | **0.5982 / 0.749 / 0.681 / 0.831** @ep17 | vs 舊 stale-GT aug 0.5925 → IoU +0.006 且 **precision 明顯升**(0.681);換 canonical 0-94 GT 未退化,符合「修正 GT 移除誤標 FP→precision 回升」。**洩漏註**:all-data ResUNet prob 對 fold0 val 洩漏,數字偏樂觀(與舊同條件可比) | `runs/craq-refine-tversky28-aug-0-94gt-2026-06-22/` |
| 2026-06-22 | craq-refine-tversky28-aug-demoholdout-0-94gt | 同配方,**leak-free demo-holdout**(held-out L-1RB1-1/A4-8/R-A4-3),讀 leak-free prob | **0.2225 / 0.364 / 0.673 / 0.249** @ep29 | **誠實泛化**:refine 對 ResUNet 0.222 **零增益**(prompt-bounded:held-out recall 0.25、漏處 prob≈0 無從修);遠低於 in-dist fold0 0.598 → in-dist 確偏樂觀。跌 vs 舊 0.515 非退化(0-94 GT 更密、held-out 70% 是沒看過的 A4-8 dense batch4)。**瓶頸=標註覆蓋** | `runs/craq-refine-tversky28-aug-demoholdout-0-94gt-2026-06-22/` |

- 三段鏈:ResUNet all-data(0-94 GT)`crack_detection_unet/runs/craq-resunet50-alldata-0-94gt-2026-06-22` → prob 711 `_data/craq_512_dataset_711_0-94/resunet_prob` → 此 refine。資料 = 乾淨 711(530 base + 181 batch4 R1,**排除半成品 R2/R1_C04**)。
- **誠實泛化(demo-holdout,leak-free)**:ResUNet held-out IoU 0.222 / refine 0.2225(零增益)。in-dist fold0 0.598 偏樂觀。ResUNet run `craq-resunet50-demoholdout-0-94gt-2026-06-22`、prob `resunet_prob_demoholdout`。

## 2026-06-24 — multiclass_512_dataset(917) 全資料 craquelure 鏈 + crack expert
| 日期 | run (slug) | 目的 | best craq IoU/F1/P/R @ep | 結論 | run 路徑 |
|---|---|---|---|---|---|
| 2026-06-24 | craq-refine-tversky28-aug-multiclass917-alldata-bestpt | 部署贏家 refine 配方(mask-prompt + Tversky β0.8 + aug)在 multiclass_512(917)二值 craq 全資料重訓,**全資料 best.pt(val=train)** | **0.5930 / 0.7445 / 0.6696 / 0.8382** @ep43 | refine 勝 stage-1 ResUNet(IoU 0.5365→0.5930,**precision 0.569→0.670** 回升,R 保 0.84);與 0-94 GT 版 0.5982 同量級。val==train 偏樂觀(無 holdout) | `runs/craq-refine-tversky28-aug-multiclass917-alldata-bestpt-2026-06-24/` |
- 三段鏈:ResUNet all-data(multiclass 917 衍生二值)`crack_detection_unet/runs/craq-resunet50-multiclass917-alldata-bestpt-2026-06-24`(craq IoU 0.5365)→ prob 917 `_data/multiclass_512_craq_bin/resunet_prob` → 此 refine。資料由 `scripts/build_binary_from_multiclass.py` 從 `_data/multiclass_512_dataset` 衍生(craq idx4→1)。
- 同批另訓 **crack** expert ResUNet(crack IoU 0.2216,見 `crack_detection_unet/EXPERIMENTS.md`)。
- 5 張 demo 推論 + 彩色 overlay 對照圖:`runs/predict-demo5-multiclass917-2026-06-24/`(craq_resunet→craq_refine@thr0.25 / crack_resunet;`comparison/*.png`)。

| 2026-06-29 | craq-modelcompare-fold0 | craquelure fold0 honest val 四模型彙整(DeepLabV3+/SegFormer vs ResUNet/SAM2-refine) | ResUNet 0.5335 / DeepLabV3+ 0.4443 / SegFormer 0.4070 (craq IoU);SAM2-refine 0.5982 引用(資料版不同 caveat) | 主流通用模型(DeepLabV3+/SegFormer)**未勝** 專案 ResUNet,差距在 precision(content-FP);訓練 run 在 `crack_detection_unet/runs/{unet,deeplabv3plus,segformer}-craqbin1027-fold0-2026-06-29` | `runs/craq-modelcompare-fold0-2026-06-29/` |

結論:`_research/decisions/2026-06-29-craquelure-model-comparison.md`。新 venv `segbaseline_env`(torch cu128 + smp 0.5.0);model factory 一般化在 `crack_detection_unet/src/unet_model.py` build_model + train.py `--arch`。

| 2026-06-29 | craq-refine-tversky28-aug-craqbin1027-fold0 | SAM2-refine **同條件**重跑(craq_bin1027 fold0,leak-free fold0-ResUNet prob),修掉舊引用 0.598 的資料版+洩漏問題 | IoU 0.5500 / P 0.6736 / R 0.7499 / F1 0.7097 @ep28 (thr0.5) | 四模型同條件下 IoU 最高:refine 0.550 > ResUNet 0.534 ≫ DeepLabV3+ 0.444 > SegFormer 0.407;refine 靠 precision、recall 反低(修剪器),增益僅 +0.016。誠實值 0.550<舊洩漏 0.598 | `runs/craq-refine-tversky28-aug-craqbin1027-fold0-2026-06-29/` |
