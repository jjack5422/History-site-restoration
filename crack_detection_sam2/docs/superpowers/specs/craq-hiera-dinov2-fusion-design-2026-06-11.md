# Craquelure Expert: SAM2 Refine + DINOv2 語意特徵注入 — 設計 spec

日期: 2026-06-11
分支: feature/cvat-craq-crack-agent
專案: crack_detection_sam2 (craquelure expert)

## 1. 背景與動機

craquelure expert 現役為兩階段 pipeline:

- Stage 1: ResUNet (crack_detection_unet) 離線輸出 craquelure 機率圖 (`predict_full.py --save_prob`, 存 `.npy`, channel[1]=craq)。
- Stage 2: `PromptedSAM2Seg` (`model_prompted_sam2.py`) — SAM2 **凍結 image encoder** + 可訓練 prompt encoder + mask decoder;ResUNet 機率經 logit 化後當 dense mask prompt 餵 prompt encoder → 精修出二值 craq mask。
- Loss `0.5*BCE + 0.5*(1 - Tversky β0.8)` @thr0.25;recommended run `craq-sam2prompt-tversky28`:**R0.80 / P0.637 / IoU0.550**。

核心問題:**precision 僅 0.637**,預測為 craq 的像素約 36% 為 false positive。假設這些 FP 多半來自彩繪/背景紋理 — 線性深色裝飾結構被當成 craquelure 龜裂網。SAM2 image encoder 全程凍結,從未在本資料上學過背景語意;DINOv2 的 dense 自監督特徵在「區域語意/材質身分」上比原始 Hiera 特徵更可分,適合補這個洞。

## 2. 目標與假設

單一可證偽假設:

> 在凍結 Hiera (SAM2 image encoder) 特徵之上,注入凍結 DINOv2 語意特徵到 mask decoder 看到的 image embedding,能在**守住 recall** 的前提下提升 craquelure 的 **precision / IoU**。

本 spec **不標註背景**(零標註成本),只驗證「加 DINOv2 特徵」這一件事本身有沒有用。背景拆語意子類 (彩繪/素面) 的 supervised 版本留待後續 spec。

## 3. 架構

**不改 SAM2 大體架構。** 保留原生 SAM2 encoder–decoder + ResUNet→prompt 路徑;唯一新增是在 `feat` 進 decoder 前包一層 adapter。`FPNSegHead` (`model_seg.py`) 不採用(那是另一條測試版路線)。

資料流:

```
凍結 SAM2 image encoder → fpn[-1]  (feat, 512輸入=32×32×256, decoder cross-attn 的語意 embedding)
                        → high_res = [conv_s0(fpn[0]), conv_s1(fpn[1])]  (stride 4/8, decoder 上採樣細節路)

凍結 DINOv2 → 518 resize → 37×37 patch tokens → proj 256 → resize 32×32 ─┐
                                                                          ▼
                                    feat ──→ [FeatFusionAdapter] ──→ feat'
                                                                          │
ResUNet logit → prompt_encoder(masks=prev_mask) → (sparse, dense)         │
                                                                          ▼
       mask_decoder(image_embeddings=feat', high_res_features=high_res, ...) → low → 上採樣 → 1ch craq mask
```

注入點原則:

- DINOv2 是粗語意,對應 32×32 的 `feat`(decoder cross-attention 的語意瓶頸)。
- **只注入 `feat`;`high_res` 完全不動** — craquelure 細裂紋來自 high_res,DINOv2 太粗不該污染細節路。

### FeatFusionAdapter (唯一新增模組)

- 輸入:`feat` (B,256,32,32) + DINOv2 特徵 (resize 對齊到 32×32, proj 到 256)。
- 融合:預設 concat → 1×1 conv → 256ch;**殘差 + zero-init**(融合分支輸出初始化為 0,使 `feat' ≈ feat`,訓練起點等於現有 0.550 行為,只學 DINOv2 帶來的增量)。
- 輸出:`feat'` (B,256,32,32),形狀與 `feat` 完全相同,直接取代 `feat` 進 decoder。

### 凍結 / 可訓練

- 凍結:SAM2 image encoder、DINOv2 backbone。
- 可訓練:`FeatFusionAdapter`(新)、prompt encoder、mask decoder(與現狀一致)。
- 參數量增加極小(僅 adapter 的 proj + 1×1 conv)→ 小資料安全。

## 4. 資料與特徵 cache

- 資料集 `_data/craq_0-94_v1/tiles_512/`(images / masks / resunet_prob 已存在)。
- DINOv2 特徵離線 cache,比照現有 `resunet_prob`:存到 `tiles_512/dinov2_feat/`(每 tile 一個 `.npy`/`.pt`,存 37×37 或已 proj 前的 raw token map)。
- 訓練時 SAM2 照舊 live forward(frozen),DINOv2 只讀 cache → 訓練幾乎不增記憶體,避開 8GB OOM。
- 增強沿用現有 trainer(僅 hflip):cache 特徵的 hflip = 翻轉 token map 張量;不做 scale aug。

## 5. 實驗矩陣與決策 gate

| 代號 | 配置 | 問題 |
|---|---|---|
| **C0** (對照 = 現役 baseline) | 現有 refine,無 DINOv2 | 即 R0.80/P0.637/IoU0.550 pipeline,同 harness/同 split 重跑 |
| **E1** (主處理) | refine + FeatFusionAdapter (concat, zero-init) | DINOv2 注入的邊際貢獻 |
| **E2** (可選,E1 有效才做) | 融合變體:concat-conv vs FiLM gate | 融合方式比較 |

決策 gate:

- **Gate 0**(免訓練,decisive):
  1. 驗證 DINOv2 (ViT-S/14, registers) 能在 `sam2_env` 載入(torch.hub 或既有套件)— 不假設已裝。
  2. dump 現有 expert 在 val 的 FP map,目視確認 FP 是否真集中在彩繪/背景紋理。若否,假設前提弱化,照跑但於結果標註此風險。
- **Gate 1**:E1 vs C0,在 matched recall 下 precision / IoU 是否上升且 recall 未崩。

## 6. 評估與成功判準

- Loss / optimizer / threshold 全部沿用 `train_craq_promptrefine`(`0.5*BCE + 0.5*(1 - Tversky β0.8)` @thr0.25),僅多 adapter 的 param group;確保 C0 vs E1 公平。
- 指標:IoU / precision / recall + threshold sweep。
- **跨現有 craq CV folds 報 mean ± std**(memory 顯示 fold 變異極大、A4-4/fold2 為 OOD outlier,單一 split 是雜訊)。
- 成功 = E1 的 IoU 在 matched recall 下顯著高於 C0 且 precision 改善;理想上逼近/超越 0.550。
- 質化:E1 vs C0 的 FP map 對比,確認 DINOv2 確實壓掉彩繪 FP。

## 7. 不做 (YAGNI)

- 不標彩繪/背景子類、不加 background aux class(留後續 spec)。
- 不解凍任何 backbone。
- 不採用 FPNSegHead / 不改 SAM2 大體架構。
- 不碰 `high_res` 路。
- 不動其他 4 個 expert (crack/loss/shrinkage/flaking)。

## 8. 待驗證 / 開放項(交由實作 plan 釐清)

- DINOv2 變體最終選定 (S/14 vs B/14) 與在 sam2_env 的載入方式。
- craq_0-94 的確切 train/val split 協定(是否有 CV folds 檔)以對齊 `craq-sam2prompt-tversky28`。
- DINOv2 token map 的 cache 格式(raw token vs 已 reshape 2D)與檔案命名。
- FeatFusionAdapter 殘差 zero-init 的具體實作(最後一層 conv weight/bias 置 0)。
