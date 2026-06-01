# YOLO-prompted SAM2 decoder fine-tune（craquelure）設計文件

- 日期: 2026-05-31
- 程式落點: `/home/zzz90/research/crack_detection_sam2`（可微 SAM2 + craq 資料 + dense-seg baseline 都在此）
- 借用: SepSAM2 的 craquelure YOLO agent(`crack_detection_SepSAM2/.../sepsam_agent_craq_cv_fold{k}`)+ 中軸取點
- 環境: venv `/home/zzz90/research/sam2_env`

## 1. 背景與動機

實測:SepSAM2 的 SAM2 是**凍結 zero-shot**,對 crack 與 craquelure 的 refine 都無貢獻(CMC ≡ YOLO-only)。本實驗驗證「**把後面的 SAM2 decoder 訓練過**」能否讓 prompt 精修真的有用。先做 craquelure(agents/split/baseline 都現成)。

不可微的 CMC 整鏈無法端到端 backprop;但 SAM2 `image_encoder→prompt_encoder→mask_decoder` 是可微的,點座標當「輸入」即可訓 decoder(標準 SAM fine-tune)。`model_prompt_seg.py:SAM2PromptSeg` 是現成模板(差別:它用 learnable prompt,我們改吃外部點)。

## 2. 目標與成功標準

- 主:trained-SAM2 在 **YOLO-point 模式**的 4-fold 平均 craquelure F1(排除 fold2)是否 **> SepSAM YOLO-only 0.541** 且 > frozen-SAM2(證明訓練 SAM2 讓 refine 有用)。
- 診斷:**GT-point oracle** 模式是否接近/超過 dense-seg expert 0.634(decoder 學習上限);oracle 與 YOLO-point 的差 = YOLO recall 天花板的代價。

## 3. 架構

新模型 `PromptedSAM2Seg`(由 `model_prompt_seg.py:SAM2PromptSeg` 改寫):
- 沿用 `model.build_sam2_model` 取 `image_encoder` / `sam_prompt_encoder` / `sam_mask_decoder`,並 override 512 輸入的 embed/pos 尺寸(同 SAM2PromptSeg 既有邏輯)。
- **prompt 改為 forward 的外部輸入**:`forward(x, point_coords, point_labels)`,不再有 `nn.Parameter` 的 learnable 點。
- `freeze_image_encoder=True`(凍結),`prompt_encoder` + `mask_decoder` 可訓。輸出 `[B,1,H,W]` binary logits,上採樣回輸入解析度。

## 4. 資料流程

- 資料: `data/labeled32_craq_v3/tiles_512`(images/masks 0/1)+ `group_split_stem.json`(4-fold,與 dense-seg / SepSAM 同 split,fold 對齊)。
- 重用 `dataset.TileSegDataset`(回傳 image tensor + mask)。
- **GT 中軸取點(訓練 + oracle eval)**:對每張 tile 的 GT craquelure mask 取 medial axis,沿骨架均勻取 N 個正點(label=1)當 prompt。用 `skimage.morphology.medial_axis`(與 SepSAM2 `geometry.mask_to_points_and_width` 同法;直接在本專案實作一個 `gt_points()` helper,避免跨專案 import)。N 用 `max(H,W)//POINTS_DIVISOR`(POINTS_DIVISOR=50,同 SepSAM)。
- **空-mask tile 處理(明確)**:prompt-conditioned 分割需要點才能分割。**訓練只用前景 tile**(GT craquelure 像素 ≥1 的 tile;空 tile 略過)。**eval**:若該 tile 取不到點(GT 空 / YOLO 沒抓到)→ 直接輸出全空 mask(不呼叫 SAM2),再與 GT 算 prf_iou(GT 也空時 F1 計為 1.0 由 `prf_iou` 邏輯處理)。
- **YOLO 取點(YOLO-point eval)**:用 SepSAM2 的 fold-k craquelure agent 推論該 tile → 二值 mask → 同樣中軸取點。

## 5. 訓練

- Loss: BCE+Dice(沿用既有 binary loss,如 `train_prompt.py:BinaryCEDiceLoss`),pos_weight 由 craquelure 像素頻率推得。
- Optimizer: AdamW,decoder/prompt_encoder lr=base_lr,encoder 凍結;cosine+warmup;AMP;grad clip 5.0(對齊 train_prompt.py)。
- imgsz=512、batch 視顯存(4-8)、epochs ~80、4-fold(同 split)。
- 輸出 `outputs/promptsam2_craq_fold{k}/best.pt`(以 val IoU 選 best;val 用 GT-point oracle 模式評,保持訓練/選模一致)。

## 6. 評估

逐 fold,兩種推論模式各算 P/R/F1/IoU(像素級,與 dense-seg 同口徑):
1. **oracle**:GT 中軸點 → trained decoder。
2. **YOLO-point**:fold-k YOLO craq agent 草稿中軸點 → trained decoder。
彙整 4-fold 平均 + per-fold,fold2 單列 + 排除-fold2 平均。

對照表(排除 fold2 平均):dense-seg+CLAHE **0.634**、SepSAM YOLO-only **0.541**、frozen-SAM2 SAM-only ~0.10。

## 7. 元件（檔案，皆在 crack_detection_sam2）
- `model_prompted_sam2.py`(新):`PromptedSAM2Seg`(外部點 prompt 版)。
- `train_promptsam2_craq.py`(新):4-fold 訓練(GT 點、BCE+Dice）。
- `gt_points.py` 或併入 train 腳本:`gt_points(mask_bin, n)` 中軸取點 helper。
- `eval_promptsam2_craq.py`(新):oracle + YOLO-point 雙模式 4-fold eval + 對照表。
- 重用: `model.py`、`dataset.py`、`augment.py`、`train_prompt.py` 的 loss。

## 8. 風險與注意
- fold2(彩繪面板)預期照樣崩 → 看排除-fold2。
- encoder 凍結 + 512(SAM2 預訓 1024)→ pos embed 內插,品質可能略損;先 512,不夠再議 1024。
- 空-mask tile 的 prompt 處理需明確定義(無正點時的行為)。
- YOLO-point 模式仍受 YOLO recall 天花板;預期 oracle 高、YOLO-point 受限——這正是要量化的。
- 跨專案:YOLO agent 在 SepSAM2,需用其 venv 能載入(C2f_SEA);eval 的 YOLO-point 模式以 SepSAM2 env 跑 agent 產點,或預先把各 fold val 的 YOLO 點快取成檔案再餵 eval(避免跨 env 同進程)。實作時採「預先快取 YOLO 點」較乾淨。

## 9. 範圍外（YAGNI）
- 不訓 image encoder(先凍結;不夠再 LoRA)。
- 不做 crack(另案)。
- 不改 SepSAM2 的 CMC 決策邏輯;本實驗只測「trained SAM2 decoder + 點」的分割品質。
- 不做 box prompt / 負點(先用既有中軸正點看 baseline)。
