# Plan: craquelure 分割 4-model 比較

- 日期: 2026-06-29
- 分支: `exp/crack-craq-dualexpert-refine`
- spec: `crack_detection_sam2/docs/superpowers/specs/2026-06-29-craquelure-seg-model-comparison-design.md`
- 目標: DeepLabV3+ / SegFormer vs ResUNet(重跑 anchor) / SAM2-refine(引用),craquelure fold0 honest val 的 P/R/IoU/F1。
- 約束: **訓練嚴格序列化**(8GB GPU + WSL,一次只跑一個模型;勿與重 IO/pip 下載並行)。

## 步驟與檢查點(檢查點=可觀察的預期結果)

### S1 建 venv `segbaseline_env`
- 指令: `python3 -m venv segbaseline_env` → pip install torch+torchvision (cu128 index) + smp + timm + albumentations + opencv-python-headless + scikit-image + scipy + `-e _lib`。
- **CP1**: `segbaseline_env/bin/python` →`torch.cuda.is_available()==True`、`get_device_capability()==(12,0)`、`smp.__version__` 有、`hasattr(smp,'DeepLabV3Plus')` 且 `hasattr(smp,'Segformer')` 皆 True。→ 符合/不符合。

### S2 一般化 model factory + `--arch`
- 改 `crack_detection_unet/src/unet_model.py`: `build_resunet` → `build_model(arch, encoder, encoder_weights, num_classes, in_channels)`,arch∈{unet,deeplabv3plus,segformer};保留 `build_resunet` 薄包裝(回溯相容)。`train.py` 加 `--arch`(預設 unet)。
- **CP2**: smoke `build_model` 三種 arch,輸入 (2,3,512,512) → 輸出 (2,2,512,512) 無誤;`param_groups` 對三者都抓到 encoder/decoder 兩組。→ 符合/不符合。

### S3 重建 craq_bin 到 1027
- 指令: `python crack_detection_sam2/scripts/build_binary_from_multiclass.py --src _data/multiclass_512_dataset --dest _data/multiclass_512_craq_bin --target_id 4 --target_name craquelure`(會覆寫 masks、複製 group_split_stem.json）。
- **CP3**: 輸出 `tiles=1027`、`positive_tiles>0`;`group_split_stem.json` fold0 train/val 數印出、val stem 清單記錄。→ 符合/不符合。

### S4 訓 ResUNet anchor (fold0) — 序列化
- run: `crack_detection_unet/runs/unet-craqbin1027-fold0-2026-06-29/`,配方 = resnet50 / img512 / bs8 / 80ep / lr3e-4 / enc_lr_mult0.1 / wd1e-4 / ce0.5+dice0.5 / median_freq / seed42 / split=craq_bin group_split_stem fold0。
- **CP4**: 訓練完成出 best.pt + fold0 craquelure {P,R,IoU,F1};IoU 落在合理區(歷史同類 ~0.5x,新 split 可能不同,記錄實際)。→ 符合/不符合/待查。

### S5 訓 DeepLabV3+ (fold0) — 序列化(S4 完才開)
- run: `...runs/deeplabv3plus-craqbin1027-fold0-2026-06-29/`,`--arch deeplabv3plus` 其餘同 S4。
- **CP5**: 訓練完成、出 fold0 craquelure 四指標。→ 符合/不符合/待查。

### S6 訓 SegFormer (fold0) — 序列化(S5 完才開)
- run: `...runs/segformer-craqbin1027-fold0-2026-06-29/`,`--arch segformer --encoder mit_b0`(8GB,OOM 則降 bs);其餘同 S4。
- **CP6**: 訓練完成、出 fold0 craquelure 四指標(若降 bs/size 須在 manifest 註記)。→ 符合/不符合/待查。

### S7 彙整比較 + 結論
- 收 4 模型(含 SAM2-refine 引用 `craq-refine-tversky28-aug-0-94gt-2026-06-22` fold0 IoU 0.598/P0.681/R0.831,**註資料版/split 差異**)→ `crack_detection_sam2/runs/craq-modelcompare-fold0-2026-06-29/comparison.{md,json}`,列 P/R/IoU/F1/params/stage/資料版註。
- **CP7**: 比較表四列齊全;結論明說 DeepLabV3+/SegFormer **勝過/持平/輸給** ResUNet 與 refine,並描述 P/R 取捨型態。→ 符合/不符合/待查。
- 寫 `_research/decisions/2026-06-29-craquelure-model-comparison.md`;`EXPERIMENTS.md` 各補列。

## 已知陷阱
- SegFormer params 遠少於 resnet50 → 表列 params,結論點出容量差。
- SAM2-refine 引用值資料版不同 → 表內 caveat,非同條件直接比。
- 8GB OOM → 先 bs 小、必要 grad accum;cudaErrorUnknown → 大 IO 勿與 GPU 並行、單張重跑。
- fold0 val 群偏 KJTHT 素面 → 報告附 fold0 stem 清單。
