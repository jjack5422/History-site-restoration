# Run: 2026-06-09-kjht42-prelabel-ra4

對 `KJTHT-SC-R-A4-3` 面板的 16 張未標 tile 做 craquelure + crack 預標,輸出 CVAT 可匯入的
「Segmentation mask 1.1」zip 供人工修正。craq/crack 兩個二元 expert 重新在使用者的「前 42 張
KJTHT 標註」上訓練(= `selected_slices/batch_1` 排序前 42 == `0-41test` default.txt 前 42)。

## 資料與環境 (Data & Env)
- 訓練標註: `_data/0-41test/SegmentationClass`(VOC palette,6 類);取 default.txt 前 42 stems
  (41 非空 + 1 空 `KJTHT-SC-L-A4-4_R2_C05`;含 2 張已標 R-A4 `R2_C03/C04`)。
- 訓練 RGB: `_data/image_1024_slices/<stem>.jpg`(1024 tiles)。
- 預標目標: `_data/_kjht42_ra4_unlabeled/`(16 張 = R-A4-3 全 24 tile 減 8 已標):
  R1_C01..C08, R2_C01/C02/C07/C08, R3_C01/C02/C07/C08。
- 二元資料集(新, 不蓋 labeled32): `_data/kjht42_craq`(305 tiles, 293 fg)、
  `_data/kjht42_crack`(215 tiles, 185 fg);tile 512 / stride 256;全量 split
  `nofold_all_train.json`(train==val,無 held-out)。
- env: craq `/home/zzz90/research/sam2_env`、crack `/home/zzz90/research/unet_env`;GPU RTX 5060。
- 程式 SHA: ddbf431。

## 指令 (Commands)
```bash
# A. build datasets (sam2_env)
python scripts/build_binary_datasets.py --seg_dir _data/_kjht42_seg \
  --image_dir _data/image_1024_slices --out_root_template _data/kjht42_{class} \
  --classes crack craquelure --tile_size 512 --stride 256 --seed 42
python scripts/make_nofold_split.py --tiles_root _data/kjht42_craq/tiles_512
python scripts/make_nofold_split.py --tiles_root _data/kjht42_crack/tiles_512

# B. train (50 epochs, all-42)
sam2_env  train.py     --tiles_root _data/kjht42_craq/tiles_512  --split .../nofold_all_train.json \
          --variant small --epochs 50 --class_names background,craquelure --output_dir runs/.../craq
unet_env  src/train.py --tiles_root _data/kjht42_crack/tiles_512 --split .../nofold_all_train.json \
          --encoder resnet50 --epochs 50 --class_names background,crack --output_dir runs/.../crack

# C. prelabel 16 R-A4 (two-pass) + merge
sam2_env  predict_full.py     --ckpt runs/.../craq/last.pt  --image_dir _data/_kjht42_ra4_unlabeled \
          --out_dir craq_raw  --tile 512 --stride 256 --save_prob
unet_env  src/predict_full.py --ckpt runs/.../crack/last.pt --image_dir _data/_kjht42_ra4_unlabeled \
          --out_dir crack_raw --tile 512 --stride 256 --save_prob
sam2_env  scripts/merge_pre_label.py --craq_prob_dir craq_raw/prob --crack_prob_dir crack_raw/prob \
          --image_dir _data/_kjht42_ra4_unlabeled --out_dir merged \
          --craq_thresh 0.5 --crack_thresh 0.5 --priority craq_over_crack

# D. package
sam2_env  scripts/package_cvat_segmask.py --voc_dir merged/voc_palette \
          --labelmap _data/0-41test/labelmap.txt --out_dir cvat_import --zip
```

## 結果 (Results)
- craquelure expert (SAM2 Hiera-small dense, `model.SAM2SemSeg`, trainable 0.91M):
  train(==val) IoU=0.41 (best 0.4273), recall 0.87 / precision 0.44 → 高召回,偏過度預測,適合預標。
  ckpt `runs/2026-06-09-kjht42-experts/craq/last.pt`(本地, .pt gitignored)。
- crack expert (ResUNet resnet50): train(==val) IoU=0.10 (best 0.1040), recall 0.95 / precision 0.10
  → **很弱**,大量假陽性(crack 在此 craquelure 為主的 42 張中極稀疏)。需重度人工修。
  ckpt `crack_detection_unet/runs/2026-06-09-kjht42-experts/crack/last.pt`(本地)。
- 預標覆蓋(merge thresh 0.5/0.5,craq_over_crack):16/16 tile 皆有 craq 與 crack;
  craq ~1.5-7.4%/tile,crack ~1.4-5.3%/tile(crack 多為假陽性)。
- 交付: `cvat_import.zip`(16 SegmentationClass + SegmentationObject + labelmap + default.txt)。
  CVAT「Import annotations → Segmentation mask 1.1」匯入後人工修。

## 注意 (Notes)
- 無 held-out 評估(全量訓練, val==train),上述 IoU 為訓練擬合, 會高估真實表現;這是預標加速器,
  使用者逐張審。若要誠實品質數字需另跑 fold 評估。
- crack 太吵時可只重跑 merge + package 並調高 `--crack_thresh`(例如 0.8),數秒完成,不需重訓。
- 中間檔 `craq_raw/` `crack_raw/`(prob .npy)較大,已 gitignore。
