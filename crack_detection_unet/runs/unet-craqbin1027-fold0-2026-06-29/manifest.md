# unet-craqbin1027-fold0-2026-06-29 (ResUNet anchor)

目的: craquelure 4-model 比較的 anchor。smp.Unet resnet50,現行 1027-tile craq_bin,fold0 honest val。
plan: _research/plans/2026-06-29-craquelure-seg-model-comparison.md
venv: segbaseline_env (torch 2.11.0+cu128, smp 0.5.0)
code: crack_detection_unet/src/{train.py,unet_model.py} (build_model+--arch), commit 後補

完整可重跑指令:
  PYTHONPATH=crack_detection_unet/src NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 \
  segbaseline_env/bin/python crack_detection_unet/src/train.py \
    --arch unet --encoder resnet50 \
    --tiles_root /home/zzz90/research/_data/multiclass_512_craq_bin \
    --split /home/zzz90/research/_data/multiclass_512_craq_bin/group_split_stem.json --fold 0 \
    --image_size 512 --batch_size 8 --epochs 80 \
    --base_lr 3e-4 --encoder_lr_mult 0.1 --weight_decay 1e-4 \
    --ce_weight 0.5 --dice_weight 0.5 --class_weight_mode median_freq --seed 42 \
    --class_names background,craquelure \
    --output_dir crack_detection_unet/runs/unet-craqbin1027-fold0-2026-06-29

資料: _data/multiclass_512_craq_bin (1027 tile, 740 craq+; target_id=4 from multiclass_512_dataset)
      fold0 train=812 val=215; val_groups=[KJTHT-SC-L-A4-4, KJTHT-SC-M-2LB1-2, KJTHT-SC-R-A4-3, MGLST-DT-1L-A2-1]
與上次差異: 資料 917->1027 (canonical multiclass_512_dataset 派生); 新 venv; train.py 加 --arch (此為 unet 預設行為)
