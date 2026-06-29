# craq-refine-tversky28-aug-craqbin1027-fold0-2026-06-29 (SAM2-refine 同條件)

目的: 讓 SAM2-refine 在與 ResUNet/DeepLabV3+/SegFormer **同條件**(craq_bin 1027, fold0 honest val,
leak-free prompt)下可直接比。修掉舊引用值(0-94 資料版 + all-data prob 洩漏)的兩個非同條件問題。
venv: sam2_env (torch 2.11.0+cu128)
配方: 沿用部署贏家 tversky28-aug (mask-prompt, variant small, tversky α0.2/β0.8, aug, 60ep, bs4, lr2e-4, fp_weight 3.0 純量)
prompt: leak-free fold0 ResUNet prob = crack_detection_unet/runs/unet-craqbin1027-fold0-2026-06-29/best.pt
        dump -> _data/multiclass_512_craq_bin/resunet_prob_fold0/prob (fold0 val prob 未洩漏)
eval: train_craq_promptrefine evaluate() pred=logits>0 (sigmoid 0.5) → 與其他三模型 argmax(0.5) 同門檻可比
完整可重跑指令:
  PYTHONPATH=crack_detection_sam2 NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 \
  sam2_env/bin/python crack_detection_sam2/train_craq_promptrefine.py \
    --tiles_root /home/zzz90/research/_data/multiclass_512_craq_bin \
    --split /home/zzz90/research/_data/multiclass_512_craq_bin/group_split_stem.json --fold 0 \
    --prob_dir /home/zzz90/research/_data/multiclass_512_craq_bin/resunet_prob_fold0 \
    --prompt_mode mask --variant small --image_size 512 \
    --tversky_alpha 0.2 --tversky_beta 0.8 --epochs 60 --batch_size 4 --base_lr 2e-4 \
    --fp_weight 3.0 --aug \
    --output_dir /home/zzz90/research/crack_detection_sam2/runs/craq-refine-tversky28-aug-craqbin1027-fold0-2026-06-29
與引用值差異: 資料 craq_0-94_v1 -> craq_bin1027; prob all-data(洩漏)-> fold0(leak-free); split 改現行 fold0
