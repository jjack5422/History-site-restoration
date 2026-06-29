# deeplabv3plus-craqbin1027-fold0-2026-06-29

目的: craquelure 4-model 比較。smp.DeepLabV3Plus resnet50,同 craq_bin 1027 fold0,同配方,只換 arch。
venv: segbaseline_env; 配方同 unet anchor (img512/bs8/80ep/lr3e-4/ce0.5+dice0.5/median_freq/seed42)
與 unet run 差異: 僅 --arch deeplabv3plus (encoder 同 resnet50,隔離架構變因)
完整指令見下方 train.log 對應命令。
