# segformer-craqbin1027-fold0-2026-06-29

目的: craquelure 4-model 比較。smp.Segformer mit_b0,同 craq_bin 1027 fold0,同配方(loss/aug/optim/epoch),換 arch+encoder。
venv: segbaseline_env; 配方同 unet anchor;encoder=mit_b0 (segformer 自帶 MiT)。
與 unet run 差異: --arch segformer --encoder mit_b0 (params ~3.7M,容量遠小於 resnet50 系,比較時須點明)。
