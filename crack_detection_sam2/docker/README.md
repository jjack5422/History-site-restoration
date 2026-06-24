# craquelure 全流程 - 5090 (Blackwell) Docker 環境

在實驗室 RTX 5090 (Blackwell, sm_120) Linux 機器上重建環境,跑完整
**ResUNet → prompt 機率圖 → SAM2 refine** 流程。

兩個階段環境的 torch 版本不同,所以是**兩個獨立 image**(不能合併):

| image | 用途 | 來源 env | torch |
|---|---|---|---|
| `craq-unet:cu128` | 階段1:ResUNet 產 prompt 機率圖 | unet_env | `2.12.0.dev20260407+cu128` (nightly) |
| `craq-sam2:cu128` | 階段2:SAM2 refine 訓練 | sam2_env | `2.11.0+cu128` (stable) |

兩者都是 CUDA 12.8 / Python 3.12 / Blackwell 相容,已凍結成可重現的 image。

Host 前提(實驗室機器已滿足):NVIDIA driver >= 570、`nvidia-container-toolkit`。

---

## 0. 關鍵概念:容器內路徑固定為 `/home/zzz90/research`

訓練腳本 `scripts/run_craq_fused.sh` 與 checkpoint symlink 都寫死絕對路徑
`/home/zzz90/research/...`。**host 放哪裡都行**,只要 `docker run` 時把它掛到容器內的
`/home/zzz90/research` 即可。以下假設 host 路徑為 `$RESEARCH`(例如 `/data/research`)。

---

## 1. 把檔案搬到實驗室機器(目前 branch 沒有 remote)

需要搬的:**code(含 git 歷史)+ 兩個 vendored editable 套件 + 資料 + 權重**。
**不要搬** `*_env/`(venv 不可攜,容器會重建)與 `runs/`(7.3G 舊輸出,重訓會重生)。

最省事:整棵 rsync,排除 venv 與舊 runs(`.git` 一併帶過去,歷史就有了):

```bash
# 在本機執行,推到實驗室機器(改成你的帳號/主機/目標路徑)
RESEARCH=/data/research        # 實驗室機器上的落地路徑
rsync -avhP \
  --exclude='*_env/' \
  --exclude='crack_detection_sam2/runs/' \
  /home/zzz90/research/  LABUSER@LAB_HOST:$RESEARCH/
```

搬過去後,實驗室機器上要有這些(全流程 B):

| 路徑 | 大小 | 階段 | 用途 |
|---|---|---|---|
| `crack_detection_unet/src/` | - | 1 | ResUNet predict/train code |
| `crack_detection_unet/runs/craq-resunet50-2026-06-10/best.pt` | - | 1 | craq ResUNet 權重 |
| `crack_detection_sam2/` | - | 2 | SAM2 訓練/評估 code |
| `segment-anything-2/` | - | 2 | SAM2 套件原始碼(editable) |
| `segment-anything-2/checkpoints/` | 1.5G | 2 | `sam2.1_hiera_*.pt` 預訓練權重 |
| `_lib/` | 120K | 1+2 | `crackseg_common` 共用套件(editable) |
| `_data/craq_0-94_v1/tiles_512/` | 1.9G | 1+2 | tiles + `resunet_prob` + `dinov2_feat` + `group_split_stem.json` |
| `_data/0-94/` | 13M | - | canonical 多類別標籤 |

> `transfer.sh` 預設不排除 `crack_detection_unet/runs/`(~6G,含上面那顆權重),所以
> 全流程需要的東西會一起過去。`resunet_prob/` 若要重算才需要階段1;沿用現成的可跳過。

---

## 2. 建 image

build context 必須是 research root(這樣 `segment-anything-2/` 與 `_lib/` 看得到)。
全流程要 build 兩個 image:

```bash
cd $RESEARCH
# 階段1:ResUNet 產 prompt 機率圖
docker build -f crack_detection_sam2/docker/Dockerfile.unet -t craq-unet:cu128 .
# 階段2:SAM2 refine 訓練
docker build -f crack_detection_sam2/docker/Dockerfile      -t craq-sam2:cu128 .
```

> 若 build 時 `nvidia/cuda:12.8.0-runtime-ubuntu24.04` 這個 tag 拉不到,
> 到 hub.docker.com/r/nvidia/cuda/tags 找最接近的 `12.8.x-runtime-ubuntu24.04` 換上。
> torch wheel 自帶 CUDA/cuDNN,base image 只需相符的 CUDA runtime。
>
> 階段1 用 **nightly** torch,wheel 會被定期清除。若 `2.12.0.dev20260407` 抓不到,
> 見 `Dockerfile.unet` 內註解改用最新 nightly。

---

## 3. 階段1:用 ResUNet 重算 prompt 機率圖

只有要換資料/fold/重訓 ResUNet 時才需要這步;沿用現成 `resunet_prob/` 可直接跳到階段2。

```bash
cd $RESEARCH
docker run --gpus all -it --rm \
  --shm-size=16g \
  -v "$PWD":/home/zzz90/research \
  craq-unet:cu128 \
  python src/predict_full.py \
    --ckpt    runs/craq-resunet50-2026-06-10/best.pt \
    --image_dir /home/zzz90/research/_data/craq_0-94_v1/tiles_512/images \
    --out_dir   /home/zzz90/research/_data/craq_0-94_v1/tiles_512/resunet_prob \
    --tile 512 --stride 384 --save_prob
```

產出寫到 `resunet_prob/prob/<stem>.npy`(每個 tile 一個),正是階段2 `--prob_dir` 讀的東西。

> `--tile/--stride/--clahe` 要跟當初產生現成 `resunet_prob` 的設定一致,否則 prompt 分布會變。
> 不確定就先重算一張、跟現成的 `.npy` 比對 shape/數值再全跑。

---

## 4. 階段2:跑 SAM2 容器(掛到固定路徑)

```bash
cd $RESEARCH
docker run --gpus all -it --rm \
  --shm-size=16g \
  -v "$PWD":/home/zzz90/research \
  craq-sam2:cu128 bash
```

進容器後 entrypoint 會印出 GPU 檢查,確認看到:

```
compute_capability=12.0
```

手動快驗:

```bash
python -c "import torch,sam2,crackseg_common; print(torch.cuda.get_device_capability())"
# 期望 (12, 0)
```

---

## 5. 階段2:重訓 SAM2 refine(5-fold,C0 baseline + E1 DINOv2-fused)

```bash
cd /home/zzz90/research/crack_detection_sam2
bash scripts/run_craq_fused.sh
```

或單跑一個 fold 看流程通不通(目前主線權重設定:recall 加權 Tversky β0.8):

```bash
python train_craq_promptrefine.py \
  --tiles_root /home/zzz90/research/_data/craq_0-94_v1/tiles_512 \
  --split      /home/zzz90/research/_data/craq_0-94_v1/tiles_512/group_split_stem.json \
  --prob_dir   /home/zzz90/research/_data/craq_0-94_v1/tiles_512/resunet_prob \
  --prompt_mode mask --tversky_alpha 0.2 --tversky_beta 0.8 \
  --epochs 60 --batch_size 4 --base_lr 2e-4 --fold 0 \
  --output_dir runs/craq-base-c0-fold0-smoketest
```

5090 24GB 顯存遠大於本機 8GB,batch_size 可往上調(8/16)再看吞吐與顯存。

---

## 疑難

- `no kernel image is available for execution`:torch 不是 Blackwell build。確認 image 內
  `torch.__version__` 帶 `+cu128` 且 `get_device_capability()` 回 `(12, 0)`。
- `RuntimeError: CUDA not available`:`docker run` 漏了 `--gpus all`,或 host driver < 570。
- DataLoader 卡死 / bus error:`--shm-size` 調大(已給 16g),或 `--num_workers` 調小。
- checkpoint symlink 找不到:確認 host 上 `segment-anything-2/checkpoints/*.pt` 實體存在,
  且容器掛載點是 `/home/zzz90/research`(symlink 是絕對路徑)。
