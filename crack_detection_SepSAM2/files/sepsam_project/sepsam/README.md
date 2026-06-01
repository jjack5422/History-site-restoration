# SepSAM 復現專案（交給 Claude Code 建構/補完）

復現論文：Zhou et al., *"Self-evolving prompting segment anything model for crack segmentation through data-driven cyclic conversations"*, **Advanced Engineering Informatics 68 (2025) 103626**, DOI: 10.1016/j.aei.2025.103626。

> **給 Claude Code 的話**：本 repo 已備好骨架與核心程式。請依本 README 與 `SPEC_復現規格書.md` 把專案跑起來並補完缺口。
> - **詳細規格、演算法、超參數、注意事項全在 `SPEC_復現規格書.md`**（含附錄 C：SEA 整合；附錄 D：SAM2 變體）。
> - 標 `[需推斷]` 之處請保留為可調參數、勿當論文事實。
> - 原論文未公開官方程式碼，部分實作為依論文敘述之合理重建。

---

## 這是什麼
SepSAM = 用一個**輕量 YOLOv8-Seg + SEA**（唯一需要訓練的 Agent）自動產生提示，引導一個**凍結的 SAM / SAM2**（大模型）做裂紋分割；兩者透過 **CMC（Cyclic Model Conversation）四輪流程**互相校正：沿裂紋中軸取點提示 → SAM 精修 → 用 Agent 信心過濾雜訊 → 衝突分析決定是否採用。SAM 全程不訓練。

## 專案結構
```
sepsam/
├── README.md                     # 本檔
├── SPEC_復現規格書.md            # 完整規格（必讀）
├── requirements.txt
├── sea_setup.py                  # 一鍵讓 ultralytics 認得 C2f_SEA（冪等、自動備份）
├── verify_sea.py                 # 驗證 SEA 整合
├── configs/
│   ├── yolov8n-seg-sea.yaml      # Agent 架構（backbone C2f → C2f_SEA）
│   ├── data_crackseg.yaml        # 訓練資料設定（Roboflow Crack-Seg）
│   └── cmc.yaml                  # CMC 超參數 + 大模型後端開關（v1 / sam2）
├── src/
│   ├── agent.py                  # YOLOv8-Seg+SEA 封裝（Agent）
│   ├── geometry.py               # 沿中軸取提示點 + 寬度（1st round）
│   ├── large_model.py            # SAM v1 / SAM2 後端工廠 + prompt 函式
│   ├── filters.py                # contour 過濾（2nd+3rd round）
│   ├── cmc.py                    # 衝突分析 + 完整四輪 pipeline
│   └── metrics.py                # P/R/F1/IoU
└── scripts/
    ├── train_agent.py            # 訓練 Agent
    ├── infer.py                  # 跑完整 CMC（可 --dump-steps 輸出四欄可視化）
    ├── eval.py                   # 在 mask 評估集算指標
    └── calibrate_sam_thresh.py   # SAM_THRESH 掃描校準（換 SAM2 後必跑）
```

## 環境（WSL2 + NVIDIA GPU）
```bash
# 0) Windows 端裝最新 NVIDIA 驅動；WSL 內驗證
nvidia-smi

# 1) 環境
conda create -n sepsam python=3.10 -y && conda activate sepsam

# 2) PyTorch（CUDA 12.1 輪子；RTX 40 系列適用）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) 其餘依賴
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git   # SAM v1

# 4) 權重
mkdir -p weights && cd weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth   # SAM ViT-H (~2.56GB)
cd ..
```

## Quickstart
```bash
# 1) 讓 ultralytics 認得 C2f_SEA（每次重裝/升級 ultralytics 後都要重跑）
python sea_setup.py
python verify_sea.py            # 期望：C2f_SEA modules: 4 ... VERIFICATION PASSED ✓

# 2) 準備資料：把 Roboflow Crack-Seg 放到 datasets/crack_seg/（train/valid/test）
#    下載頁：https://universe.roboflow.com/university-bswxt/crack-bphdr
#    若匯出結構不同，調整 configs/data_crackseg.yaml 的路徑

# 3) 訓練 Agent（先少量 epoch 跑通，再放大到 500）
python scripts/train_agent.py --epochs 500 --batch 16 --imgsz 416 --device 0
#    完成後把 runs/sepsam_agent/weights/best.pt 填回 configs/cmc.yaml 的 agent_ckpt

# 4) 推論（單張或資料夾；--dump-steps 出四欄中間結果）
python scripts/infer.py --source path/to/img.jpg --dump-steps

# 5) 評估（影像 + GT mask，依檔名配對）
python scripts/eval.py --images datasets/cfd/images --masks datasets/cfd/masks
```

## 切換到 SAM2（可選，見 SPEC 附錄 D）
```bash
pip install "git+https://github.com/facebookresearch/sam2.git"   # 可能需較新 torch
# 下載 SAM2.1 權重（用該 repo 的 checkpoints/download_ckpts.sh）到 weights/
# 編輯 configs/cmc.yaml：sam_backend: sam2
python scripts/calibrate_sam_thresh.py --images datasets/cfd/images --masks datasets/cfd/masks --limit 80
#   把建議的 SAM_THRESH 寫回 configs/cmc.yaml（SAM2 的分數分布與 v1 不同，必須重新校準）
```

## 建構里程碑（建議順序，細節見 SPEC §12）
1. 環境可用、`verify_sea.py` 通過。
2. `infer.py --dump-steps` 在單張圖能輸出四欄（draft / SAM raw / SAM filtered / final）。
3. Agent 訓練得到 `best.pt`。
4. `eval.py` 在 CFD 等評估集得到 P/R/F1/IoU，量級對照 SPEC 中論文 Table 6/7。
5. （可選）v1 vs SAM2 對照、SEA / CMC ablation。

## 重要提醒
- **SAM 永遠凍結**：只有 Agent（YOLOv8-Seg+SEA）需要訓練。任何讓 SAM 需要訓練的設計都偏離本論文。
- **載入訓練好的權重前，環境要先跑過 `sea_setup.py`**，否則 checkpoint 內的 `C2f_SEA` 無法 unpickle。
- **Self-collected（381 張）未公開**，無法完全重現論文 Table 6；用公開資料（Roboflow 訓練、CFD/VT 評估）驗證管線即可。
