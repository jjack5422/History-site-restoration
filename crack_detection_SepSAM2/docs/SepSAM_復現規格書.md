# SepSAM 復現規格書（給 Claude Code 的建構文件）

> **論文來源**：Zhizhang Zhou, Wenbo Hu, Guangda Xu, You Dong, *"Self-evolving prompting segment anything model for crack segmentation through data-driven cyclic conversations"*, **Advanced Engineering Informatics 68 (2025) 103626**. DOI: 10.1016/j.aei.2025.103626
>
> **本文件目的**：作為在 **WSL2** 上從零復現 SepSAM 的完整規格與交接文件。內容包含模型架構、核心演算法（含論文 pseudocode 轉成可實作的 Python 指引）、環境設定、訓練/推論流程、專案結構與評估方式。
>
> ⚠️ **重要前提**：原論文**未公開官方程式碼**（論文僅寫 "Data will be made available on request"）。因此本文件中的實作細節，部分是依論文文字敘述做的**合理重建**。每個關鍵點都標註：
> - `[論文明確]` — 論文中有明確數值或描述，照做即可。
> - `[需推斷]` — 論文沒寫死，以下為合理預設值/實作決策，Claude Code 可調整，並建議在程式中以參數化方式保留彈性。

---

## 0. TL;DR — 給 Claude Code 的任務摘要

請建立一個名為 `sepsam` 的 Python 專案，實作以下系統：

1. **Agent 模型**：YOLOv8-Seg + 自訂的 **SEA（Squeeze-and-Excitation Attention）** 模組（加在 backbone 的 C2f 之後）。這是**唯一需要訓練**的部分（trainable weight 約 6.75 MB for v8n）。
2. **Large 模型**：Meta 的 **SAM（ViT-H，frozen）**，不訓練，直接載入官方權重。
3. **CMC（Cyclic Model Conversation）推論管線**：4 輪流程，把 Agent 的初步結果轉成沿裂紋中軸的提示點餵給 SAM，再用 Agent 的領域知識過濾 SAM 雜訊，最後做衝突分析決定最終輸出。
4. **訓練腳本**（只訓練 Agent）、**推論腳本**（跑完整 CMC）、**評估腳本**（Precision / Recall / F1 / IoU / mAP）。

核心精神：**SAM 全程凍結**，靠一個輕量 Agent「對話式」地引導它，避免 fine-tune 大模型造成的過擬合與算力負擔。

---

## 1. 方法總覽（SepSAM 是什麼）

SepSAM 解決的問題：SAM 雖然泛化能力強，但 (1) 沒見過裂紋這種「細長管狀」結構，(2) 需要人工提示，(3) 模型 ~2GB 不利現場部署。而傳統訓練式裂紋模型（U-Net、DeepLab、純 YOLO）在訓練分布外**嚴重過擬合**。

SepSAM 的做法：用一個**輕量、可部署**的 YOLOv8-Seg 當「提示代理（prompting agent）」，自動產生提示引導**凍結的 SAM**，並透過多輪「對話 + 衝突分析」互相校正：
- Agent 懂裂紋（領域知識）但邊界粗糙、受標註偏差影響。
- SAM 邊界精緻、不受人工標註偏差影響，但缺領域知識、會在裂紋外產生碎片雜訊。
- CMC 機制讓兩者**取長補短**。

**設計上的關鍵差異**（相對其他 auto-prompt SAM）：別人多用「矩形框」提示（適合塊狀物件），但裂紋是不規則細長線狀，框提示無法反映走向。SepSAM 改用 **沿中軸（along-axis / medial-axis）取點**作為提示。

---

## 2. 系統架構

### 2.1 兩個模型的角色分工（論文 Table 1）

| 角色 | 功能 | 特性 |
|---|---|---|
| **Agent Model**（YOLOv8-Seg + SEA） | (1) 產生初步 mask 草稿；(2) 沿中軸產生提示點（含裂紋走向資訊）；(3) 用領域知識過濾 SAM 雜訊 | 輕量、可部署 IoT；邊界較弱；受人工標註偏差影響 |
| **SAM Model**（frozen） | 根據 Agent 的提示精修邊界 | 邊界精緻；不受人工標註影響；權重大、缺領域知識 |
| **CMC 融合機制** | (1) 把草稿轉成提示點並估算裂紋寬度；(2) 把 SAM 輸出轉成可分析的 contour；(3) 比較 SAM/Agent 衝突與信心，做最終決策 | 透過互補維持穩健輸出 |

### 2.2 Agent Model：YOLOv8-Seg + SEA `[論文明確 + 需推斷]`

- 基礎模型：**YOLOv8-Seg**（ultralytics）。論文同時測了 **v8n**（輕量，6.752 MB trainable）與 **v8x**（154.45 MB trainable）。`[論文明確]`
  - 預設用 `yolov8n-seg`（對齊論文主打的輕量部署訴求）；可切換 `yolov8x-seg` 重現最佳指標。
- 輸出：`f(I) → {B, P, M}`，即 bounding boxes `B`、class 機率 `P`、instance segmentation mask `M`。**`B` 的信心與 `M` 都會在 CMC 中用到。** `[論文明確]`
- **SEA 模組**：在 backbone 的 **C2f 層輸出之後**加上一個 Squeeze-and-Excitation 區塊。`[論文明確]`
  - 公式（論文 Eq. 8）：`SEA(x) = x · σ(W₂ · δ(W₁ · x))`，其中 `δ`=ReLU、`σ`=sigmoid，squeeze 用 global average pooling。
  - reduction ratio `r`：論文未指定 → 用 SE 預設 `r=16`。`[需推斷]`
  - 作用：論文 ablation 顯示 SEA 主要功能是**抑制過擬合**（在 unseen data 上 validation loss 更低更穩）。

#### Loss（論文 Eq. 2–6，沿用 YOLOv8-Seg 原生 loss）`[論文明確]`
`L_total = λ_bbox·L_bbox + λ_dfl·L_dfl + λ_cls·L_cls + λ_mask·L_mask`
- `L_bbox` = CIoU loss、`L_dfl` = distribution focal loss、`L_cls`/`L_mask` = cross-entropy。
- 這些**就是 ultralytics YOLOv8-Seg 的預設 loss**，不需自行重寫；用官方訓練流程即可。`λ` 用 ultralytics 預設。`[需推斷：論文未列 λ 值]`

### 2.3 Large Model：SAM（frozen）`[論文明確 + 需推斷]`

- 用 Meta 官方 **Segment Anything**。論文表中 SAM「total weight ≈ 2.5k MB」對應 **ViT-H**（`sam_vit_h_4b8939.pth`，約 2.56 GB）→ **預設用 `vit_h`**。`[推斷自權重大小]`
- 三大模組：image encoder、prompt encoder、mask decoder。**全部凍結**，不做任何 fine-tune。`[論文明確]`
- 提示型態：**只用 point prompts**（沿中軸取的正樣本點），不用框。`[論文明確]`
- 8GB GPU 可跑 ViT-H 推論（論文 inferring memory 約 7.5–8 GB）。若記憶體吃緊，提供 `vit_l` / `vit_b` 作為退路選項。`[需推斷]`

### 2.4 CMC 四輪對話 Pipeline（論文 Fig. 2、Fig. 5、Fig. 6）`[論文明確]`

完整資料流（對照論文 Fig. 2，紅字 1st–4th 為四輪）：

```
輸入影像 (unseen)
  │
  ▼
[Agent: YOLOv8-Seg+SEA] ──► draft mask S, boxes B, confidences conf
  │
  │  (1st round: 沿中軸提示)
  ▼
medial axis M(S)  +  distance transform   ─► 沿軸均勻取 N 個正樣本點 + 估算寬度 w
  │
  ▼
[SAM (frozen): point prompts] ──► Raw Prompting Result (raw mask) + sam_score
  │
  │  (2nd round: border following)
  ▼
cv2.findContours ─► contour instances（Quantified Mask）
  │
  │  (3rd round: 用 Agent 領域知識過濾)
  ▼
contour_filter（依 YOLO 信心保留前 k 大面積 contour）─► filtered mask Mc
  │
  │  (4th round: 衝突分析)
  ▼
conflict(Mc, S) < ε  AND  sam_score > SAM_THRESH ?
   ├─ 是 ─► 最終輸出 = Mc（SAM 精修結果）
   └─ 否 ─► 最終輸出 = S（回退到 Agent 草稿）
```

四輪語意：
- **1st（沿中軸提示）**：把 Agent 草稿轉成「沿裂紋走向、落在裂紋內」的提示點。
- **2nd（border following）**：把 SAM 的 raw mask 用邊界追蹤（Suzuki 演算法）拆成 contour 實例。
- **3rd（領域知識過濾）**：用 Agent 偵測到的有效裂紋數 `k`，只保留面積前 `k` 大的 contour，濾掉 SAM 在裂紋外的碎片雜訊。
- **4th（衝突分析）**：若 SAM 結果偏離草稿太多（提示可能跑出裂紋外導致誤導），或 SAM 信心不足，則回退到 Agent 草稿，確保穩健。

---

## 3. 核心演算法（含 pseudocode → 可實作 Python 指引）

> 以下 Python 為**實作指引**，非論文原始碼。函式介面建議照此設計以便 Claude Code 模組化。

### 3.1 1st round — 沿中軸取提示點 + 寬度估算（論文 Eq. 9, 10）`[論文明確（方法）/ 需推斷（取樣細節）]`

論文：取 Agent 草稿 `S` 的 **medial axis（中軸）** `M(S)`，沿軸**均勻取點**；中軸點落在裂紋內機率最高、且不偏向裂紋任一邊。寬度 `w = 2 × d(x, ∂S)`（中軸點到邊界的距離 ×2，用最近鄰/距離轉換取得）。

```python
import numpy as np
from skimage.morphology import medial_axis

def mask_to_points_and_width(mask_bin: np.ndarray, n_points: int):
    """
    mask_bin: HxW，0/1 binary（Agent 草稿，已二值化）
    n_points: 取樣點數，= max(H, W) // 50  (見超參數)
    回傳: pts (K,2) float [(x,y), ...] 給 SAM；widths (K,)
    """
    skel, dist = medial_axis(mask_bin.astype(bool), return_distance=True)
    ys, xs = np.where(skel)                      # 中軸像素座標
    if xs.size == 0:
        return np.empty((0, 2)), np.empty((0,))
    # [需推斷] 論文寫「沿中軸均勻取樣」。簡單版：對中軸像素索引均勻取樣。
    # 更忠實版（建議實作為可選）：先把中軸像素依曲線排序（如以最近鄰串成路徑），
    #   再依弧長等距取樣，避免分支處取樣不均。
    k = min(n_points, xs.size)
    sel = np.linspace(0, xs.size - 1, num=k).astype(int)
    pts = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float32)  # SAM 用 (x, y)
    widths = 2.0 * dist[ys[sel], xs[sel]]
    return pts, widths
```

**注意**：所有取樣點都是**正樣本（前景）**，給 SAM 時 `point_labels = 1`。論文未提到負樣本點。`[論文明確]`

### 3.2 SAM 提示推論 `[論文明確（介面）]`

```python
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

def build_sam(ckpt: str, model_type="vit_h", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=ckpt).to(device)
    sam.eval()
    return SamPredictor(sam)

def sam_prompt(predictor, image_rgb: np.ndarray, pts: np.ndarray):
    """回傳 raw mask (HxW uint8 0/255) 與 sam_score(float, SAM 的 IoU 預測)"""
    if pts.shape[0] == 0:
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), np.uint8), 0.0
    predictor.set_image(image_rgb)
    labels = np.ones(pts.shape[0], dtype=np.int64)
    masks, scores, _ = predictor.predict(
        point_coords=pts, point_labels=labels, multimask_output=False)
    raw = (masks[0].astype(np.uint8) * 255)
    return raw, float(scores[0])
```

### 3.3 2nd + 3rd round — contour 過濾（論文 Pseudo code 1）`[論文明確]`

```python
import cv2
import numpy as np

def contour_filter(mask: np.ndarray, yolo_conf: list, conf_thresh=0.5):
    """
    mask: SAM raw mask (HxW uint8 0/255)
    yolo_conf: Agent 偵測各 instance 的信心 list
    回傳: 只保留前 top_n 大面積 contour 的 mask
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros_like(mask)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    top_n = int(np.sum(np.array(yolo_conf) > conf_thresh))   # 高信心偵測數 k
    top_n = max(top_n, 0)
    accepted = contours_sorted[:top_n]
    out = np.zeros_like(mask)
    if accepted:
        cv2.drawContours(out, accepted, -1, 255, thickness=cv2.FILLED)
    return out
```
> 論文比較了三種過濾法（Table 2）：SAM 官方（濾 <100px 孤立 mask，但對不同影像尺寸不通用）、形態學 erosion-dilation（會傷到正確邊界）、以及本文的「border following + 依面積與領域知識過濾」（不傷正確 mask、抗雜訊）。**採用本文方法。**

### 3.4 4th round — 衝突分析（論文 Eq. 11 + Pseudo code 2）`[論文明確]`

論文 Eq. 11 判準：
```
Σ(Mc − Ms ⊗ Mc) / Σ Ms  <  ε        (⊗ = bitwise AND, ε = 1.5)
```
即「Mc 中**不與草稿 Ms 重疊**的部分面積」/「草稿 Ms 面積」要小於 ε。`Mc` = 3rd round 過濾後的 SAM mask；`Ms` = Agent 草稿。

```python
def conflict_ratio(mc: np.ndarray, ms: np.ndarray) -> float:
    mc_b = mc.astype(bool); ms_b = ms.astype(bool)
    inter = np.logical_and(mc_b, ms_b).sum()
    mc_only = mc_b.sum() - inter            # |Mc \ Ms|
    denom = max(int(ms_b.sum()), 1)
    return mc_only / denom
```

### 3.5 完整主迴圈（論文 Pseudo code 2）`[論文明確]`

```python
def cmc_predict(image_rgb, agent, predictor, hp):
    """回傳最終裂紋 mask (HxW uint8 0/255)"""
    # Step 1: Agent 草稿 + 信心
    mask_yolo, yolo_conf = agent.predict(image_rgb, conf=hp.YOLO_CONF_1)
    # Step 2: 沿中軸取提示點
    n_pts = max(image_rgb.shape[:2]) // hp.POINTS_DIVISOR     # //50
    pts, widths = mask_to_points_and_width(mask_yolo > 0, n_pts)
    # Step 3: SAM 提示
    mask_sam, sam_score = sam_prompt(predictor, image_rgb, pts)
    # Step 4: 用 Agent 領域知識過濾 SAM 雜訊
    mask_sam = contour_filter(mask_sam, yolo_conf, hp.YOLO_CONF_2)
    # Step 5: 衝突分析 + 信心門檻 → 決策
    conflict = conflict_ratio(mask_sam, mask_yolo)
    if conflict < hp.CONFLICTION_RATIO and sam_score > hp.SAM_THRESH:
        return mask_sam          # 接受 SAM 精修
    else:
        return mask_yolo         # 回退到 Agent 草稿
```

---

## 4. 超參數（論文 Table 3）`[論文明確]`

| 參數 | 說明 | 預設值 |
|---|---|---|
| `YOLO_CONF_1` | 高於此信心的裂紋 instance 才進入沿軸提示流程 | **0.0** |
| `YOLO_CONF_2` | Agent 接受 SAM 精修 contour 的門檻（contour_filter 用） | **0.5** |
| `SAM_THRESH` | SAM IoU 預測門檻，低於此的提示結果被拒 | **0.85** |
| `CONFLICTION_RATIO` (ε) | Agent 與 SAM 預測的衝突比，過高則拒絕 SAM 輸出 | **1.50** |
| `POINTS_QUANTITY` | 沿軸取樣點數 | **`max(H, W) // 50`** |

最佳取樣密度的額外發現（論文 §5.4）：**約 60 點/影像**時 precision 最佳；點太多會有大量點落在裂紋外，反而誤導 SAM 使 precision 階梯式下降。`POINTS_QUANTITY = max(H,W)//50` 對 1024 影像約 20 點、對更大影像更多，是個保守設定。建議把點數做成可調參數。

建議把以上集中成一個 `configs/cmc.yaml`：
```yaml
YOLO_CONF_1: 0.0
YOLO_CONF_2: 0.5
SAM_THRESH: 0.85
CONFLICTION_RATIO: 1.50
POINTS_DIVISOR: 50      # n_points = max(H,W) // POINTS_DIVISOR
sam_model_type: vit_h
sam_ckpt: weights/sam_vit_h_4b8939.pth
agent_ckpt: runs/segment/train/weights/best.pt
```

---

## 5. 資料集（論文 Table 4）`[論文明確]`

| 資料集 | 影像尺寸 | 數量 | 用途 | 來源 |
|---|---|---|---|---|
| **Roboflow Crack-Seg** | 416×416 | 3717 train / 200 val / 112 test | **訓練 Agent**（所有模型都用它訓練） | https://universe.roboflow.com/university-bswxt/crack-bphdr |
| **CFD (Crack Forest Dataset)** | 480×320 | 118（多為細裂紋，平均寬 2.31px） | 評估 | 論文 ref [13]（Shi et al. 2016） |
| **Self-collected** | 1024×1024 | 381（平均寬 11.46px，線狀/網狀/不規則） | 評估（高品質） | 網路 + 真實橋樑（**未公開**） |
| **PChun** | 5184×3456 | 98（平均寬 4.82px） | zero-shot 評估 | 論文 ref [68]（Pang-jo Chun） |
| **VT Crack Collection** | 448×448 | 9900 train / 1096 val（含 Crack500、DeepCrack、GAPs、Rissbilder、Volker） | zero-shot 評估 | 論文 ref [69]，doi:10.7294/16625056.v1 |

實作注意：
- **訓練只需 Roboflow Crack-Seg**（YOLO segmentation 格式：images + labels 的 polygon txt）。`[論文明確]`
- 評估用資料集（CFD/PChun/VT）多為**語意分割 mask 格式**，非 YOLO 格式 → 評估走「像素級 mask 比對」，不需轉成 YOLO 訓練格式。`[需推斷：論文未給轉檔細節]`
- Self-collected 未公開，無法完全重現該表；用公開資料集即可驗證管線正確性。

`configs/data_crackseg.yaml`（ultralytics 格式範例）：
```yaml
path: datasets/crack_seg
train: train/images
val: valid/images
test: test/images
names:
  0: crack
```

---

## 6. 環境設定（WSL2）

> 假設：Windows 11 + WSL2（Ubuntu 22.04）+ NVIDIA GPU（如論文用的 RTX 4060 8GB）。

### 6.1 前置（Windows 主機端）
1. 安裝**最新版 NVIDIA 顯示卡驅動**（Windows 端，內含 WSL GPU 支援）。**WSL 內不要另外裝顯卡驅動**。
2. 安裝 WSL2 + Ubuntu：PowerShell 執行 `wsl --install -d Ubuntu-22.04`。

### 6.2 在 WSL 內驗證 GPU
```bash
nvidia-smi      # 應看到你的 GPU；看不到就先修好 Windows 驅動與 WSL2
```

### 6.3 建立環境（Miniconda）
```bash
# 安裝 miniconda（若尚未安裝）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate

conda create -n sepsam python=3.10 -y
conda activate sepsam
```

### 6.4 安裝套件
```bash
# PyTorch（CUDA 12.1 輪子；RTX 40 系列適用。若驅動較舊可改 cu118）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 核心依賴
pip install ultralytics opencv-python scikit-image numpy scipy matplotlib pyyaml tqdm

# Segment Anything（Meta 官方）
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 6.5 下載權重
```bash
mkdir -p weights && cd weights
# SAM ViT-H（約 2.56 GB）
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# （可選，記憶體吃緊時的退路）
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth   # ViT-L
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth   # ViT-B
cd ..
# YOLOv8-Seg 預訓練（ultralytics 會自動下載 yolov8n-seg.pt / yolov8x-seg.pt，亦可手動）
```

### 6.6 驗證安裝
```bash
python - <<'PY'
import torch, ultralytics, cv2, skimage, segment_anything
print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("ultralytics:", ultralytics.__version__)
PY
```

---

## 7. 訓練流程（只訓練 Agent；SAM 不訓練）

### 7.1 訓練設定（論文 §4.1, §5.1）`[論文明確]`
- batch size **16**，影像 **416×416**。
- epochs：主比較用 **200**；ablation 跑到 **500**（建議 ≥300 才能展現 SEA 抗過擬合）。
- 資料：Roboflow Crack-Seg 3717 張訓練。
- device：單張 GPU（8GB 可行）。

### 7.2 SEA 模組整合（**最關鍵、最易出錯的一步**）`[需推斷：論文未給整合方式]`

> ✅ **已備好可直接使用的解法 → 見文末「附錄 C」。** 附錄 C 提供 `yolov8n-seg-sea.yaml`、`sea_setup.py`（一鍵讓 ultralytics 認得 `C2f_SEA`，冪等且自動備份）、`verify_sea.py`（驗證），是**建議採用的正式做法**。**請以附錄 C 為實作依據**；下方方式 A/B 僅作原理說明。
>
> 關鍵原因：ultralytics 的 `train()` 會**從 yaml 重新建模**，不會沿用你在 Python 裡改過的 model 物件。所以必須讓 yaml 直接寫 `C2f_SEA` 且 `parse_model` 認得它（= 附錄 C 的做法），事後 wrapper 的修改會在訓練時被丟棄。

論文把 SE 加在 YOLOv8-Seg backbone 的 **C2f 輸出之後**。實作建議：

**`src/models/sea.py`**
```python
import torch, torch.nn as nn, torch.nn.functional as F
from ultralytics.nn.modules import C2f   # 視 ultralytics 版本而定

class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(c // r, 1), 1)
        self.fc2 = nn.Conv2d(max(c // r, 1), c, 1)
    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)          # squeeze: GAP
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))            # excitation
        return x * s                               # scale

class C2f_SEA(C2f):
    """C2f 後接 SE（論文 Eq.8：加在 C2f 最終輸出）"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.se = SE(c2)
    def forward(self, x):
        return self.se(super().forward(x))
```

**整合到 ultralytics（兩種方式，擇一；建議方式 A）：**

- **方式 A（自訂 yaml + 註冊模組）**：複製 `yolov8-seg.yaml` 成 `configs/yolov8-seg-sea.yaml`，把 backbone 中的 `C2f` 改成 `C2f_SEA`；訓練前把 `C2f_SEA` 註冊進 ultralytics 的模型解析命名空間：
  ```python
  import ultralytics.nn.tasks as tasks
  from src.models.sea import C2f_SEA
  setattr(tasks, "C2f_SEA", C2f_SEA)        # 讓 parse_model 的 globals() 找得到
  # 注意：不同 ultralytics 版本 parse_model 解析方式略有差異，需實測。
  ```
  > ⚠️ Claude Code 請對照**實際安裝的 ultralytics 版本**驗證此註冊方式是否生效（用一張圖跑一次 forward 確認 SE 有被呼叫）。這是整個復現裡最可能卡關的點。

- **方式 B（退路）**：載入標準 `yolov8x-seg`，在建立模型後遍歷 `model.model.modules()`，把 backbone 的 `C2f` 動態替換成 `C2f_SEA` 並搬移既有權重。較 hacky，可能影響權重載入，僅在方式 A 不通時使用。

### 7.3 訓練指令
```python
# scripts/train_agent.py
import ultralytics.nn.tasks as tasks
from src.models.sea import C2f_SEA
setattr(tasks, "C2f_SEA", C2f_SEA)

from ultralytics import YOLO
model = YOLO("configs/yolov8-seg-sea.yaml")     # 從自訂架構訓練
model.train(
    data="configs/data_crackseg.yaml",
    epochs=500, imgsz=416, batch=16, device=0,
    project="runs", name="sepsam_agent",
)
```
訓練完權重在 `runs/sepsam_agent/weights/best.pt`，填回 `configs/cmc.yaml` 的 `agent_ckpt`。

> **過擬合監控**：論文重點是 SEA 抗過擬合。請記錄 train/val 的 mAP50 與 loss 曲線（ultralytics 會自動存），並做「有/無 SEA」對照，重現論文 Fig. 13。

---

## 8. 推論流程（完整 CMC）

```python
# scripts/infer.py（骨架）
import cv2, yaml, numpy as np
from types import SimpleNamespace
from ultralytics import YOLO
from src.models.sea import C2f_SEA
import ultralytics.nn.tasks as tasks; setattr(tasks, "C2f_SEA", C2f_SEA)
from src.sam_prompt import build_sam
from src.cmc import cmc_predict   # 內含 §3.5 主迴圈

class Agent:
    def __init__(self, ckpt): self.m = YOLO(ckpt)
    def predict(self, image_rgb, conf=0.0):
        r = self.m.predict(image_rgb, conf=conf, retina_masks=True, verbose=False)[0]
        h, w = image_rgb.shape[:2]
        if r.masks is None:
            return np.zeros((h, w), np.uint8), []
        m = r.masks.data.cpu().numpy()               # (N,H,W)
        mask = (m.sum(0) > 0).astype(np.uint8) * 255
        conf_list = r.boxes.conf.cpu().numpy().tolist()
        return mask, conf_list

hp = SimpleNamespace(**yaml.safe_load(open("configs/cmc.yaml")))
agent = Agent(hp.agent_ckpt)
predictor = build_sam(hp.sam_ckpt, hp.sam_model_type, "cuda")

img = cv2.cvtColor(cv2.imread("path/to/image.jpg"), cv2.COLOR_BGR2RGB)
final_mask = cmc_predict(img, agent, predictor, hp)
cv2.imwrite("out_mask.png", final_mask)
```

額外可加（論文有，但屬加值功能）：沿中軸的**裂紋寬度分佈圖**（用 §3.1 回傳的 `widths`），重現論文 Fig. 21 的 width distribution。`[論文明確（概念）/ 需推斷（畫圖細節）]`

---

## 9. 評估指標（論文 §3.5, Eq. 12–16）`[論文明確]`

像素級分割指標：
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2·P·R / (P + R)`
- `IoU = |交集| / |聯集|`
- 偵測用 `mAP`（ultralytics 內建，評 Agent 偵測用）。

`src/metrics.py` 建議實作「給定 pred mask 與 gt mask（皆二值）回傳 P/R/F1/IoU」的函式，並支援**依裂紋寬度分桶**（論文常用 width>3、width>20 分組比較），以重現 Table 6/7。

> **評估上的坑（論文 Limitations）**：人工標註本身有偏差，SepSAM 傾向貼合「真實裂紋邊界」而非標註，導致在標註偏差大的資料集（PChun、Rissbilder）上指標被**低估**。報告結果時建議同時放質化圖（白=標註、紅=FP、綠=TP）佐證。

---

## 10. 建議專案結構

```
sepsam/
├── README.md
├── requirements.txt
├── configs/
│   ├── yolov8-seg-sea.yaml      # 自訂 SEA 架構（由 yolov8-seg.yaml 改）
│   ├── data_crackseg.yaml       # Roboflow 資料設定
│   └── cmc.yaml                 # CMC 超參數（§4）
├── weights/
│   └── sam_vit_h_4b8939.pth
├── datasets/
│   ├── crack_seg/               # Roboflow（YOLO seg 格式）
│   ├── cfd/  pchun/  vt/         # 評估用（mask 格式）
├── src/
│   ├── models/sea.py            # SE / C2f_SEA（§7.2）
│   ├── geometry.py              # medial axis 取點 + 寬度（§3.1）
│   ├── sam_prompt.py            # build_sam / sam_prompt（§3.2）
│   ├── filters.py               # contour_filter（§3.3）
│   ├── cmc.py                   # conflict_ratio + cmc_predict（§3.4, §3.5）
│   └── metrics.py               # P/R/F1/IoU（§9）
└── scripts/
    ├── train_agent.py           # 只訓練 Agent（§7.3）
    ├── infer.py                 # 跑完整 CMC（§8）
    └── eval.py                  # 在評估集上算指標
```

---

## 11. 實作注意事項 / 已知模糊點（請 Claude Code 特別留意）

1. **SEA 整合（§7.2）是最大風險點**：ultralytics 版本差異會影響自訂模組註冊。請先寫一個最小測試：建模 → 跑一張圖 forward → 確認 `C2f_SEA.forward` 被呼叫、輸出 shape 正確，再開始訓練。
2. **medial axis 取樣順序（§3.1）**：論文說「沿中軸均勻取樣」。最簡單版是對 skeleton 像素索引均勻取；但分支多的網狀裂紋會取樣不均。建議實作「依弧長等距取樣」的較忠實版本，並用參數切換。
3. **SAM 模型大小**：預設 `vit_h`（對齊論文 ~2.5GB 權重）。8GB GPU 跑得動但接近上限；若 OOM，先降到 `vit_l`/`vit_b` 並在報告中註明。
4. **reduction ratio `r=16`、loss 權重 `λ`**：論文未給，用常見預設，**標記為可調超參數**，勿宣稱為論文原值。
5. **資料集格式不一致**：訓練集（Roboflow）是 YOLO polygon 格式；評估集多為 mask 格式。評估流程要走像素比對，別硬套 YOLO val。
6. **Self-collected（381 張）未公開**：無法完全重現論文 Table 6。用公開資料集驗證管線即可。
7. **點數 vs precision 的取捨（§4 / §5.4）**：點太多 precision 反而下降。把 `POINTS_DIVISOR`（預設 50）做成可調，並可加「最佳約 60 點/影像」的提示。
8. **CMC 是純推論期機制**：除了 Agent（YOLOv8-Seg+SEA）需要訓練，CMC 各步驟（中軸取點、contour 過濾、衝突分析）都是**確定性後處理**，不需訓練。

---

## 12. 建構里程碑 Checklist（建議 Claude Code 依序完成）

- [ ] **M0 環境**：WSL2 GPU 可用、`torch.cuda.is_available()==True`、SAM/YOLO 權重就位、`infer` 依賴可 import。
- [ ] **M1 SAM 基線**：寫 `sam_prompt`，手動給幾個點能在一張裂紋圖上得到 raw mask（驗證 SAM 載入正確）。
- [ ] **M2 幾何模組**：`mask_to_points_and_width` 能從一張二值 mask 取出沿中軸的點並可視化（疊在圖上看點是否落在裂紋內）。
- [ ] **M3 Agent 架構**：`C2f_SEA` 整合成功，`yolov8-seg-sea.yaml` 能建模並 forward（最小測試通過）。
- [ ] **M4 Agent 訓練**：在 Roboflow Crack-Seg 上訓練（先少量 epoch 跑通流程，再跑 200–500 epoch），得到 `best.pt`。
- [ ] **M5 CMC 串接**：`contour_filter` + `conflict_ratio` + `cmc_predict` 串成完整 4 輪，對單張圖輸出最終 mask，並能 dump 中間結果（draft / points / raw SAM / filtered / final）重現論文 Fig. 18 那種四欄可視化。
- [ ] **M6 評估**：`eval.py` 在 CFD 等評估集上算 P/R/F1/IoU，並依寬度分桶，對照論文 Table 6/7 量級。
- [ ] **M7 Ablation（可選）**：有/無 SEA、有/無 CMC 的對照，重現 Fig. 13 / Fig. 15 的趨勢。

---

## 附錄 A：建議依賴版本（`requirements.txt`）

```
torch>=2.1          # 安裝時用 --index-url https://download.pytorch.org/whl/cu121
torchvision
ultralytics>=8.1.0
opencv-python>=4.8
scikit-image>=0.22
numpy>=1.24
scipy>=1.11
matplotlib>=3.7
pyyaml>=6.0
tqdm>=4.66
# segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git
```

## 附錄 B：關鍵參考連結

- 論文 DOI：https://doi.org/10.1016/j.aei.2025.103626
- Segment Anything（程式碼 + 權重）：https://github.com/facebookresearch/segment-anything
- Ultralytics YOLOv8：https://github.com/ultralytics/ultralytics
- Roboflow Crack-Seg 資料集：https://universe.roboflow.com/university-bswxt/crack-bphdr
- VT Concrete Crack Conglomerate：doi:10.7294/16625056.v1
- 中軸 / thinning：scikit-image `medial_axis`、Zhang-Suen thinning（論文 ref [62]）
- border following：Suzuki & Abe 1985（論文 ref [63]，即 `cv2.findContours`）
- SE 注意力：Hu et al., Squeeze-and-Excitation Networks, CVPR 2018（論文 ref [59]）

---

### 給 Claude Code 的最後提醒
- 凡標 `[需推斷]` 之處，請把它做成**可調參數**並在程式註解標明「論文未指定」，**不要當成論文事實**陳述。
- 先用「少量 epoch + 單張圖 smoke test」把整條管線（M1–M5）跑通，再放大訓練規模，能省下大量除錯時間。
- 整個系統的正確性核心是：**SAM 永遠凍結，靠 Agent 沿中軸取點來提示，再用 Agent 信心過濾、用衝突分析回退**。任何讓 SAM 反而需要訓練的設計都偏離了本論文。

---

## 附錄 C：SEA 整合 — 可直接使用的檔案與步驟（**正式做法，取代 §7.2 的整合說明**）

### C.0 為什麼一定要 yaml-native
ultralytics 的 `Trainer` 在 `train()` 時會**根據 `model.yaml` 重新建構模型**（`get_model(cfg=...)`），不會沿用你在 Python 端修改過的 model 物件。因此：
- ❌「載入標準模型 → 事後把 C2f 包上 SE」的 wrapper 做法，會在訓練時被丟棄。
- ✅ 正解：YAML 直接寫 `C2f_SEA`，並讓 ultralytics 的 `parse_model` 認得 `C2f_SEA`（才能正確處理 channel 數與 repeat 數）。

`sea_setup.py` 就是把 `C2f_SEA` 注入 `ultralytics/nn/tasks.py` 來達成這件事。此法可正常支援 **train / resume / save / load**。

### C.1 三個檔案（放置位置）
```
sepsam/
├── configs/yolov8n-seg-sea.yaml   # 架構（backbone 4 個 C2f → C2f_SEA）
├── sea_setup.py                   # 一鍵修補 ultralytics（冪等、自動備份）
└── verify_sea.py                  # 驗證（新 process 跑）
```

### C.2 操作步驟
```bash
conda activate sepsam

# 1) 修補 ultralytics（每次重裝/升級 ultralytics 後都要重跑一次）
python sea_setup.py
#   會印出 tasks.py 路徑、備份檔，與 "PATCH WRITTEN ✓"

# 2) 驗證（務必用新 process）
python verify_sea.py
#   期望輸出： C2f_SEA modules: 4 ... VERIFICATION PASSED ✓

# 3) 訓練 Agent（從自訂架構，從零訓練）
yolo segment train \
  model=configs/yolov8n-seg-sea.yaml \
  data=configs/data_crackseg.yaml \
  epochs=500 imgsz=416 batch=16 device=0
# 或 Python：
#   from ultralytics import YOLO
#   YOLO("configs/yolov8n-seg-sea.yaml").train(
#       data="configs/data_crackseg.yaml", epochs=500, imgsz=416, batch=16, device=0)
```

### C.3 `configs/yolov8n-seg-sea.yaml`（完整內容）
> 僅把 backbone 第 2/4/6/8 層的 `C2f` 改成 `C2f_SEA`（論文：SEA 加在 backbone）。head 維持 `C2f`。
> 檔名含 `n` → 自動套 n 尺度；要用 x，複製成 `yolov8x-seg-sea.yaml`（內容相同，靠檔名選尺度）。

```yaml
nc: 1                       # 類別數：crack（單類）
scales:                     # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]
  s: [0.33, 0.50, 1024]
  m: [0.67, 0.75, 768]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.25, 512]

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]        # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]       # 1-P2/4
  - [-1, 3, C2f_SEA, [128, True]]    # 2          <-- SEA
  - [-1, 1, Conv, [256, 3, 2]]       # 3-P3/8
  - [-1, 6, C2f_SEA, [256, True]]    # 4          <-- SEA
  - [-1, 1, Conv, [512, 3, 2]]       # 5-P4/16
  - [-1, 6, C2f_SEA, [512, True]]    # 6          <-- SEA
  - [-1, 1, Conv, [1024, 3, 2]]      # 7-P5/32
  - [-1, 3, C2f_SEA, [1024, True]]   # 8          <-- SEA
  - [-1, 1, SPPF, [1024, 5]]         # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]        # 11  cat backbone P4
  - [-1, 3, C2f, [512]]              # 12
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]        # 14  cat backbone P3
  - [-1, 3, C2f, [256]]              # 15  (P3/8-small)
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]       # 17  cat head P4
  - [-1, 3, C2f, [512]]              # 18  (P4/16-medium)
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]        # 20  cat head P5
  - [-1, 3, C2f, [1024]]             # 21  (P5/32-large)
  - [[15, 18, 21], 1, Segment, [nc, 32, 256]]   # 22  Segment(P3, P4, P5)
```

### C.4 `sea_setup.py` 做了什麼（重點）
- 在 `ultralytics/nn/tasks.py` 中：(1) 於 `def parse_model` 前插入 `SE` 與 `C2f_SEA` 類別定義；(2) 把 `C2f_SEA` 加進 `parse_model` 內部的模組辨識集合（讓它比照 `C2f` 取得 `c1,c2` 與 repeat 數）。
- **冪等**：重複執行不會重覆插入。**自動備份** `tasks.py.sepsam.bak`。
- `C2f_SEA` 定義：`forward` 與 `forward_split` 都在 `C2f` 原輸出後接 `SE`；`SE(x)=x·σ(W₂·δ(W₁·x))`，reduction `r=16`（論文未指定，可調 `[需推斷]`）。
- 只改 backbone 那 4 層是由 **yaml** 決定的；`sea_setup.py` 只負責讓 `C2f_SEA` 這個名字可用。

### C.5 重要注意事項
1. **每次 `pip install -U ultralytics` 或重裝後，要重跑 `python sea_setup.py`**（重裝會還原修補）。
2. **驗證一定用新 process**：`sea_setup.py` 改的是磁碟上的檔案，已載入的 Python 行程看不到變更；`verify_sea.py` 另開行程才會吃到。
3. **resume 可用**：因為架構是 yaml-native，`resume=True` 重建模型時仍會得到 `C2f_SEA`。
4. **載入訓練好的權重**：checkpoint 內會 pickle 到 `ultralytics.nn.tasks.C2f_SEA` 這個類別路徑。所以在**載入權重前，該環境必須已經跑過 `sea_setup.py`**（類別存在於該路徑才能 unpickle）。推論端只要：`python sea_setup.py`（若尚未修補）→ 再 `YOLO("runs/.../best.pt")` 即可。
5. **版本相容性**：`sea_setup.py` 用「只在 `parse_model` 函式內替換獨立 `C2f` token」的方式加入辨識集合，可容忍多數 ultralytics 版本差異；但若 ultralytics 大改 `parse_model` 結構導致 `verify_sea.py` 失敗，請依 `verify_sea.py` 的錯誤訊息，手動在 `tasks.py` 的兩個模組集合（含 `C2f` 的那兩處 `{...}`）補上 `C2f_SEA`，並確認類別已定義。
6. **不要在別處另外定義 `C2f_SEA`**：以 `sea_setup.py` 注入到 `ultralytics.nn.tasks` 為單一來源，避免 pickle 類別路徑不一致。

### C.6 `verify_sea.py` 期望輸出
```
config: configs/yolov8n-seg-sea.yaml
C2f_SEA modules: 4
forward test (1x3x416x416)...
VERIFICATION PASSED ✓  (C2f_SEA active, forward OK)
```
（n 尺度下 backbone 4 個 C2f_SEA；數字應 = 4。若為 0 → 修補未生效，回到 C.2 步驟 1。）

---

## 附錄 D：SAM2 變體（**可選** — 讓 Claude Code 在 SAM v1 / SAM2 之間切換）

### D.0 定位與前提
- 本附錄提供把凍結大模型從 **SAM v1（ViT-H）** 換成 **SAM2（Hiera）** 的做法。
- **CMC 四輪邏輯、沿中軸取點、contour 過濾、衝突分析、YOLOv8+SEA Agent、`sea_setup.py` 全部不變**；只換「建立大模型」與「`sam_prompt`」這兩處，因為介面相同（吃影像+點 → 回傳 mask 與 IoU 分數）。
- ⚠️ **這是對論文的改動，不是忠實復現**：論文用 SAM v1，Table 5/6/7 的數字是 v1 跑的。要對論文數字 → 用 v1；要做改良系統（細裂紋/古蹟方向）→ 可用 SAM2，但結果須以「SepSAM 改良版」報告，不可直接對照論文表。`[實作決策]`
- SAM2 相對 v1 通常**更輕、更快**（Hiera backbone），與論文「輕量、可現場部署」訴求同方向。

### D.1 安裝與環境注意
```bash
conda activate sepsam
pip install "git+https://github.com/facebookresearch/sam2.git"
# 權重：用官方 repo 內的 checkpoints/download_ckpts.sh 下載（最保險），或從 HF 取得。
# 8GB GPU 建議 base_plus 或 large。
```
- ⚠️ **PyTorch 版本可能要升**：SAM2 需要較新的 torch（約 ≥2.3，以 sam2 repo 的 `requirements`/`setup.py` 為準）。若 §6 安裝的 torch 過舊，先升級 torch/torchvision（用對應 CUDA 輪子），再裝 sam2。
- **不要使用 SAM2 的 memory / video 機制**：SepSAM 是單張影像，只用 `SAM2ImagePredictor`，memory bank 完全用不到。

### D.2 設定檔名與 checkpoint 對照（SAM2.1）`[論文外 / 依官方 repo]`
| 尺寸 | model_cfg | checkpoint | 適用 |
|---|---|---|---|
| tiny | `configs/sam2.1/sam2.1_hiera_t.yaml` | `sam2.1_hiera_tiny.pt` | 最快、最省 |
| small | `configs/sam2.1/sam2.1_hiera_s.yaml` | `sam2.1_hiera_small.pt` | 輕量 |
| base_plus | `configs/sam2.1/sam2.1_hiera_b+.yaml` | `sam2.1_hiera_base_plus.pt` | **8GB GPU 推薦折衷** |
| large | `configs/sam2.1/sam2.1_hiera_l.yaml` | `sam2.1_hiera_large.pt` | 最佳品質 |

> `model_cfg` 是 sam2 套件內建的設定路徑（不是你自己的檔），由 `build_sam2(model_cfg, ckpt)` 解析。

### D.3 在 `configs/cmc.yaml` 增加後端切換開關
在 §4 的 `cmc.yaml` 基礎上加上 `sam_backend` 與 SAM2 欄位：
```yaml
# --- 共用 ---
YOLO_CONF_1: 0.0
YOLO_CONF_2: 0.5
SAM_THRESH: 0.85            # ⚠️ 換 SAM2 後需重新校準（見 D.5）
CONFLICTION_RATIO: 1.50
POINTS_DIVISOR: 50
agent_ckpt: runs/sepsam_agent/weights/best.pt
device: cuda

# --- 大模型後端切換：'v1' 或 'sam2' ---
sam_backend: sam2

# v1 (SAM ViT-H) 用：
sam_model_type: vit_h
sam_ckpt: weights/sam_vit_h_4b8939.pth

# sam2 用：
sam2_cfg: configs/sam2.1/sam2.1_hiera_b+.yaml
sam2_ckpt: weights/sam2.1_hiera_base_plus.pt
```

### D.4 統一的大模型工廠（`src/large_model.py`）
此檔依 `hp.sam_backend` 回傳 `(predictor, prompt_fn)`；兩種後端的 `prompt_fn` 介面相同，CMC 主迴圈無需區分。

```python
# src/large_model.py
import numpy as np
import torch


def build_large_model(hp):
    """依 hp.sam_backend 回傳 (predictor, prompt_fn)。"""
    if hp.sam_backend == "v1":
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[hp.sam_model_type](checkpoint=hp.sam_ckpt).to(hp.device)
        sam.eval()
        return SamPredictor(sam), sam_prompt_v1

    elif hp.sam_backend == "sam2":
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(hp.sam2_cfg, hp.sam2_ckpt, device=hp.device)
        return SAM2ImagePredictor(model), sam_prompt_sam2

    raise ValueError(f"unknown sam_backend: {hp.sam_backend!r} (use 'v1' or 'sam2')")


def sam_prompt_v1(predictor, image_rgb, pts):
    """SAM v1：回傳 (mask uint8 0/255, sam_score float)。"""
    if pts.shape[0] == 0:
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), np.uint8), 0.0
    predictor.set_image(image_rgb)
    labels = np.ones(pts.shape[0], dtype=np.int64)
    masks, scores, _ = predictor.predict(
        point_coords=pts, point_labels=labels, multimask_output=False)
    return (masks[0].astype(np.uint8) * 255), float(scores[0])


def sam_prompt_sam2(predictor, image_rgb, pts):
    """SAM2：介面與 v1 完全相同。"""
    if pts.shape[0] == 0:
        h, w = image_rgb.shape[:2]
        return np.zeros((h, w), np.uint8), 0.0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_rgb)
        labels = np.ones(pts.shape[0], dtype=np.int64)
        masks, scores, _ = predictor.predict(
            point_coords=pts, point_labels=labels, multimask_output=False)
    return (masks[0].astype(np.uint8) * 255), float(scores[0])
```

**對 §3.5 主迴圈的唯一調整**：把寫死的 `sam_prompt(...)` 改成傳入的 `prompt_fn(...)`：
```python
def cmc_predict(image_rgb, agent, predictor, prompt_fn, hp):
    mask_yolo, yolo_conf = agent.predict(image_rgb, conf=hp.YOLO_CONF_1)
    n_pts = max(image_rgb.shape[:2]) // hp.POINTS_DIVISOR
    pts, widths = mask_to_points_and_width(mask_yolo > 0, n_pts)
    mask_sam, sam_score = prompt_fn(predictor, image_rgb, pts)   # ← 後端無關
    mask_sam = contour_filter(mask_sam, yolo_conf, hp.YOLO_CONF_2)
    conflict = conflict_ratio(mask_sam, mask_yolo)
    if conflict < hp.CONFLICTION_RATIO and sam_score > hp.SAM_THRESH:
        return mask_sam
    return mask_yolo
```
推論端建立方式：
```python
from src.large_model import build_large_model
predictor, prompt_fn = build_large_model(hp)
final = cmc_predict(img, agent, predictor, prompt_fn, hp)
```

### D.5 `SAM_THRESH` 重新校準（換 SAM2 後**務必做**）
原因：SAM2 的 IoU 預測分數（`scores`）分布與 v1 不同，論文的 `SAM_THRESH=0.85` 不一定最佳。建議用驗證集做門檻掃描，挑使 F1 或 IoU 最佳者。

```python
# scripts/calibrate_sam_thresh.py（骨架）
import numpy as np, yaml
from types import SimpleNamespace
from src.large_model import build_large_model
from src.cmc import mask_to_points_and_width, contour_filter, conflict_ratio
from src.metrics import prf_iou   # 回傳 (P,R,F1,IoU)

hp = SimpleNamespace(**yaml.safe_load(open("configs/cmc.yaml")))
predictor, prompt_fn = build_large_model(hp)
# 載入 agent ...（同 infer.py）

val_items = [...]   # [(image_rgb, gt_mask), ...] 取驗證子集（~50–100 張即可）

# 1) 先跑一次，蒐集每張的 (sam_score, mask_sam, mask_yolo, gt)
records = []
for img, gt in val_items:
    mask_yolo, yolo_conf = agent.predict(img, conf=hp.YOLO_CONF_1)
    n = max(img.shape[:2]) // hp.POINTS_DIVISOR
    pts, _ = mask_to_points_and_width(mask_yolo > 0, n)
    mask_sam, score = prompt_fn(predictor, img, pts)
    mask_sam = contour_filter(mask_sam, yolo_conf, hp.YOLO_CONF_2)
    conf = conflict_ratio(mask_sam, mask_yolo)
    records.append((score, conf, mask_sam, mask_yolo, gt))

# 2) 掃描門檻，挑最佳（同時可順便掃 CONFLICTION_RATIO）
best = None
for thr in np.arange(0.60, 0.96, 0.05):
    f1s = []
    for score, conf, m_sam, m_yolo, gt in records:
        chosen = m_sam if (conf < hp.CONFLICTION_RATIO and score > thr) else m_yolo
        f1s.append(prf_iou(chosen, gt)[2])     # F1
    mean_f1 = float(np.mean(f1s))
    if best is None or mean_f1 > best[1]:
        best = (round(float(thr), 2), mean_f1)
print("best SAM_THRESH:", best[0], " mean F1:", round(best[1], 4))
```
做法重點：
- 驗證子集用**有 mask ground truth** 的資料（如 CFD 的一部分）。
- 先掃 `SAM_THRESH ∈ {0.60, 0.65, …, 0.95}`；若想更精細，再對最佳點附近細掃，並可同時掃 `CONFLICTION_RATIO`。
- 把選出的最佳值寫回 `configs/cmc.yaml` 的 `SAM_THRESH`。
- 其餘超參數（`YOLO_CONF_*`、`POINTS_DIVISOR`）大致沿用論文值即可；`POINTS_QUANTITY` 仍建議維持「最佳約 60 點/影像」的觀察。

### D.6 驗證 SAM2 後端可用
```python
import yaml
from types import SimpleNamespace
from src.large_model import build_large_model
import numpy as np

hp = SimpleNamespace(**yaml.safe_load(open("configs/cmc.yaml")))
assert hp.sam_backend == "sam2"
predictor, prompt_fn = build_large_model(hp)
dummy = np.zeros((416, 416, 3), dtype=np.uint8)
pts = np.array([[208, 208]], dtype=np.float32)
mask, score = prompt_fn(predictor, dummy, pts)
print("SAM2 OK — mask:", mask.shape, mask.dtype, "score:", score)
```
看到 mask 形狀正確、score 為合理 float（0~1）即代表 SAM2 後端接通。

### D.7 小結
- 切換成本：**一個工廠檔 `large_model.py` + `cmc_predict` 傳入 `prompt_fn` + cmc.yaml 加 `sam_backend`**，其餘不動。
- 用 `sam_backend: v1`（對論文）或 `sam2`（改良）即可一鍵切換，方便做 v1 vs SAM2 的對照實驗。
- 換 SAM2 後**唯一一定要做的調整是 `SAM_THRESH` 重新校準**（D.5）。
