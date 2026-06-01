"""
cmc.py — Cyclic Model Conversation 主流程（四輪）。
- 1st: Agent 草稿 → 沿中軸取提示點（geometry）
- 2nd: SAM 提示推論（prompt_fn，由 large_model 提供，後端無關）
- 2nd+3rd: contour 過濾（filters）
- 4th: 衝突分析 + SAM 信心門檻 → 決策（接受 SAM 精修 / 回退 YOLO 草稿）
對應論文 Eq. 11 與 Pseudo code 2。
"""
import numpy as np

from .geometry import mask_to_points_and_width
from .filters import contour_filter


def conflict_ratio(mc, ms):
    """
    論文 Eq. 11：Σ(Mc − Ms⊗Mc) / Σ Ms
    = |Mc 中不與草稿 Ms 重疊的部分| / |Ms|
    Args:
        mc: 3rd round 過濾後的 SAM mask
        ms: Agent 初始草稿
    """
    mc_b = np.asarray(mc).astype(bool)
    ms_b = np.asarray(ms).astype(bool)
    inter = int(np.logical_and(mc_b, ms_b).sum())
    mc_only = int(mc_b.sum()) - inter
    denom = max(int(ms_b.sum()), 1)
    return mc_only / denom


def cmc_predict(image_rgb, agent, predictor, prompt_fn, hp, return_intermediates=False):
    """
    執行完整 CMC，回傳最終裂紋 mask（HxW uint8 0/255）。

    Args:
        image_rgb:  HxWx3 RGB（uint8）
        agent:      Agent 實例（src.agent.Agent），.predict(img, conf) -> (mask, conf_list)
        predictor:  大模型 predictor（由 build_large_model 建立）
        prompt_fn:  提示函式（sam_prompt_v1 或 sam_prompt_sam2），介面相同
        hp:         超參數（SimpleNamespace，來自 cmc.yaml）
        return_intermediates: True 時額外回傳各階段中間結果（供可視化）

    Returns:
        final_mask  或  (final_mask, info_dict)
    """
    # 1st round: Agent 草稿 + 各 instance 信心
    mask_yolo, yolo_conf = agent.predict(image_rgb, conf=hp.YOLO_CONF_1)

    # 1st round: 沿中軸取提示點
    n_pts = max(image_rgb.shape[:2]) // hp.POINTS_DIVISOR
    pts, widths = mask_to_points_and_width(mask_yolo > 0, n_pts)

    # 2nd round: SAM 提示（後端無關）
    mask_sam_raw, sam_score = prompt_fn(predictor, image_rgb, pts)

    # 2nd + 3rd round: border following + 依 Agent 信心過濾
    mask_sam = contour_filter(mask_sam_raw, yolo_conf, hp.YOLO_CONF_2)

    # 4th round: 衝突分析 + 信心門檻
    conflict = conflict_ratio(mask_sam, mask_yolo)
    accept = (conflict < hp.CONFLICTION_RATIO) and (sam_score > hp.SAM_THRESH)
    final = mask_sam if accept else mask_yolo

    if return_intermediates:
        info = {
            "draft": mask_yolo,
            "points": pts,
            "widths": widths,
            "sam_raw": mask_sam_raw,
            "sam_filtered": mask_sam,
            "sam_score": float(sam_score),
            "conflict": float(conflict),
            "decision": "sam" if accept else "yolo",
        }
        return final, info
    return final
