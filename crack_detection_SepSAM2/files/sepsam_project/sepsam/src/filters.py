"""
filters.py — 用 Agent 領域知識過濾 SAM 雜訊（CMC 2nd + 3rd round）。
對應論文 Pseudo code 1：border following 取 contour，依面積排序，
保留與「YOLO 高信心偵測數 k」相同數量的前 k 大 contour。
"""
import cv2
import numpy as np


def contour_filter(mask, yolo_conf, conf_thresh=0.5):
    """
    Args:
        mask:       SAM raw mask，HxW uint8（0/255）
        yolo_conf:  Agent 偵測各 instance 的信心 list
        conf_thresh: 高信心門檻（= YOLO_CONF_2）

    Returns:
        只保留前 top_n 大面積 contour 的 mask（uint8 0/255）。
        top_n = 信心 > conf_thresh 的 YOLO 偵測數。
    """
    mask = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)

    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    top_n = int(np.sum(np.asarray(yolo_conf, dtype=float) > conf_thresh))
    top_n = max(top_n, 0)

    out = np.zeros_like(mask)
    if top_n > 0:
        # 註：若 top_n == 0（沒有任何高信心 YOLO 偵測），依論文邏輯會得到空 mask；
        #     後續衝突分析會因此回退到 YOLO 草稿。此為忠實於 Pseudo code 1 的行為。
        cv2.drawContours(out, contours_sorted[:top_n], -1, 255, thickness=cv2.FILLED)
    return out
