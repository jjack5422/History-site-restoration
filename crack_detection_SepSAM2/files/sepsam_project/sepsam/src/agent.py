"""
agent.py — SepSAM 的提示代理（YOLOv8-Seg + SEA）封裝。

注意：載入訓練好的權重前，環境必須已執行過 `python sea_setup.py`，
否則 checkpoint 內的 C2f_SEA 類別（位於 ultralytics.nn.tasks）無法 unpickle。
"""
import cv2
import numpy as np
from ultralytics import YOLO


class Agent:
    def __init__(self, ckpt, device="cuda"):
        self.model = YOLO(ckpt)
        self.device = device

    def predict(self, image_rgb, conf=0.0):
        """
        Returns:
            mask:      HxW uint8（0/255），所有 instance mask 的聯集
            conf_list: list[float]，各偵測 instance 的信心
        """
        r = self.model.predict(
            image_rgb, conf=conf, retina_masks=True, device=self.device, verbose=False
        )[0]
        h, w = image_rgb.shape[:2]

        if r.masks is None:
            return np.zeros((h, w), np.uint8), []

        m = r.masks.data.cpu().numpy()                  # (N, Hm, Wm)
        mask = (m.sum(0) > 0).astype(np.uint8) * 255
        if mask.shape != (h, w):                        # retina_masks 通常已對齊；保險
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        conf_list = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []
        return mask, conf_list
