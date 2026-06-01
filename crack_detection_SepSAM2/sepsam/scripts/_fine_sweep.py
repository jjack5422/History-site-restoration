"""Fine-grain sweep around the coarse best (SAM_THRESH, CONFLICTION_RATIO).
Uses same cache pattern as calibrate_sam_thresh.py but with denser grids."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, glob, cv2, numpy as np, yaml
from types import SimpleNamespace
from src.agent import Agent
from src.cmc import conflict_ratio
from src.filters import contour_filter
from src.geometry import mask_to_points_and_width
from src.large_model import build_large_model
from src.metrics import prf_iou

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def index_by_stem(folder):
    out = {}
    for p in glob.glob(os.path.join(folder, "*")):
        if os.path.splitext(p)[1].lower() in IMG_EXTS:
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/cmc.yaml")
ap.add_argument("--images", required=True)
ap.add_argument("--masks", required=True)
ap.add_argument("--limit", type=int, default=100)
args = ap.parse_args()

hp = SimpleNamespace(**yaml.safe_load(open(args.config, encoding="utf-8")))
from ultralytics.nn.tasks import C2f_SEA  # noqa: F401
agent = Agent(hp.agent_ckpt, device=hp.device)
predictor, prompt_fn = build_large_model(hp)
imgs = index_by_stem(args.images); gts = index_by_stem(args.masks)
stems = sorted(set(imgs) & set(gts))[: args.limit]
cache = []
for stem in stems:
    bgr = cv2.imread(imgs[stem]); gt = cv2.imread(gts[stem], cv2.IMREAD_GRAYSCALE)
    if bgr is None or gt is None: continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask_yolo, yolo_conf = agent.predict(rgb, conf=hp.YOLO_CONF_1)
    n = max(rgb.shape[:2]) // hp.POINTS_DIVISOR
    pts, _ = mask_to_points_and_width(mask_yolo > 0, n)
    m_raw, score = prompt_fn(predictor, rgb, pts)
    m_sam = contour_filter(m_raw, yolo_conf, hp.YOLO_CONF_2)
    conf = conflict_ratio(m_sam, mask_yolo)
    cache.append((score, conf, m_sam, mask_yolo, gt > 0))

print(f"cached {len(cache)} samples")

thrs  = np.round(np.arange(0.50, 0.91, 0.025), 3)
confs = np.round(np.arange(0.50, 2.05, 0.10), 3)

# print top-10 globally
records = []
for cr in confs:
    for thr in thrs:
        f1s, ious = [], []
        for score, c, m_sam, m_yolo, gt in cache:
            chosen = m_sam if (c < cr and score > thr) else m_yolo
            P,R,F,I = prf_iou(chosen, gt)
            f1s.append(F); ious.append(I)
        records.append((float(thr), float(cr), float(np.mean(f1s)), float(np.mean(ious))))

records.sort(key=lambda r: -r[2])
print("\nTop-10 (thr, conf_ratio, F1, IoU):")
for thr, cr, F, I in records[:10]:
    print(f"  thr={thr:.3f}  conf={cr:.2f}  F1={F:.4f}  IoU={I:.4f}")

# also report "YOLO-only baseline" (force chosen = m_yolo) and "SAM-only-when-available"
f1_yolo = float(np.mean([prf_iou(m_yolo, gt)[2] for _,_,_,m_yolo,gt in cache]))
f1_sam  = float(np.mean([prf_iou(m_sam, gt)[2]  for _,_,m_sam,_,gt in cache]))
print(f"\nReference: YOLO-only F1={f1_yolo:.4f}  SAM-only F1={f1_sam:.4f}")
