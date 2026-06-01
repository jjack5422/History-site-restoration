"""eval_promptsam2_craq.py — oracle(GT 點)+ YOLO-point(快取)兩模式,逐 fold 評 PromptedSAM2Seg。"""
import argparse, glob, json, os
import cv2, numpy as np, torch
from model_prompted_sam2 import PromptedSAM2Seg
from gt_points import gt_points

TILES = "data/labeled32_craq_v3/tiles_512"
DENSE = {0: 0.634, 1: 0.614, 2: 0.040, 3: 0.655}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def prf(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    tp = int((p & g).sum()); fp = int((p & ~g).sum()); fn = int((~p & g).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    f1 = 2 * pr * rc / max(pr + rc, 1e-8); iou = tp / max(tp + fp + fn, 1)
    if (tp + fp + fn) == 0:
        return 1.0, 1.0, 1.0, 1.0
    return pr, rc, f1, iou


def load_img(path, image_size):
    bgr = cv2.imread(path); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (image_size, image_size))
    t = ((r / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    return torch.from_numpy(t[None].astype(np.float32)), rgb.shape[:2]


@torch.no_grad()
def predict(model, img_t, pts, image_size, device, orig_hw):
    h, w = orig_hw
    if len(pts) == 0:
        return np.zeros((h, w), np.uint8)
    sx, sy = image_size / w, image_size / h
    pc = np.array([[x * sx, y * sy] for x, y in pts], np.float32)[None]
    pl = np.ones((1, pc.shape[1]), np.int64)
    logits = model(img_t.to(device), torch.from_numpy(pc).to(device), torch.from_numpy(pl).to(device))
    m = (logits.squeeze().cpu().numpy() > 0).astype(np.uint8)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def eval_fold(k, mode, image_size, n_points, device):
    model = PromptedSAM2Seg(variant="small", image_size=image_size, device=device).to(device)
    ck = torch.load(f"outputs/promptsam2_craq_fold{k}/best.pt", map_location=device)
    model.load_state_dict(ck["model"]); model.eval()
    vi = os.path.join(TILES, f"craqfold{k}", "val_images")
    vm = os.path.join(TILES, f"craqfold{k}", "val_masks")
    yp = json.load(open(os.path.join(TILES, f"craqfold{k}", "yolo_points.json"))) if mode == "yolo" else None
    recs = []
    for p in sorted(glob.glob(os.path.join(vi, "*.png"))):
        st = os.path.splitext(os.path.basename(p))[0]
        gt = cv2.imread(os.path.join(vm, st + ".png"), 0)
        img_t, hw = load_img(p, image_size)
        if mode == "oracle":
            pts, _ = gt_points(gt > 0, n_points); pts = pts.tolist()
        else:
            pts = yp.get(st, [])
        pred = predict(model, img_t, pts, image_size, device, hw)
        recs.append(prf(pred, gt > 0))
    a = np.array(recs, float).mean(0)
    return {"P": a[0], "R": a[1], "F1": a[2], "IoU": a[3], "n": len(recs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--n_points", type=int, default=10)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for mode in ["oracle", "yolo"]:
        print(f"\n==== mode={mode} ====")
        rows = []
        for k in args.folds:
            r = eval_fold(k, mode, args.image_size, args.n_points, device)
            rows.append((k, r))
            print(f"fold{k} n={r['n']} F1={r['F1']:.3f} P={r['P']:.3f} R={r['R']:.3f} IoU={r['IoU']:.3f} | dense-seg={DENSE[k]}")
        no2 = [r for kk, r in rows if kk != 2]
        m4 = sum(r["F1"] for _, r in rows) / len(rows)
        mno2 = sum(r["F1"] for r in no2) / max(len(no2), 1)
        print(f"mean F1 4-fold={m4:.3f}  排除fold2={mno2:.3f}  (dense-seg 0.634 / SepSAM-YOLO 0.541)")


if __name__ == "__main__":
    main()
