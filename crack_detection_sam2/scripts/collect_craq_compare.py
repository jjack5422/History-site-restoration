"""Collect craquelure fold0 metrics from unet-style run dirs (log.json) into a
comparison table. best epoch = max val miou (= craquelure IoU, bg ignored).

Usage:
  python collect_craq_compare.py \
      --run unet=crack_detection_unet/runs/unet-craqbin1027-fold0-2026-06-29 \
      --run deeplabv3plus=.../deeplabv3plus-craqbin1027-fold0-2026-06-29 \
      --run segformer=.../segformer-craqbin1027-fold0-2026-06-29 \
      --out crack_detection_sam2/runs/craq-modelcompare-fold0-2026-06-29
Cited rows (e.g. SAM2-refine, different data version) added via --cite.
"""
import argparse, json, os


def best_craq(run_dir):
    log = json.load(open(os.path.join(run_dir, "log.json")))
    hist = log["history"]
    best = max((h for h in hist if h["val"] and h["val"].get("miou") == h["val"].get("miou")),
               key=lambda h: h["val"]["miou"])
    cq = best["val"]["per_class"]["craquelure"]
    return {
        "epoch": best["epoch"],
        "precision": cq["precision"], "recall": cq["recall"],
        "iou": cq["iou"], "f1": cq["f1"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="label=run_dir")
    ap.add_argument("--cite", action="append", default=[],
                    help="label=iou,prec,rec,f1,note (manually cited row)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for spec in args.run:
        label, rd = spec.split("=", 1)
        m = best_craq(rd)
        m["model"] = label
        m["run"] = rd
        m["note"] = f"fold0 honest val, best ep{m['epoch']}"
        rows.append(m)
    for spec in args.cite:
        label, rest = spec.split("=", 1)
        iou, prec, rec, f1, note = (rest.split(",", 4) + ["", "", "", "", ""])[:5]
        rows.append({"model": label, "iou": float(iou), "precision": float(prec),
                     "recall": float(rec), "f1": float(f1), "run": "(cited)", "note": note})

    json.dump(rows, open(os.path.join(args.out, "comparison.json"), "w"), indent=2)

    lines = ["# craquelure fold0 model comparison", "",
             "| model | precision | recall | IoU | F1 | run | note |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['precision']:.4f} | {r['recall']:.4f} | "
                     f"{r['iou']:.4f} | {r['f1']:.4f} | {r.get('run','')} | {r.get('note','')} |")
    md = "\n".join(lines) + "\n"
    open(os.path.join(args.out, "comparison.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
