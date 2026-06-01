"""oversample_dense_panels.py — Stage 3:依 panel 裂紋實例數過採樣某 fold 的 train list。

倍率公式:repeat = clip(round(panel_instances / median_panel_instances), 1, 3)
(panel_instances = 該 panel 所有 train tile 的 label 行數總和;median 取各 train panel 的中位數)

用法:
  python scripts/oversample_dense_panels.py --pool datasets/heritage_ft_cv_clahe --fold 0 \
      --configs_dir configs --yaml_prefix data_heritage_cv_os
"""
import argparse, os, statistics, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from build_heritage_cv import tile_to_panel  # noqa: E402


def repeat_factor(n, median):
    if median <= 0:
        return 1
    return int(max(1, min(3, round(n / median))))


def tile_instance_count(label_path):
    if not os.path.isfile(label_path):
        return 0
    with open(label_path) as f:
        return sum(1 for line in f if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--configs_dir", default="configs")
    ap.add_argument("--yaml_prefix", default="data_heritage_cv_os")
    args = ap.parse_args()

    pool = os.path.abspath(args.pool)
    lbl_dir = os.path.join(pool, "labels")
    train_txt = os.path.join(pool, f"fold{args.fold}", "train.txt")
    img_paths = [l.strip() for l in open(train_txt) if l.strip()]

    panel_inst, panel_tiles = {}, {}
    for ip in img_paths:
        stem = os.path.splitext(os.path.basename(ip))[0]
        panel = tile_to_panel(stem)
        n = tile_instance_count(os.path.join(lbl_dir, stem + ".txt"))
        panel_inst[panel] = panel_inst.get(panel, 0) + n
        panel_tiles.setdefault(panel, []).append(ip)

    if not panel_inst:
        sys.exit(f"train.txt 沒有任何 tile/panel: {train_txt}")
    med = statistics.median(panel_inst.values())
    out_lines = []
    for panel, tiles in panel_tiles.items():
        rep = repeat_factor(panel_inst[panel], med)
        out_lines += tiles * rep
        print(f"panel {panel}: inst={panel_inst[panel]} rep={rep} ({len(tiles)} tiles)")

    out_txt = os.path.join(pool, f"fold{args.fold}", "train_oversampled.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    val_txt = os.path.join(pool, f"fold{args.fold}", "val.txt")
    yaml_path = os.path.join(os.path.abspath(args.configs_dir),
                             f"{args.yaml_prefix}_fold{args.fold}.yaml")
    os.makedirs(os.path.abspath(args.configs_dir), exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"# oversampled train (Stage 3) fold {args.fold}, median_inst={med}\n"
                f"path: {pool}\ntrain: {out_txt}\nval: {val_txt}\nnc: 1\nnames:\n  - crack\n")
    print(f"median_inst={med}  train {len(img_paths)}->{len(out_lines)} tiles  yaml={yaml_path}")


if __name__ == "__main__":
    main()
