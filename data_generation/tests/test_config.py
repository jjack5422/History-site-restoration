from synthgen.config import load_config


def test_load_config_has_required_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "seed: 42\n"
        "tile_size: 512\nstride: 256\nslice_size: 1024\n"
        "ratio: {crack: 8, craq: 1}\n"
        "target_tiles: {crack: 800, craq: 100}\n"
        "base: {manifest: data/base_manifest.json, allow_reuse: true, max_reuse: 20}\n"
        "crack: {n_curves: [2,6], taper_alpha: 2.0, taper_sigma: 0.5, branch_p: [0.3,0.5], target_fg: [0.05,2.0]}\n"
        "craq: {cell_px: [32,60], jitter: 0.3, edge_w: 1, break_p: 0.1, target_fg: [0.5,9.0]}\n"
        "appearance: {profile_dir: data/appearance, min_contrast: 12, erosion: 2, blur_sigma: 2}\n"
        "keep_negative_ratio: 0.1\nmax_regen_fail: 50\n"
    )
    cfg = load_config(str(p))
    assert cfg["seed"] == 42
    assert cfg["target_tiles"]["crack"] == 800
    assert cfg["crack"]["taper_alpha"] == 2.0
