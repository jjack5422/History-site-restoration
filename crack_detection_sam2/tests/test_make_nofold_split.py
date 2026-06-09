import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from make_nofold_split import build_nofold


def test_build_nofold_all_tiles_train_and_val():
    idx = {"items": [{"tile": "a.png"}, {"tile": "b.png"}, {"tile": "c.png"}]}
    out = build_nofold(idx)
    assert len(out["folds"]) == 1
    f = out["folds"][0]
    assert f["fold"] == 0
    assert f["train"] == ["a.png", "b.png", "c.png"]
    assert f["val"] == f["train"]          # final-expert style: eval on all, keep last.pt
    print("OK test_make_nofold_split")


if __name__ == "__main__":
    test_build_nofold_all_tiles_train_and_val()
