import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.oversample_dense_panels import repeat_factor


def test_repeat_factor_formula():
    # 公式 clip(round(n/median),1,3);Python round 用 banker's rounding
    assert repeat_factor(50, 100) == 1     # round(0.5)=0 -> clip ->1
    assert repeat_factor(100, 100) == 1    # round(1.0)=1
    assert repeat_factor(150, 100) == 2    # round(1.5)=2
    assert repeat_factor(250, 100) == 2    # round(2.5)=2 (banker's)
    assert repeat_factor(350, 100) == 3    # round(3.5)=4 -> clip ->3
    assert repeat_factor(1000, 100) == 3   # clip 上限 3


if __name__ == "__main__":
    test_repeat_factor_formula()
    print("OK test_oversample")
