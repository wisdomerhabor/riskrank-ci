import numpy as np
import pytest

from riskrank.evaluate import holm, paired_test, rank_biserial


def test_nine_same_sign_pairs_reach_the_section_3_10_4_floor():
    # 2/2^9 = 0.0039 < 0.05/8 = 0.00625 (section 3.10.4).
    sys_v = np.linspace(0.3, 0.5, 9)
    res = paired_test(sys_v, sys_v - 0.1)
    assert res["p"] == pytest.approx(2 / 2**9)
    assert res["p"] < 0.05 / 8


def test_eight_pairs_cannot_pass_first_holm_step():
    sys_v = np.linspace(0.3, 0.5, 8)
    assert paired_test(sys_v, sys_v - 0.1)["p"] == pytest.approx(2 / 2**8)  # 0.0078 > 0.00625


def test_zero_differences_dropped_and_counted():
    res = paired_test(np.array([0.5, 0.4, 0.3]), np.array([0.5, 0.3, 0.2]))
    assert res["zero_pairs_dropped"] == 1 and res["n_nonzero"] == 2


def test_rank_biserial_kerby():
    # Differences +3, +2, -1: ranks 3, 2, 1 -> (5 - 1) / 6 = 2/3.
    assert rank_biserial(np.array([3.0, 2.0, -1.0])) == pytest.approx(2 / 3)
    assert rank_biserial(np.array([1.0, -1.0])) == 0
    # Two-project corpus takes only -1, -1/3, 0, 1/3, 1 (section 3.10.4).
    assert rank_biserial(np.array([2.0, -1.0])) == pytest.approx(1 / 3)


def test_holm_is_monotone_and_matches_hand_computation():
    adj = holm([0.01, 0.04, 0.03, 0.005])
    # sorted 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1 -> max(0.06,0.04)=0.06
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
