import numpy as np

from possession.external import ALL_MATCHES, TEST, TRAIN, VAL, labels_from_state


def test_labels_from_state():
    alive = np.array([0, 1, 1, 1], dtype=bool)
    owner = np.array([1, 0, 1, -1], dtype=np.int8)
    # dead -> D (2), home -> H (0), away -> A (1), alive with no owner -> X (3)
    assert labels_from_state(alive, owner).tolist() == [2, 0, 1, 3]


def test_split_is_pathcrf_order():
    assert ALL_MATCHES == sorted(TRAIN + VAL + TEST)
    assert TRAIN == ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"]
    assert VAL == ["J03WQQ"]
    assert TEST == ["J03WR9"]
