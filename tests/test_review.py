import numpy as np
import pandas as pd
import pytest

from possession.review import candidates, read_labels, spans, public_truth


def test_spans_cover_every_frame():
    assert spans(np.array(list('HHDDAX'))) == [[0, 2, 'H'], [2, 4, 'D'], [4, 5, 'A'], [5, 6, 'X']]


def test_review_rejects_misaligned_submission(tmp_path):
    path = tmp_path / 'labels.csv'
    pd.DataFrame({'frame': [1, 2], 'label': ['H', 'D']}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='alignment'):
        read_labels(path, 2)


def test_gap_candidates_split_at_halves_and_ignore_outside():
    n = 400
    period = np.r_[np.ones(150), np.zeros(100), np.full(150, 2)]
    labels = np.where(period > 0, 'H', 'X')
    result = candidates(labels, labels.copy(), np.ones(n, bool), period)
    gaps = [r for r in result if r['kind'] == 'Ball missing']
    assert [(r['frame'], r['end']) for r in gaps] == [(0, 150), (250, 400)]


def test_disagreement_flag_uses_frozen_model_outputs():
    labels = np.full(100, 'H')
    baseline = labels.copy()
    baseline[25:75] = 'A'
    result = candidates(labels, baseline, np.zeros(100, bool), np.ones(100))
    assert len(result) == 1
    assert result[0]['kind'] == 'Models disagree'
    assert result[0]['frame'] == 25 and result[0]['end'] == 75


def test_public_truth_does_not_treat_ungraded_frames_as_labels(tmp_path):
    path = tmp_path / 'public.csv'
    pd.DataFrame({'frame': range(6), 'label': list('XHHDXX')}).to_csv(path, index=False)
    assert public_truth(path, 6) == [[1, 3, 'H'], [3, 4, 'D']]
