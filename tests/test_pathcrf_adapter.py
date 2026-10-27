import numpy as np
import pandas as pd

from possession.pathcrf import base_table, team_probability
from possession.tracking import Tracking


def test_team_probability_mapping():
    micro_pred = pd.DataFrame({"edge_src": ["home_3", "away_1", "out_left"],
                               "edge_dst": ["home_3", "home_2", "out_left"]},
                              index=pd.Index([2, 3, 4], name="frame_id"))
    p = team_probability(micro_pred, n_frames=6)
    np.testing.assert_allclose(p, [0.5, 0.5, 0.95, 0.05, 0.5, 0.5])


def test_base_table_frames_align():
    period = np.array([0, 1, 1, 1, 0, 2, 2, 2, 2, 0], dtype=np.int8)
    n = len(period)
    players = np.arange(n * 4 * 2, dtype=np.float32).reshape(n, 4, 2)
    trk = Tracking(ball=np.zeros((n, 2), dtype=np.float32), players=players,
                   team=np.array([0, 0, 1, 1], dtype=np.int8), period=period, pitch=(105.0, 68.0))
    alive = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0], dtype=bool)

    table = base_table(trk, alive)

    assert table["frame_id"].tolist() == [1, 2, 3, 5, 6, 7, 8]
    np.testing.assert_allclose(table["timestamp"], [0, 0.04, 0.08, 0, 0.04, 0.08, 0.12])
    assert table["period_id"].tolist() == [1, 1, 1, 2, 2, 2, 2]
    assert table["ball_state"].tolist() == ["alive", "alive", "dead", "alive", "alive", "alive", "dead"]
    # slot 2 is the first away slot -> away_1; slot 1 is the second home slot -> home_2
    assert table["away_1_x"].tolist() == players[[1, 2, 3, 5, 6, 7, 8], 2, 0].tolist()
    assert table["home_2_y"].tolist() == players[[1, 2, 3, 5, 6, 7, 8], 1, 1].tolist()
