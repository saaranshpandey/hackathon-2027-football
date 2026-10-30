"""Transformer dead-ball head with the existing DBC possession head."""


def combine(t, db, c):
    """Keep T's dead probability and average DB/C home-given-live probabilities."""
    return t[0], (db[1] + c[1]) / 2


def approach_t_dbc(trk, device='cpu'):
    from possession.nn import approach_c
    from possession.pathcrf import approach_db
    from possession.transformer import approach_t

    return combine(approach_t(trk, device=device), approach_db(trk, device=device), approach_c(trk))
