import numpy as np
import pytest

from possession.boundary import Config, adjust, evidence, signed_distance
from possession.compare_boundary import perturb
from tests.test_nn import make


def track(n=100):
    t=make([20,30],[70,80],n=n)
    t.ball[:]=[50,34]
    return t


def test_signed_distances_all_edges():
    b=np.array([[0,34],[105,34],[50,0],[50,68],[-1,34],[106,34],[50,-1],[50,69],[50,34],[np.nan,0]])
    np.testing.assert_allclose(signed_distance(b,(105,68)),[0,0,0,0,-1,-1,-1,-1,34,np.nan],equal_nan=True)


def test_sustained_outside_boosts_dead_only_and_does_not_mutate():
    t=track();t.ball[:]=[-1,34]
    d=np.full(100,.2,np.float32);h=np.full(100,.7,np.float32)
    od,oh,_=adjust(t,d,h)
    assert (od>d).all() and (od<=1).all()
    np.testing.assert_array_equal(oh,h)
    np.testing.assert_array_equal(d,np.full(100,.2,np.float32))


def test_missing_breaks_evidence_and_does_not_carry_forward():
    t=track();t.ball[:]=[-1,34];t.ball[30:60]=np.nan
    _,s=evidence(t)
    assert (s[:30]>0).all() and (s[60:]>0).all()
    assert (s[30:60]==0).all()


def test_short_spikes_near_line_and_gross_outliers_ignored():
    t=track();t.ball[:]=[-.2,34];t.ball[20:25]=[-1,34];t.ball[50:80]=[-50,34]
    assert not evidence(t)[1].any()


def test_evidence_never_joins_halves_or_outside_periods():
    t=track(20);t.ball[:]=[-1,34];t.period[:7]=1;t.period[7:13]=0;t.period[13:]=2
    assert not evidence(t)[1].any()
    t.period[:10]=1;t.period[10:]=2
    assert (evidence(t)[1]>0).all()  # each half independently long enough


def test_inside_does_not_force_live():
    t=track();d=np.full(100,.95);h=np.full(100,.3)
    od,oh,_=adjust(t,d,h)
    np.testing.assert_array_equal(od,d)
    np.testing.assert_array_equal(oh,h)


def test_noise_preserves_gaps_and_is_reproducible():
    t=track();t.ball[10]=np.nan
    a=perturb(t,4,sigma=.5,bias=.75,jump_fraction=.1)
    b=perturb(t,4,sigma=.5,bias=.75,jump_fraction=.1)
    np.testing.assert_array_equal(a.ball,b.ball)
    assert np.isnan(a.ball[10]).all()
    np.testing.assert_array_equal(t.ball[0],[50,34])


def test_invalid_config():
    with pytest.raises(ValueError):Config(min_frames=0)
