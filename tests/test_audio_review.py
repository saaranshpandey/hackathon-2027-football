from possession.audio_review import merge_candidates, window_starts, inspection_windows


def test_windows_cover_tail_and_short_audio():
    assert window_starts(27,10,5)==[0,5,10,15,17]
    assert window_starts(3,10,5)==[0]
    assert window_starts(20,10,5)==[0,5,10]


def test_merge_uses_audio_scores_only_and_keeps_separate_events():
    rows=[dict(start_seconds=a,end_seconds=a+10,whistle_score=s) for a,s in [(0,.3),(5,.5),(10,.1),(20,.7)]]
    assert merge_candidates(rows,.2)==[
        dict(start_seconds=0,end_seconds=15,peak_score=.5,windows=2),
        dict(start_seconds=20,end_seconds=30,peak_score=.7,windows=1)]


def test_no_whistle_on_zero_scores():
    assert merge_candidates([dict(start_seconds=0,end_seconds=10,whistle_score=0)],.2)==[]


def test_inspection_ranking_is_not_thresholded_detection():
    rows=[dict(start_seconds=a,end_seconds=a+10,whistle_score=s) for a,s in [(0,.001),(5,.01),(20,.002)]]
    ranked=inspection_windows(rows)
    assert [r['start_seconds'] for r in ranked]==[5,20]
    assert merge_candidates(rows,.2)==[]
