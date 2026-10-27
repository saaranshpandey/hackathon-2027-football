from possession.pathcrf import load_model


def test_pathcrf_model_loads():
    model = load_model("cpu")
    assert model.macro_type == "poss_edge"
    assert getattr(model, "crf", None) is not None
