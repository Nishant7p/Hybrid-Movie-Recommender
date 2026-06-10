import json
from pathlib import Path
from cf_pipeline.utils.io import save_result, load_result


def test_save_and_load_round_trip(tmp_path):
    payload = {"experiment": "pop", "metrics": {"HR@10": 0.5}}
    out = tmp_path / "x.json"
    save_result(payload, out)
    loaded = load_result(out)
    assert loaded["experiment"] == "pop"
    assert "git_sha" in loaded
    assert "timestamp" in loaded
