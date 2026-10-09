import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
from zimortal.training.active_schedule_v2 import priority


def test_promotions_prefer_measured_separation_over_noisy_tie(tmp_path):
    (tmp_path / "candidates").mkdir()
    (tmp_path / "labels").mkdir()
    configuration = {"quotas": {}, "validation": 500, "adaptive": {"minimum": 128}}
    current = {"training_strata": {}, "qualified": {}}
    scores = []
    for key, gap, radius in [("separated", 100, 20), ("uncertain", 0, 2000)]:
        intervals = np.zeros((2, 2, 2))
        intervals[0, 1] = [gap - radius, gap + radius]
        np.savez(
            tmp_path / "labels" / f"{key}.npz", q=np.array([gap / 100, 0]), intervals=intervals
        )
        (tmp_path / "labels" / f"{key}.json").write_text(
            json.dumps({"trace": [{"resolved": False}]})
        )
        row = (
            key,
            str(tmp_path / "candidates" / f"{key}.json"),
            "deferred",
            10,
            1024,
            "[]",
            2,
            "train",
        )
        scores.append(priority(row, current, configuration))
    assert scores[0] > scores[1] + 15
