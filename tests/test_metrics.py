import numpy as np

from cardet.data import parse_locations, write_locations
from cardet.metrics import average_precision, score_detections, sweep_thresholds


def test_single_scale_ellipse_criterion():
    truth = {0: [(50, 100)]}
    # axes: 10 px vertically (25% of 40), 25 px horizontally (25% of 100)
    assert score_detections(truth, {0: [(60, 100)]}).tp == 1
    assert score_detections(truth, {0: [(50, 125)]}).tp == 1
    assert score_detections(truth, {0: [(61, 100)]}).tp == 0
    assert score_detections(truth, {0: [(58, 118)]}).tp == 0  # 0.64 + 0.5184 > 1


def test_one_detection_per_object():
    s = score_detections({0: [(10, 10)]}, {0: [(10, 10), (11, 12)]})
    assert (s.tp, s.fp, s.n_objects) == (1, 1, 1)
    assert s.precision == 0.5 and s.recall == 1.0


def test_missed_objects_and_empty_images():
    s = score_detections({0: [(10, 10)], 1: [(5, 5), (5, 200)]}, {0: [], 1: [(5, 5)]})
    assert (s.tp, s.fp, s.n_objects) == (1, 0, 3)
    assert round(s.f1, 4) == round(2 * 1 * (1 / 3) / (1 + 1 / 3), 4)


def test_multiscale_criterion_checks_scale():
    truth = {0: [(50, 50, 120)]}
    assert score_detections(truth, {0: [(50, 50, 120)]}, multiscale=True).tp == 1
    assert score_detections(truth, {0: [(50, 50, 140)]}, multiscale=True).tp == 1  # 20 < 30
    assert score_detections(truth, {0: [(50, 50, 160)]}, multiscale=True).tp == 0  # wrong scale


def test_sweep_matches_direct_scoring():
    truth = {0: [(10, 10)], 1: [(20, 30)]}
    scored = {0: [((10, 10), 0.9), ((40, 80), 0.6)], 1: [((20, 31), 0.4)]}
    rows = sweep_thresholds(truth, scored, False, [0.3, 0.5, 0.95])
    assert [(r["tp"], r["fp"]) for r in rows] == [(2, 1), (1, 1), (0, 0)]


def test_average_precision_perfect_detector():
    rows = [{"recall": 1.0, "precision": 1.0}]
    assert np.isclose(average_precision(rows), 1.0)


def test_locations_roundtrip(tmp_path):
    p = tmp_path / "found.txt"
    write_locations(p, {0: [(1, 2)], 2: [(3, 4), (-5, 6)]}, n_images=3)
    assert p.read_text() == "0: (1,2)\n1:\n2: (3,4) (-5,6)\n"
    assert parse_locations(p) == {0: [(1, 2)], 1: [], 2: [(3, 4), (-5, 6)]}


def test_parse_official_truth_files(params):
    single = parse_locations(params.path("true_locations"))
    scale = parse_locations(params.path("true_locations_scale"))
    assert len(single) == 170 and sum(map(len, single.values())) == 200
    assert len(scale) == 108 and sum(map(len, scale.values())) == 139
    assert all(len(t) == 3 for v in scale.values() for t in v)
