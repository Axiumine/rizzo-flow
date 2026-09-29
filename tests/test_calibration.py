"""Temperature fitting (calibration.py): input checks, the search grid and the saved file."""

import json
import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from test_core_support import exactly

from rizzo_flow.calibration import Calibration, LabeledLogits, fit_temperature

FIXED_ROWS = [{"type": "choice", "logits": [0.0, 1.0], "label_index": i % 2} for i in range(10)]
FIXED_ROWS_SHA256 = "1f5f05a9a412e521fe4ae1b407d1b8cc3369e10327e62f51e76e447bfe2019e0"


def rows(kind: str, logits: list[float], labels: list[int]) -> list[dict[str, Any]]:
    return [{"type": kind, "logits": list(logits), "label_index": label} for label in labels]


def nll(logits: list[float], labels: list[int], temperature: float) -> float:
    """Mean negative log-likelihood of the labels under softmax(logits / temperature)."""
    total = 0.0
    for label in labels:
        values = [x / temperature for x in logits]
        total += math.log(sum(math.exp(v) for v in values)) - values[label]
    return total / len(labels)


def calibrated_rows() -> list[dict[str, Any]]:
    """Logits [0, ln 3] with the label true 30 times out of 40: T = 1 is exactly right."""
    return rows("choice", [0.0, math.log(3)], [1] * 30 + [0] * 10)


# input checks ---------------------------------------------------------------------------------


def test_fitting_needs_rows():
    with pytest.raises(ValueError, match=exactly("Calibration requires labeled rows")):
        fit_temperature([], "fp")


def test_ten_examples_per_type_are_the_minimum():
    fit_temperature(rows("choice", [0.0, 1.0], [0, 1] * 5), "fp")
    with pytest.raises(
        ValueError, match=exactly("Provide at least 10 calibration examples for choice")
    ):
        fit_temperature(rows("choice", [0.0, 1.0], [0, 1] * 4 + [0]), "fp")


def test_every_type_needs_its_own_ten_examples():
    enough = rows("choice", [0.0, 1.0], [0, 1] * 5)
    few = rows("boolean", [0.0, 1.0], [0, 1] * 4 + [1])
    with pytest.raises(
        ValueError, match=exactly("Provide at least 10 calibration examples for boolean")
    ):
        fit_temperature(enough + few, "fp")
    with pytest.raises(
        ValueError, match=exactly("Provide at least 10 calibration examples for boolean")
    ):
        fit_temperature(few + enough, "fp")


def test_the_label_must_index_a_candidate():
    good = rows("choice", [0.0, 1.0, 2.0], [0, 1, 2] * 4)
    fit_temperature(good, "fp")
    bad = good + rows("choice", [0.0, 1.0, 2.0], [3])
    with pytest.raises(ValueError, match=exactly("Label index is outside the candidate list")):
        fit_temperature(bad, "fp")


@pytest.mark.parametrize("logits", [[0.0, math.nan], [math.inf, 0.0], [0.0, -math.inf]])
def test_logits_must_be_finite(logits):
    bad = rows("choice", [0.0, 1.0], [0, 1] * 5) + rows("choice", logits, [0])
    with pytest.raises(ValueError, match=exactly("At least two finite logits are required")):
        fit_temperature(bad, "fp")


@pytest.mark.parametrize(
    "row",
    [
        {"type": "choice", "logits": [0.0], "label_index": 0},
        {"type": "choice", "logits": [0.0] * 27, "label_index": 0},
        {"type": "choice", "logits": [0.0, 1.0], "label_index": -1},
        {"type": "choice", "logits": [0.0, 1.0], "label_index": 1.0},
        {"type": "choice", "logits": [0.0, 1.0], "label_index": "1"},
        {"type": "text", "logits": [0.0, 1.0], "label_index": 0},
        {"type": "choice", "logits": [0.0, 1.0], "label_index": 0, "note": "x"},
        {"type": "choice", "logits": [0.0, 1.0]},
    ],
)
def test_rows_follow_the_labeled_logits_schema(row):
    with pytest.raises(ValidationError):
        fit_temperature([row], "fp")


def test_labeled_logits_accept_two_to_26_candidates():
    widest = LabeledLogits.model_validate(
        {"type": "score", "logits": [0.0] * 26, "label_index": 25}
    )
    assert (widest.type, len(widest.logits), widest.label_index) == ("score", 26, 25)
    narrowest = LabeledLogits.model_validate(
        {"type": "numeric", "logits": [0.0, 1.0], "label_index": 0}
    )
    assert (narrowest.type, narrowest.logits, narrowest.label_index) == ("numeric", [0.0, 1.0], 0)


# the fit ----------------------------------------------------------------------------------------


def test_a_calibrated_model_keeps_temperature_one():
    fit = fit_temperature(calibrated_rows(), "fp")
    assert fit.temperatures == pytest.approx({"choice": 1.0}, rel=1e-12)
    metrics = fit.fit_metrics["choice"]
    expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert metrics["rows"] == 40
    assert metrics["fit_nll_before"] == pytest.approx(expected)
    assert metrics["fit_nll_after"] == pytest.approx(metrics["fit_nll_before"], rel=1e-12)


def test_an_overconfident_model_is_cooled_down_to_the_top_of_the_grid():
    labels = [1, 0] * 10  # right half of the time despite a margin of 8 logits
    fit = fit_temperature(rows("choice", [0.0, 8.0], labels), "fp")
    assert fit.temperatures["choice"] == pytest.approx(20.0)  # 0.05 * 400
    metrics = fit.fit_metrics["choice"]
    assert metrics["fit_nll_before"] == pytest.approx(nll([0.0, 8.0], labels, 1.0))
    assert metrics["fit_nll_after"] == pytest.approx(nll([0.0, 8.0], labels, 20.0))
    assert metrics["fit_nll_after"] < metrics["fit_nll_before"] / 2


def test_an_underconfident_model_is_sharpened_down_to_the_bottom_of_the_grid():
    labels = [1] * 12
    fit = fit_temperature(rows("choice", [0.0, 0.5], labels), "fp")
    assert fit.temperatures["choice"] == pytest.approx(0.05)
    metrics = fit.fit_metrics["choice"]
    assert metrics["fit_nll_after"] == pytest.approx(nll([0.0, 0.5], labels, 0.05))
    assert metrics["fit_nll_after"] < 1e-4 < metrics["fit_nll_before"]


def test_the_search_finds_the_optimum_to_within_the_grid_step():
    # With logits [0, 4] and the label true 4 times in 5, the optimum is T = 4 / ln 4.
    labels = [1] * 40 + [0] * 10
    fit = fit_temperature(rows("choice", [0.0, 4.0], labels), "fp")
    optimum = 4 / math.log(4)
    assert fit.temperatures["choice"] == pytest.approx(optimum, rel=0.02)
    minimum = -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
    assert fit.fit_metrics["choice"]["fit_nll_after"] == pytest.approx(minimum, abs=1e-3)


def test_only_points_of_the_search_grid_can_be_chosen():
    # Logits [0, 4] with the label true 13 times in 16: the optimum is T = 4 / ln(13/3) = 2.728.
    # Point 160 of the grid, 0.05 * 400 ** (160 / 240) = 2.7144, is the best point of it, although
    # e = 2.7183 would fit slightly better: the search may return nothing but a grid point.
    labels = [1] * 13 + [0] * 3
    grid = [0.05 * 400 ** (i / 240) for i in range(241)]
    best = min(grid, key=lambda t: nll([0.0, 4.0], labels, t))
    assert best == pytest.approx(2.7144, abs=1e-4)
    assert nll([0.0, 4.0], labels, math.e) < nll([0.0, 4.0], labels, best)
    fit = fit_temperature(rows("choice", [0.0, 4.0], labels), "fp")
    assert fit.temperatures["choice"] == pytest.approx(best)


def test_each_type_gets_its_own_temperature_and_metrics():
    data = calibrated_rows() + rows("score", [0.0, 8.0, 0.0], [1, 0] * 10)
    fit = fit_temperature(data, "fp")
    assert set(fit.temperatures) == set(fit.fit_metrics) == {"choice", "score"}
    assert fit.temperatures["choice"] == 1.0
    assert fit.temperatures["score"] > 2
    assert fit.fit_metrics["choice"]["rows"] == 40
    assert fit.fit_metrics["score"]["rows"] == 20


def test_the_calibration_records_the_fingerprint_dataset_hash_and_status():
    fit = fit_temperature(FIXED_ROWS, "model-x")
    assert fit.fingerprint == "model-x"
    assert fit.dataset_sha256 == FIXED_ROWS_SHA256
    assert fit.version == 1
    assert fit.status == "fitted_requires_held_out_validation"


def test_the_dataset_hash_ignores_key_order_but_not_content():
    reordered = [dict(reversed(list(row.items()))) for row in FIXED_ROWS]
    assert fit_temperature(reordered, "fp").dataset_sha256 == FIXED_ROWS_SHA256
    changed = [*FIXED_ROWS[:-1], {**FIXED_ROWS[-1], "label_index": 0}]
    assert fit_temperature(changed, "fp").dataset_sha256 != FIXED_ROWS_SHA256


# the saved file ---------------------------------------------------------------------------------


def write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_fit_survives_a_round_trip_through_a_file(tmp_path):
    fit = fit_temperature(calibrated_rows(), "mod\u00e8le-\u03b1")  # not ASCII: read as UTF-8
    path = tmp_path / "fit.json"
    path.write_text(fit.model_dump_json(), encoding="utf-8")
    loaded = Calibration.from_file(path)
    assert loaded == fit
    assert loaded.fingerprint == "mod\u00e8le-\u03b1"


def test_from_file_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        Calibration.from_file(tmp_path / "missing.json")


def valid_payload() -> dict[str, Any]:
    return fit_temperature(calibrated_rows(), "fp").model_dump()


@pytest.mark.parametrize(
    "change",
    [
        {"version": 2},
        {"status": "validated"},
        {"temperatures": {"choice": 0}},
        {"temperatures": {"choice": -1.5}},
        {"temperatures": {"text": 1.5}},
        {"temperatures": {"choice": "hot"}},
        {"extra": 1},
        {"fingerprint": 3},
    ],
)
def test_from_file_rejects_malformed_calibrations(tmp_path, change):
    path = write(tmp_path / "bad.json", {**valid_payload(), **change})
    with pytest.raises(ValidationError):
        Calibration.from_file(path)


def test_from_file_rejects_non_finite_temperatures_in_the_json(tmp_path):
    path = tmp_path / "nan.json"
    text = json.dumps({**valid_payload(), "temperatures": {"choice": 1.0}})
    path.write_text(text.replace('"choice": 1.0', '"choice": NaN'), encoding="utf-8")
    with pytest.raises(ValidationError):
        Calibration.from_file(path)
    path.write_text(text.replace('"choice": 1.0', '"choice": Infinity'), encoding="utf-8")
    with pytest.raises(ValidationError):
        Calibration.from_file(path)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_from_file_checks_the_temperatures_again_after_parsing(tmp_path, monkeypatch, bad):
    # Pydantic already refuses these; the explicit check must still hold if the schema is relaxed.
    monkeypatch.setattr(
        Calibration,
        "model_validate_json",
        classmethod(lambda cls, data: cls.model_construct(**json.loads(data))),
    )
    payload = {**valid_payload(), "temperatures": {"boolean": 1.0, "choice": bad}}
    path = write(tmp_path / "relaxed.json", payload)
    with pytest.raises(
        ValueError, match=exactly("Calibration temperatures must be finite and positive")
    ):
        Calibration.from_file(path)


def test_from_file_accepts_what_the_explicit_check_accepts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Calibration,
        "model_validate_json",
        classmethod(lambda cls, data: cls.model_construct(**json.loads(data))),
    )
    payload = {**valid_payload(), "temperatures": {"boolean": 0.5, "choice": 2.0}}
    assert Calibration.from_file(write(tmp_path / "ok.json", payload)).temperatures == {
        "boolean": 0.5,
        "choice": 2.0,
    }
