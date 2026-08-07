"""Configuration loading and override behaviour."""
import pytest

from gaitscreen.config import Config, ConfigError


def test_defaults_load(cfg):
    assert cfg["video.fps_variability_min"] == 50
    assert cfg["filter.kind"] == "butterworth"
    assert cfg["flagging.absolute.gait_speed_mps.high_risk_below"] == 0.6


def test_unknown_key_raises_rather_than_returning_none(cfg):
    """A silent None here would become a silently disabled threshold."""
    with pytest.raises(ConfigError):
        cfg["flagging.absolute.gait_speed_mps.typo"]
    assert cfg.get("nope.not.here", 42) == 42


def test_overrides_are_deep_merged(cfg):
    overridden = cfg.with_overrides(
        {"flagging": {"absolute": {"gait_speed_mps": {"high_risk_below": 0.7}}}}
    )
    assert overridden["flagging.absolute.gait_speed_mps.high_risk_below"] == 0.7
    # Sibling keys survive the merge.
    assert overridden["flagging.absolute.gait_speed_mps.moderate_risk_below"] == 1.0
    assert cfg["flagging.absolute.gait_speed_mps.high_risk_below"] == 0.6


def test_user_yaml_merges_over_defaults(tmp_path):
    path = tmp_path / "override.yaml"
    path.write_text("video:\n  fps_variability_min: 30\n", encoding="utf-8")

    loaded = Config.load(path)
    assert loaded["video.fps_variability_min"] == 30
    assert loaded["video.fps_warn_below"] == 30  # untouched default


def test_section_returns_a_config_view(cfg):
    section = cfg.section("filter.butterworth")
    assert section["cutoff_hz"] == 6.0
    with pytest.raises(ConfigError):
        cfg.section("filter.kind")


def test_every_core_metric_has_a_deterioration_direction():
    """Trend flagging must know which way is worse for each metric."""
    from gaitscreen.types import CORE_METRICS, DETERIORATION_DIRECTION

    assert set(CORE_METRICS) == set(DETERIORATION_DIRECTION)
    assert all(v in (-1, 1) for v in DETERIORATION_DIRECTION.values())


def test_every_core_metric_has_a_minimum_detectable_change(cfg):
    """Without an MDC floor, a trend flag can fire on measurement noise."""
    from gaitscreen.types import CORE_METRICS

    mdc = cfg["flagging.trend.minimum_detectable_change"]
    assert set(CORE_METRICS) <= set(mdc)


def test_resolve_path_handles_relative_and_absolute(cfg, tmp_path):
    relative = cfg.resolve_path("storage.db_path", tmp_path)
    assert relative.is_absolute()
    assert relative.parent.parent == tmp_path or relative.parent == tmp_path / "data"
