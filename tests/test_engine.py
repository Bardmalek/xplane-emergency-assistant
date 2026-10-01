"""Unit tests for the telemetry parser and the rule-based decision engine.

No X-Plane, network, or audio device is needed: states are built directly.
"""
import pytest

from decision_engine import (
    SEVERITY_EMERGENCY,
    SEVERITY_WARNING,
    MLDecisionEngine,
    RuleBasedEngine,
)
from flight_state import ENGINE_NOT_INSTALLED, EngineState, FlightState
from xplane_udp import (
    ALT_AGL_FT,
    ALT_MSL_FT,
    ENGINE_RPM_1,
    ENGINE_RPM_2,
    PITCH_DEG,
    ROLL_DEG,
    SPEED_IAS_KIAS,
)

RUNNING, FAILED = 2200.0, 0.0


def state(ias=120.0, pitch=2.0, roll=0.0, agl=5000.0, rpms=(RUNNING, RUNNING)):
    engines = [EngineState(rpm=r, throttle_cmd=1.0, installed=True) for r in rpms]
    return FlightState(ias_kts=ias, pitch_deg=pitch, roll_deg=roll, alt_agl_ft=agl,
                       alt_msl_ft=agl, engines=engines)


def ids(alerts):
    return {a["id"] for a in alerts}


@pytest.fixture
def engine():
    return RuleBasedEngine()


# ── normal flight ─────────────────────────────────────────────────────────────

def test_level_cruise_raises_nothing(engine):
    assert engine.evaluate(state()) == []


def test_attitude_rules_stay_quiet_on_the_ground(engine):
    # A parked aircraft sitting at an odd pitch/roll is not "airborne" and must not alert.
    assert engine.evaluate(state(ias=0, pitch=30, roll=70, agl=0)) == []


# ── engine failures after takeoff ─────────────────────────────────────────────

def test_single_engine_failure_low_says_do_not_raise_gear(engine):
    alerts = engine.evaluate(state(ias=90, pitch=8, agl=300, rpms=(FAILED, RUNNING)))
    (a,) = [x for x in alerts if x["id"] == "post_takeoff_single_eng_fail"]
    assert a["severity"] == SEVERITY_EMERGENCY
    assert "Engine 1" in a["message"] and "Do NOT raise landing gear" in a["message"]


def test_single_engine_failure_higher_says_climb_first(engine):
    alerts = engine.evaluate(state(ias=100, pitch=8, agl=800, rpms=(RUNNING, FAILED)))
    (a,) = [x for x in alerts if x["id"] == "post_takeoff_single_eng_fail"]
    assert "Engine 2" in a["message"] and "Climb to safe altitude" in a["message"]


def test_single_engine_rule_ignores_single_engine_aircraft(engine):
    # One installed engine failing is the all-engines case, not the multi-engine one.
    alerts = engine.evaluate(state(ias=90, pitch=8, agl=300, rpms=(FAILED,)))
    assert "post_takeoff_single_eng_fail" not in ids(alerts)
    assert "post_takeoff_all_eng_fail" in ids(alerts)


def test_all_engines_failed_below_200ft_says_land_straight_ahead(engine):
    alerts = engine.evaluate(state(ias=85, pitch=8, agl=150, rpms=(FAILED, FAILED)))
    (a,) = [x for x in alerts if x["id"] == "post_takeoff_all_eng_fail"]
    assert "Land straight ahead" in a["message"] and "turn back" in a["message"]


def test_all_engines_failed_above_200ft_says_best_glide(engine):
    alerts = engine.evaluate(state(ias=85, pitch=8, agl=600, rpms=(FAILED, FAILED)))
    (a,) = [x for x in alerts if x["id"] == "post_takeoff_all_eng_fail"]
    assert "best glide" in a["message"] and "Land straight ahead" not in a["message"]


def test_engine_failure_in_cruise_is_not_a_takeoff_emergency(engine):
    alerts = engine.evaluate(state(agl=5000, rpms=(FAILED, RUNNING)))
    assert "post_takeoff_single_eng_fail" not in ids(alerts)


# ── stall and attitude ────────────────────────────────────────────────────────

def test_stall_warning_needs_low_speed_and_nose_up(engine):
    assert "stall_warning" in ids(engine.evaluate(state(ias=50, pitch=8, agl=400)))
    assert "stall_warning" not in ids(engine.evaluate(state(ias=50, pitch=0, agl=400)))
    assert "stall_warning" not in ids(engine.evaluate(state(ias=90, pitch=8, agl=400)))


def test_low_airspeed_is_a_warning_before_it_is_a_stall(engine):
    alerts = engine.evaluate(state(ias=65, pitch=3, agl=2000))
    (a,) = alerts
    assert a["id"] == "low_airspeed" and a["severity"] == SEVERITY_WARNING


@pytest.mark.parametrize("pitch, expected", [(26, "unusual_attitude_pitch_up"),
                                             (-21, "unusual_attitude_pitch_down")])
def test_extreme_pitch(engine, pitch, expected):
    assert expected in ids(engine.evaluate(state(pitch=pitch, agl=3000)))


@pytest.mark.parametrize("roll, side", [(65, "right"), (-65, "left")])
def test_extreme_bank_reports_the_correct_side(engine, roll, side):
    alerts = engine.evaluate(state(roll=roll, agl=3000))
    (a,) = [x for x in alerts if x["id"] == "unusual_attitude_bank"]
    assert f"to the {side}" in a["message"]


def test_limits_are_strict_inequalities(engine):
    assert engine.evaluate(state(pitch=25, roll=60, agl=3000)) == []


# ── telemetry parsing ─────────────────────────────────────────────────────────

def test_from_raw_maps_fields_and_marks_missing_engines_not_installed():
    raw = {SPEED_IAS_KIAS: 95.0, PITCH_DEG: 7.5, ROLL_DEG: -3.0, ALT_MSL_FT: 1200.0,
           ALT_AGL_FT: 700.0, ENGINE_RPM_1: 2300.0, ENGINE_RPM_2: ENGINE_NOT_INSTALLED}
    s = FlightState.from_raw(raw)
    assert (s.ias_kts, s.pitch_deg, s.roll_deg, s.alt_agl_ft) == (95.0, 7.5, -3.0, 700.0)
    assert s.num_engines_installed == 1 and s.num_engines_running == 1
    assert not s.any_engine_failed


def test_engine_failed_threshold():
    assert EngineState(rpm=299.9, throttle_cmd=1, installed=True).is_failed
    assert EngineState(rpm=300.0, throttle_cmd=1, installed=True).is_running
    assert not EngineState(rpm=0, throttle_cmd=0, installed=False).is_failed


# ── ML extension point ────────────────────────────────────────────────────────

def test_ml_engine_without_a_model_is_inert(tmp_path):
    ml = MLDecisionEngine(model_path=str(tmp_path / "missing.pkl"))
    assert ml.evaluate(state(agl=300, rpms=(FAILED, FAILED))) == []
