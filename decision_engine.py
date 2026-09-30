"""
decision_engine.py
-------------------
Abstract DecisionEngine interface + RuleBasedEngine implementation.

To swap in an ML model later:
  1. Subclass DecisionEngine
  2. Implement evaluate(state) → list[Alert]
  3. Pass your instance to EmergencyAssistant(decision_engine=MyMLEngine())

Nothing else changes.
"""

from __future__ import annotations
import abc
import time
import logging
from dataclasses import dataclass
from typing import Optional
from flight_state import FlightState

log = logging.getLogger(__name__)

# ── Alert schema ─────────────────────────────────────────────────────────────

SEVERITY_WARNING  = "WARNING"   # Awareness; pilot should monitor
SEVERITY_CAUTION  = "CAUTION"   # Degraded condition; action soon
SEVERITY_EMERGENCY = "EMERGENCY" # Immediate action required

@dataclass
class Alert:
    id: str           # Unique scenario ID (used for cooldown dedup)
    severity: str     # WARNING / CAUTION / EMERGENCY
    message: str      # Spoken instruction — clear, imperative, concise
    state: Optional[FlightState] = None   # Snapshot that triggered it


# ── Abstract interface ────────────────────────────────────────────────────────

class DecisionEngine(abc.ABC):
    """
    Contract: receive a FlightState, return zero or more Alerts.
    Implementations must be stateless OR manage their own state internally.
    """

    @abc.abstractmethod
    def evaluate(self, state: FlightState) -> list[Alert]:
        """Evaluate current flight state and return a list of active alerts."""
        ...


# ── Rule-Based Engine ─────────────────────────────────────────────────────────

class RuleBasedEngine(DecisionEngine):
    """
    Deterministic decision tree covering the most safety-critical scenarios.
    Rules are evaluated in priority order (highest severity first).
    Each rule is a callable: (FlightState) → Optional[Alert]
    """

    def __init__(self):
        # Ordered list of rule methods — add new rules here
        self._rules = [
            self._rule_post_takeoff_all_engines_failed,
            self._rule_post_takeoff_single_engine_failure,
            self._rule_stall_warning,
            self._rule_unusual_attitude_pitch_up,
            self._rule_unusual_attitude_pitch_down,
            self._rule_unusual_attitude_bank,
            self._rule_low_airspeed_airborne,
        ]

        # Track previous state for trend detection
        self._prev_state: Optional[FlightState] = None
        self._prev_time: float = 0.0

    # ── Public ────────────────────────────────────────────────────────

    def evaluate(self, state: FlightState) -> list[Alert]:
        alerts = []
        for rule in self._rules:
            alert = rule(state)
            if alert:
                log.debug("Rule fired: %s", alert.id)
                alerts.append(vars(alert))   # convert to dict for main.py

        self._prev_state = state
        self._prev_time = time.monotonic()
        return alerts

    # ── Rules (return Alert or None) ──────────────────────────────────

    def _rule_post_takeoff_all_engines_failed(self, s: FlightState) -> Optional[Alert]:
        """Complete power loss during takeoff / initial climb."""
        if s.in_takeoff_phase and s.all_engines_failed:
            # Determine if field is likely still reachable
            if s.alt_agl_ft < 200:
                msg = (
                    "MAYDAY. All engines failed. Below two hundred feet. "
                    "Land straight ahead. Do not attempt to turn back. "
                    "Establish best glide. Declare emergency."
                )
            else:
                msg = (
                    "MAYDAY. All engines failed. "
                    "Establish best glide speed now. "
                    "Identify nearest suitable landing area. "
                    "Declare emergency on 121 point 5."
                )
            return Alert(
                id="post_takeoff_all_eng_fail",
                severity=SEVERITY_EMERGENCY,
                message=msg,
                state=s,
            )
        return None

    def _rule_post_takeoff_single_engine_failure(self, s: FlightState) -> Optional[Alert]:
        """Single engine failure during takeoff / initial climb (multi-engine aircraft)."""
        if (
            s.in_takeoff_phase
            and s.num_engines_installed >= 2
            and s.any_engine_failed
            and not s.all_engines_failed
        ):
            failed_idx = next(
                i + 1 for i, e in enumerate(s.engines)
                if e.is_failed
            )
            if s.alt_agl_ft < 400:
                msg = (
                    f"Engine {failed_idx} failure below four hundred feet. "
                    "Do NOT raise landing gear yet. "
                    "Maintain runway heading. "
                    "Climb at Vyse. Identify — Verify — Feather."
                )
            else:
                msg = (
                    f"Engine {failed_idx} has failed. "
                    "Climb to safe altitude first. "
                    "Mixtures — Props — Throttles — full forward. "
                    "Identify failed engine with rudder pressure. "
                    "Verify — Feather — Declare emergency."
                )
            return Alert(
                id="post_takeoff_single_eng_fail",
                severity=SEVERITY_EMERGENCY,
                message=msg,
                state=s,
            )
        return None

    def _rule_stall_warning(self, s: FlightState) -> Optional[Alert]:
        """Imminent stall: very low airspeed + high pitch + airborne."""
        stall_ias_approx = 55.0   # kts — ideally aircraft-specific
        if s.is_airborne and s.ias_kts < stall_ias_approx and s.pitch_deg > 5:
            return Alert(
                id="stall_warning",
                severity=SEVERITY_EMERGENCY,
                message=(
                    "STALL WARNING. Lower the nose NOW. "
                    "Full power. Level wings. "
                    "Do not pull back."
                ),
                state=s,
            )
        return None

    def _rule_unusual_attitude_pitch_up(self, s: FlightState) -> Optional[Alert]:
        if s.pitch_deg > 25 and s.is_airborne:
            return Alert(
                id="unusual_attitude_pitch_up",
                severity=SEVERITY_EMERGENCY,
                message=(
                    "Unusual attitude. Extreme pitch up. "
                    "Reduce power. Lower nose. Level wings."
                ),
                state=s,
            )
        return None

    def _rule_unusual_attitude_pitch_down(self, s: FlightState) -> Optional[Alert]:
        if s.pitch_deg < -20 and s.is_airborne:
            return Alert(
                id="unusual_attitude_pitch_down",
                severity=SEVERITY_EMERGENCY,
                message=(
                    "Unusual attitude. Nose low. "
                    "Reduce bank. Add back pressure. "
                    "Do not exceed Vne."
                ),
                state=s,
            )
        return None

    def _rule_unusual_attitude_bank(self, s: FlightState) -> Optional[Alert]:
        if abs(s.roll_deg) > 60 and s.is_airborne:
            direction = "right" if s.roll_deg > 0 else "left"
            return Alert(
                id="unusual_attitude_bank",
                severity=SEVERITY_EMERGENCY,
                message=(
                    f"Extreme bank to the {direction}. "
                    "Roll wings level now. Recover altitude."
                ),
                state=s,
            )
        return None

    def _rule_low_airspeed_airborne(self, s: FlightState) -> Optional[Alert]:
        """Approaching stall — earlier warning."""
        if s.is_airborne and s.ias_kts < 70 and s.pitch_deg > 0:
            return Alert(
                id="low_airspeed",
                severity=SEVERITY_WARNING,
                message="Low airspeed. Add power. Check pitch attitude.",
                state=s,
            )
        return None


# ── ML Engine stub ────────────────────────────────────────────────────────────

class MLDecisionEngine(DecisionEngine):
    """
    Placeholder for a trained ML model (e.g. scikit-learn, PyTorch, ONNX).
    
    To implement:
      1. Load your model in __init__
      2. Featurise FlightState in _to_features()
      3. Map model output classes → Alert objects in evaluate()

    Example with an ONNX model:
        import onnxruntime as ort
        self.session = ort.InferenceSession("emergency_classifier.onnx")

    Example with scikit-learn:
        import joblib
        self.model = joblib.load("emergency_rf.pkl")
    """

    # Maps model output class index → (alert_id, severity, message)
    CLASS_MAP = {
        0: None,  # Normal flight — no alert
        1: ("engine_failure_ml", SEVERITY_EMERGENCY,
            "Engine failure detected. Execute engine failure checklist now."),
        2: ("stall_ml", SEVERITY_EMERGENCY,
            "Stall predicted. Lower the nose. Add full power immediately."),
        3: ("unusual_attitude_ml", SEVERITY_EMERGENCY,
            "Unusual attitude. Recover using attitude indicator."),
    }

    def __init__(self, model_path: str = "model.pkl"):
        self.model = None
        self._load_model(model_path)

    def _load_model(self, path: str):
        try:
            import joblib
            self.model = joblib.load(path)
            log.info("ML model loaded from %s", path)
        except Exception as exc:
            log.warning("Could not load ML model (%s). Engine inactive.", exc)

    @staticmethod
    def _to_features(s: FlightState) -> list[float]:
        """Convert FlightState to a flat feature vector matching training data."""
        return [
            s.ias_kts,
            s.pitch_deg,
            s.roll_deg,
            s.alt_agl_ft,
            s.engines[0].rpm if len(s.engines) > 0 else 0.0,
            s.engines[1].rpm if len(s.engines) > 1 else 0.0,
            float(s.is_airborne),
            float(s.in_takeoff_phase),
        ]

    def evaluate(self, state: FlightState) -> list[Alert]:
        if self.model is None:
            return []

        features = self._to_features(state)
        prediction = self.model.predict([features])[0]
        mapping = self.CLASS_MAP.get(prediction)
        if mapping is None:
            return []

        alert_id, severity, message = mapping
        return [vars(Alert(id=alert_id, severity=severity, message=message, state=state))]
