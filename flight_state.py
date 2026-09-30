"""
flight_state.py
----------------
Typed, normalized snapshot of aircraft state derived from raw X-Plane DATA.
This is the single data structure passed between the UDP layer and the
decision engine, keeping both sides decoupled.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time

# Import the key lookup constants from the UDP module
from xplane_udp import (
    SPEED_IAS_KIAS,
    PITCH_DEG,
    ROLL_DEG,
    ALT_MSL_FT,
    ALT_AGL_FT,
    ENGINE_RPM_1,
    ENGINE_RPM_2,
    ENGINE_RPM_3,
    ENGINE_RPM_4,
    THROTTLE_CMD_1,
)

# RPM threshold below which an engine is considered "failed"
ENGINE_FAILED_RPM_THRESHOLD = 300.0
# RPM value X-Plane uses for "engine not installed" / not applicable
ENGINE_NOT_INSTALLED = -999.0


@dataclass
class EngineState:
    rpm: float
    throttle_cmd: float
    installed: bool

    @property
    def is_running(self) -> bool:
        return self.installed and self.rpm >= ENGINE_FAILED_RPM_THRESHOLD

    @property
    def is_failed(self) -> bool:
        return self.installed and self.rpm < ENGINE_FAILED_RPM_THRESHOLD


@dataclass
class FlightState:
    # ── Timestamp ─────────────────────────────────────────────────────
    timestamp: float = field(default_factory=time.monotonic)

    # ── Speeds ────────────────────────────────────────────────────────
    ias_kts: float = 0.0        # Indicated Air Speed in knots

    # ── Attitude ──────────────────────────────────────────────────────
    pitch_deg: float = 0.0      # nose-up positive
    roll_deg: float = 0.0       # right-wing-down positive

    # ── Position ──────────────────────────────────────────────────────
    alt_msl_ft: float = 0.0
    alt_agl_ft: float = 0.0

    # ── Engines (up to 4) ─────────────────────────────────────────────
    engines: list[EngineState] = field(default_factory=list)

    # ── Derived / convenience ─────────────────────────────────────────
    @property
    def num_engines_installed(self) -> int:
        return sum(1 for e in self.engines if e.installed)

    @property
    def num_engines_running(self) -> int:
        return sum(1 for e in self.engines if e.is_running)

    @property
    def any_engine_failed(self) -> bool:
        return any(e.is_failed for e in self.engines)

    @property
    def all_engines_failed(self) -> bool:
        installed = [e for e in self.engines if e.installed]
        return bool(installed) and all(e.is_failed for e in installed)

    @property
    def is_airborne(self) -> bool:
        return self.alt_agl_ft > 50.0

    @property
    def in_takeoff_phase(self) -> bool:
        """Heuristic: low altitude, climbing, high throttle."""
        return (
            self.is_airborne
            and self.alt_agl_ft < 1500.0
            and self.pitch_deg > 3.0
        )

    # ── Factory ───────────────────────────────────────────────────────

    @classmethod
    def from_raw(cls, raw: dict) -> "FlightState":
        """Build a FlightState from the raw (group, float_idx) → value dict."""

        def get(key, default=0.0):
            return raw.get(key, default)

        throttle = get(THROTTLE_CMD_1, 0.0)

        engines = []
        for rpm_key in (ENGINE_RPM_1, ENGINE_RPM_2, ENGINE_RPM_3, ENGINE_RPM_4):
            rpm = get(rpm_key, ENGINE_NOT_INSTALLED)
            installed = rpm > ENGINE_NOT_INSTALLED + 1  # not -999
            engines.append(EngineState(
                rpm=rpm if installed else 0.0,
                throttle_cmd=throttle,
                installed=installed,
            ))

        return cls(
            ias_kts=get(SPEED_IAS_KIAS),
            pitch_deg=get(PITCH_DEG),
            roll_deg=get(ROLL_DEG),
            alt_msl_ft=get(ALT_MSL_FT),
            alt_agl_ft=get(ALT_AGL_FT),
            engines=engines,
        )

    def __str__(self) -> str:
        eng_str = " | ".join(
            f"E{i+1}:{'RUN' if e.is_running else ('FAIL' if e.is_failed else 'OFF')}"
            for i, e in enumerate(self.engines) if e.installed
        )
        return (
            f"IAS={self.ias_kts:.0f}kts  ALT={self.alt_agl_ft:.0f}ft AGL  "
            f"P={self.pitch_deg:.1f}°  R={self.roll_deg:.1f}°  [{eng_str}]"
        )
