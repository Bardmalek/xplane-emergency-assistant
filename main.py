"""
X-Plane Emergency Decision Assistant
-------------------------------------
Reads real-time flight data from X-Plane via UDP, detects emergency scenarios,
and delivers spoken instructions to the pilot within 0.5 seconds.

Architecture is modular: swap DecisionEngine for an ML model without touching
the data pipeline or TTS layer.
"""

import threading
import time
import logging
from xplane_udp import XPlaneUDPReader
from flight_state import FlightState
from decision_engine import DecisionEngine, RuleBasedEngine
from tts_engine import TTSEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


class EmergencyAssistant:
    def __init__(
        self,
        xplane_ip: str = "127.0.0.1",
        xplane_port: int = 49000,
        decision_engine: DecisionEngine = None,
        poll_hz: float = 20.0,          # How often we poll X-Plane (20 Hz)
        cooldown_sec: float = 5.0,      # Seconds before repeating same alert
    ):
        self.reader = XPlaneUDPReader(ip=xplane_ip, port=xplane_port)
        self.engine: DecisionEngine = decision_engine or RuleBasedEngine()
        self.tts = TTSEngine()
        self.poll_interval = 1.0 / poll_hz
        self.cooldown_sec = cooldown_sec

        self._running = False
        self._last_alert: dict[str, float] = {}  # scenario_id -> last_spoken_time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the assistant in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Emergency assistant started (%.0f Hz polling)", 1.0 / self.poll_interval)

    def stop(self):
        self._running = False
        self.reader.close()
        log.info("Emergency assistant stopped.")

    def run_blocking(self):
        """Run in the main thread (use for CLI entry point)."""
        self.start()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self):
        self.reader.open()
        while self._running:
            t0 = time.monotonic()
            try:
                raw = self.reader.read_latest()
                if raw:
                    state = FlightState.from_raw(raw)
                    alerts = self.engine.evaluate(state)
                    for alert in alerts:
                        self._maybe_speak(alert)
            except Exception as exc:
                log.warning("Loop error: %s", exc)

            elapsed = time.monotonic() - t0
            sleep_time = self.poll_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _maybe_speak(self, alert: dict):
        """Respect cooldown so the same alert doesn't repeat every 50 ms."""
        sid = alert["id"]
        now = time.monotonic()
        last = self._last_alert.get(sid, 0.0)
        if now - last >= self.cooldown_sec:
            self._last_alert[sid] = now
            log.warning("ALERT [%s] %s", alert["severity"], alert["message"])
            self.tts.speak(alert["message"])


if __name__ == "__main__":
    assistant = EmergencyAssistant()
    assistant.run_blocking()
