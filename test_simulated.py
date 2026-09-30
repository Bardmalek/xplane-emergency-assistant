"""
test_simulated.py
------------------
Runs the full pipeline against SimulatedXPlaneReader (no X-Plane required).
Prints flight state and any triggered alerts to the terminal.
TTS will speak if pyttsx3 / espeak is installed; otherwise prints to console.

Run:
    python test_simulated.py
"""

import time
import logging
from xplane_udp import SimulatedXPlaneReader
from flight_state import FlightState
from decision_engine import RuleBasedEngine
from tts_engine import TTSEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

COOLDOWN_SEC = 5.0
DURATION_SEC = 30

def main():
    reader = SimulatedXPlaneReader()
    engine = RuleBasedEngine()
    tts = TTSEngine()

    last_alert: dict[str, float] = {}

    reader.open()
    print("\n" + "="*70)
    print("  X-PLANE EMERGENCY ASSISTANT — Simulated Flight Test")
    print("  Timeline: 0-5s taxi | 5-15s climb | 15s+ ENGINE 1 FAILS")
    print("="*70 + "\n")

    t_start = time.monotonic()
    while time.monotonic() - t_start < DURATION_SEC:
        raw = reader.read_latest()
        if raw:
            state = FlightState.from_raw(raw)
            print(f"  {state}")

            alerts = engine.evaluate(state)
            for alert in alerts:
                sid = alert["id"]
                now = time.monotonic()
                if now - last_alert.get(sid, 0.0) >= COOLDOWN_SEC:
                    last_alert[sid] = now
                    print(f"\n  ⚠️  [{alert['severity']}] {alert['message']}\n")
                    tts.speak(alert["message"], severity=alert["severity"])

        time.sleep(0.1)

    print("\n[Test complete]")
    reader.close()

if __name__ == "__main__":
    main()
