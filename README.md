# X-Plane Emergency Decision Assistant

[![CI](https://github.com/Bardmalek/xplane-emergency-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Bardmalek/xplane-emergency-assistant/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)

A small Python prototype that reads X-Plane UDP telemetry, turns it into a typed flight state, evaluates rule-based emergency conditions, and speaks prioritized alerts. It includes a simulated telemetry source so the pipeline can be explored without X-Plane.

> **Simulation only.** This is an experimental decision-support prototype, not certified avionics or a source of real-world flight instructions.

## Flow

```text
X-Plane UDP / simulator → FlightState → RuleBasedEngine → alert queue → speech or console
```

## Try it

```bash
python3 test_simulated.py
```

The simulation runs for 30 seconds and introduces an engine failure after 15 seconds. Speech uses `pyttsx3` when available and falls back to console output.

## Tests

```bash
pip install pytest
pytest -q
```

18 unit tests run in well under a second, with no X-Plane, network, or audio device needed. They cover each emergency rule (single and total engine failure at different altitudes, stall, extreme pitch and bank, low airspeed), the thresholds at their exact boundaries, the "stay quiet on the ground and in cruise" cases, telemetry parsing, and the ML extension point being inert without a model. I also checked that they have teeth: deliberately breaking a threshold, a comparison, or a left/right swap makes them fail. CI runs them on Python 3.10 and 3.12.

## Connect X-Plane

Configure X-Plane to send network data to `127.0.0.1:49000` at 20 Hz or higher. Enable data indices **3** (speeds), **17** (attitude/heading), **20** (altitude), **34** (engine RPM), and **37** (throttle). Then run:

```bash
pip install pyttsx3
python3 main.py
```

## Code map

| File | Purpose |
| --- | --- |
| `xplane_udp.py` | X-Plane UDP parser and simulated reader |
| `flight_state.py` | Typed telemetry snapshot |
| `decision_engine.py` | Rules and an ML engine interface |
| `tts_engine.py` | Prioritized speech queue |
| `main.py` | Live orchestration |
| `test_simulated.py` | Standalone 30 s simulation (a demo, not a test) |
| `tests/` | Unit tests for the rules and the telemetry parser |

The ML engine is an extension point; the repository does not include a trained model.
