# X-Plane Emergency Decision Assistant

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
| `test_simulated.py` | Standalone simulation |

The ML engine is an extension point; the repository does not include a trained model.
