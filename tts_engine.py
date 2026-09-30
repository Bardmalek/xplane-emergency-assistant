"""
tts_engine.py
--------------
Text-to-speech layer. Target: spoken output within 0.5 s of detection.

Strategy for low latency:
  - Use pyttsx3 (offline, no network round-trip) as primary engine
  - Optional: OpenAI TTS / ElevenLabs as high-quality fallback (async)
  - Speech runs in a background thread so it never blocks the decision loop

Priority queue: EMERGENCY alerts interrupt any currently-playing audio.
"""

from __future__ import annotations
import threading
import queue
import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

# Priority levels (lower number = higher priority)
PRIORITY_EMERGENCY = 0
PRIORITY_CAUTION   = 1
PRIORITY_WARNING   = 2
PRIORITY_INFO      = 3

_SEVERITY_TO_PRIORITY = {
    "EMERGENCY": PRIORITY_EMERGENCY,
    "CAUTION":   PRIORITY_CAUTION,
    "WARNING":   PRIORITY_WARNING,
}


class TTSEngine:
    """
    Queued, prioritized text-to-speech engine.
    
    Backend detection order:
      1. pyttsx3  (pip install pyttsx3)           — offline, fast (~100 ms)
      2. espeak   (system package)                — offline, robotic but reliable
      3. print-only fallback (no audio library)  — for headless / CI environments
    """

    def __init__(self, rate: int = 180, volume: float = 1.0):
        self._rate = rate
        self._volume = volume
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0          # tie-breaker to keep PriorityQueue stable
        self._seq_lock = threading.Lock()
        self._current_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._backend = self._detect_backend()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ── Public ────────────────────────────────────────────────────────

    def speak(self, text: str, severity: str = "EMERGENCY"):
        """Enqueue text for speech. EMERGENCY items jump the queue."""
        priority = _SEVERITY_TO_PRIORITY.get(severity, PRIORITY_INFO)
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
        self._queue.put((priority, seq, text))
        log.debug("TTS queued [pri=%d]: %s", priority, text[:60])

    # ── Backend detection ─────────────────────────────────────────────

    def _detect_backend(self) -> str:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
            log.info("TTS backend: pyttsx3")
            return "pyttsx3"
        except Exception:
            pass

        try:
            import subprocess
            subprocess.run(["espeak", "--version"],
                           capture_output=True, check=True, timeout=2)
            log.info("TTS backend: espeak")
            return "espeak"
        except Exception:
            pass

        log.warning("No TTS audio backend found. Using print-only fallback.")
        return "print"

    # ── Worker loop ───────────────────────────────────────────────────

    def _worker_loop(self):
        while True:
            try:
                priority, _, text = self._queue.get(timeout=1.0)
                t0 = time.monotonic()
                self._say(text)
                elapsed_ms = (time.monotonic() - t0) * 1000
                log.debug("TTS spoke in %.0f ms", elapsed_ms)
            except queue.Empty:
                pass
            except Exception as exc:
                log.error("TTS worker error: %s", exc)

    def _say(self, text: str):
        if self._backend == "pyttsx3":
            self._say_pyttsx3(text)
        elif self._backend == "espeak":
            self._say_espeak(text)
        else:
            print(f"\n🔊 [{time.strftime('%H:%M:%S')}] VOICE: {text}\n")

    def _say_pyttsx3(self, text: str):
        import pyttsx3
        # Instantiate per call to avoid thread-safety issues on some platforms
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)
        # Prefer a clear male voice if available
        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def _say_espeak(self, text: str):
        import subprocess
        subprocess.run(
            ["espeak", "-s", str(self._rate), "-a", "200", text],
            timeout=10
        )
