"""
xplane_udp.py
--------------
Connects to X-Plane's built-in UDP "DATA" output (legacy format) and
continuously reads the latest packet into a shared dict.

X-Plane DATA packet format
───────────────────────────
Each 36-byte group: 4-byte index (int) + 8 × float32
Total packet: 5-byte header ("DATA*") + N × 36-byte groups

You must configure X-Plane:
  Settings → Data Output → check "Send network data" to IP 127.0.0.1 port 49000
  Enable at least these DATA indices:
    3  – Speeds (IAS, etc.)
    17 – pitch/roll/headings
    20 – altitude / AGL
    34 – engine RPM
    37 – throttle commanded
    58 – engine failures flags  (or use custom DataRefs via RREF)

For production use, prefer RREF (DataRef subscription) for exact parameters.
This module supports both DATA (index-based) and a simulated feed for testing.
"""

import socket
import struct
import threading
import time
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ── X-Plane DATA index → human-readable group name ──────────────────────────
DATA_INDEX_NAMES = {
    3:  "speeds",
    17: "pitch_roll_hdg",
    20: "altitudes",
    34: "engine_rpm",
    37: "throttle_cmd",
    58: "engine_failures",
}

# Offsets within each DATA group (8 floats, 0-indexed)
# These come from Austin Meyer's DATA spec / X-Plane SDK docs
SPEED_IAS_KIAS     = (3,  0)   # group 3, float 0 → indicated airspeed (kts)
SPEED_TAS          = (3,  2)
PITCH_DEG          = (17, 0)   # nose-up positive
ROLL_DEG           = (17, 1)
ALT_MSL_FT         = (20, 2)   # altitude MSL in feet
ALT_AGL_FT         = (20, 3)
ENGINE_RPM_1       = (34, 0)
ENGINE_RPM_2       = (34, 1)
ENGINE_RPM_3       = (34, 2)
ENGINE_RPM_4       = (34, 3)
THROTTLE_CMD_1     = (37, 0)


class XPlaneUDPReader:
    """
    Listens on a UDP socket for X-Plane DATA packets.
    Non-blocking: the latest parsed values are stored in self.data (dict).
    Thread-safe via a read lock.
    """

    HEADER = b"DATA"

    def __init__(self, ip: str = "127.0.0.1", port: int = 49000,
                 buffer_size: int = 4096):
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self._sock: Optional[socket.socket] = None
        self._data: dict = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------

    def open(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((self.ip, self.port))
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        log.info("UDP listener opened on %s:%d", self.ip, self.port)

    def close(self):
        self._running = False
        if self._sock:
            self._sock.close()

    def read_latest(self) -> Optional[dict]:
        """Return a shallow copy of the latest parsed DATA dict, or None."""
        with self._lock:
            return dict(self._data) if self._data else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _receive_loop(self):
        while self._running:
            try:
                packet, _ = self._sock.recvfrom(self.buffer_size)
                parsed = self._parse_packet(packet)
                if parsed:
                    with self._lock:
                        self._data.update(parsed)
            except socket.timeout:
                pass  # normal – no packet arrived in 1 s
            except Exception as exc:
                log.debug("Receive error: %s", exc)

    @staticmethod
    def _parse_packet(packet: bytes) -> Optional[dict]:
        """
        Parse an X-Plane DATA packet.
        Returns dict keyed by (group_index, float_index) → float value.
        """
        if len(packet) < 5 or packet[:4] != b"DATA":
            return None

        data = {}
        offset = 5  # skip "DATA*"
        while offset + 36 <= len(packet):
            group_idx = struct.unpack_from("<i", packet, offset)[0]
            floats = struct.unpack_from("<8f", packet, offset + 4)
            for fi, val in enumerate(floats):
                data[(group_idx, fi)] = val
            offset += 36

        return data or None


# ── Simulator for offline development / unit tests ──────────────────────────

class SimulatedXPlaneReader(XPlaneUDPReader):
    """
    Produces synthetic flight data without needing X-Plane running.
    Simulates a normal takeoff followed by a post-takeoff engine failure.
    """

    def __init__(self):
        super().__init__()
        self._sim_time = 0.0

    def open(self):
        self._running = True
        self._thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._thread.start()
        log.info("Simulated X-Plane reader started (no real UDP connection)")

    def _sim_loop(self):
        while self._running:
            t = self._sim_time
            self._sim_time += 0.05
            synthetic = self._generate_state(t)
            with self._lock:
                self._data.update(synthetic)
            time.sleep(0.05)

    @staticmethod
    def _generate_state(t: float) -> dict:
        """
        Phase timeline:
          0–5 s   : taxi / takeoff roll (IAS rising, alt = 0, engines normal)
          5–15 s  : climb out (IAS 90 kts, alt climbing, engines normal)
          15+ s   : ENGINE 1 FAILURE (RPM drops to 0)
        """
        if t < 5:
            ias = t * 18          # 0 → 90 kts
            alt_agl = 0.0
            pitch = 2.0
            rpm1, rpm2 = 2400.0, 2400.0
        elif t < 15:
            ias = 90.0
            alt_agl = (t - 5) * 80   # climbing ~800 fpm
            pitch = 8.0
            rpm1, rpm2 = 2400.0, 2400.0
        else:
            ias = 85.0
            alt_agl = 800.0 + (t - 15) * 20
            pitch = 5.0
            rpm1 = max(0.0, 2400.0 - (t - 15) * 480)  # drops over 5 s
            rpm2 = 2400.0

        return {
            SPEED_IAS_KIAS:  ias,
            PITCH_DEG:       pitch,
            ROLL_DEG:        0.0,
            ALT_MSL_FT:      alt_agl + 152,   # field elevation ~500 ft
            ALT_AGL_FT:      alt_agl,
            ENGINE_RPM_1:    rpm1,
            ENGINE_RPM_2:    rpm2,
            ENGINE_RPM_3:    -999.0,           # not installed
            ENGINE_RPM_4:    -999.0,
            THROTTLE_CMD_1:  1.0,
        }
