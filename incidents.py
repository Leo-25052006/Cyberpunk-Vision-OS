"""
incidents.py  -  Track B layer (incident triage + simulated dispatch)

Sits next to app.py. app.py feeds it the live headcount; this module decides when a
crowd situation becomes an INCIDENT, saves evidence (snapshot + 5-second replay),
stores everything in SQLite, and simulates dispatching the nearest response unit.

Severity ladder (share of zone capacity):
    Info      >= 60 %      occupancy rising
    Warning   >= 80 %      crowd nearing capacity
    Critical  >= 100 %     capacity exceeded
"""
import math
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "incidents.db"
EVIDENCE_DIR = BASE_DIR / "evidence"

# ------------------------------------------------------------------ tunable rules
INFO_RATIO = 0.60        # >= 60 % of capacity  -> Info
WARNING_RATIO = 0.80     # >= 80 % of capacity  -> Warning
RAISE_HOLD_S = 1.5       # level must hold this long before an incident is raised (no flicker alerts)
CLEAR_HOLD_S = 3.0       # crowd must stay normal this long before the episode is closed
COOLDOWN_S = 8.0         # pause after an episode closes before a new one can start

SEVERITY = {1: "Info", 2: "Warning", 3: "Critical"}
TITLES = {1: "Occupancy rising", 2: "Crowd nearing capacity", 3: "Overcrowding: capacity exceeded"}

ZONE = "Zone A - Plaza Walkway"
CAMERA = "CAM-01"

# Simple floor-plan coordinates (0-100 x, 0-100 y) used for the nearest-unit calculation and the map
INCIDENT_POS = {"x": 52, "y": 48}
UNITS = [
    {"id": "SEC-A", "name": "Security Patrol Alpha", "type": "Security", "base": "Main Gate",    "x": 12, "y": 22},
    {"id": "SEC-B", "name": "Security Patrol Bravo", "type": "Security", "base": "Library Block", "x": 84, "y": 26},
    {"id": "MED-1", "name": "First-Aid Team",        "type": "Medical",  "base": "Medical Room",  "x": 30, "y": 82},
    {"id": "RSP-1", "name": "Rapid Response Van",    "type": "Response", "base": "Admin Block",   "x": 88, "y": 78},
]
ETA_SECONDS_PER_UNIT = 0.4   # simulated travel speed (shortened so the demo stays quick)
MIN_ETA_S = 8

# ------------------------------------------------------------------ storage helpers
_lock = threading.RLock()
_version = 0


def version():
    return _version


def _bump():
    global _version
    _version += 1


@contextmanager
def _db():
    with _lock:
        con = sqlite3.connect(DB_PATH, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()


def _log(con, kind, message):
    con.execute("INSERT INTO events (ts, kind, message) VALUES (?, ?, ?)", (time.time(), kind, message))


def init_db():
    EVIDENCE_DIR.mkdir(exist_ok=True)
    with _db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at REAL NOT NULL,
            level INTEGER NOT NULL,
            title TEXT NOT NULL,
            zone TEXT NOT NULL,
            camera TEXT NOT NULL,
            headcount INTEGER NOT NULL,
            capacity INTEGER NOT NULL,
            peak INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'Open',
            unit_id TEXT,
            dispatched_at REAL,
            eta_s INTEGER,
            arrived_at REAL,
            resolved_at REAL,
            cleared_at REAL,
            note TEXT,
            evidence_ts REAL,
            frame_count INTEGER NOT NULL DEFAULT 0
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            kind TEXT NOT NULL,
            message TEXT NOT NULL
        )""")


def _unit(unit_id):
    return next((u for u in UNITS if u["id"] == unit_id), None)


def _distance(unit):
    return math.hypot(unit["x"] - INCIDENT_POS["x"], unit["y"] - INCIDENT_POS["y"])


def _eta(unit):
    return max(MIN_ETA_S, int(round(_distance(unit) * ETA_SECONDS_PER_UNIT)))


def compute_level(headcount, capacity):
    """0 = normal, 1 = Info, 2 = Warning, 3 = Critical."""
    if capacity <= 0:
        return 0
    ratio = headcount / capacity
    if ratio >= 1.0:
        return 3
    if ratio >= WARNING_RATIO:
        return 2
    if ratio >= INFO_RATIO:
        return 1
    return 0


# ------------------------------------------------------------------ evidence
def save_evidence(incident_id, snapshot_jpeg, clip_jpegs):
    """Snapshot (full-size keyframe) + replay frames (the last ~5 s before the alert)."""
    folder = EVIDENCE_DIR / str(incident_id)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("f*.jpg"):
        old.unlink()
    (folder / "snapshot.jpg").write_bytes(snapshot_jpeg)
    for i, jpg in enumerate(clip_jpegs):
        (folder / f"f{i:02d}.jpg").write_bytes(jpg)
    with _db() as con:
        con.execute("UPDATE incidents SET evidence_ts=?, frame_count=? WHERE id=?",
                    (time.time(), len(clip_jpegs), incident_id))


# ------------------------------------------------------------------ rule engine (state machine)
class IncidentEngine:
    """Turns a stream of (headcount, capacity) readings into incidents.
    One 'episode' = from the crowd first getting elevated until it returns to normal.
    Inside an episode the incident is escalated (Info -> Warning -> Critical), never duplicated."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.active_id = None
        self.active_level = 0
        self.peak = 0
        self.pending_level = 0
        self.pending_since = 0.0
        self.normal_since = None
        self.last_close = 0.0

    def update(self, headcount, capacity, now, get_evidence):
        level = compute_level(headcount, capacity)

        if self.active_id is not None and headcount > self.peak:
            self.peak = headcount
            with _db() as con:
                con.execute("UPDATE incidents SET peak=? WHERE id=?", (self.peak, self.active_id))

        # raise / escalate (only after the level has held steady for RAISE_HOLD_S)
        if level > self.active_level:
            self.normal_since = None
            if self.pending_level != level:
                self.pending_level, self.pending_since = level, now
            elif now - self.pending_since >= RAISE_HOLD_S:
                if self.active_id is not None or now - self.last_close >= COOLDOWN_S:
                    self._raise(level, headcount, capacity, get_evidence)
                self.pending_level = 0
        else:
            self.pending_level = 0

        # close the episode once the crowd has been normal for CLEAR_HOLD_S
        if level == 0 and self.active_id is not None:
            if self.normal_since is None:
                self.normal_since = now
            elif now - self.normal_since >= CLEAR_HOLD_S:
                self._close_episode(now)
        elif level > 0:
            self.normal_since = None

    def _raise(self, level, headcount, capacity, get_evidence):
        snapshot, clip = get_evidence()
        with _db() as con:
            if self.active_id is None:
                cur = con.execute(
                    "INSERT INTO incidents (created_at, level, title, zone, camera, headcount, capacity, peak) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (time.time(), level, TITLES[level], ZONE, CAMERA, headcount, capacity, headcount))
                self.active_id = cur.lastrowid
                self.peak = headcount
                _log(con, "INCIDENT", f"Incident #{self.active_id} raised - {SEVERITY[level]}: {TITLES[level]} "
                                      f"({headcount}/{capacity})")
            else:
                con.execute("UPDATE incidents SET level=?, title=?, headcount=?, capacity=?, peak=? WHERE id=?",
                            (level, TITLES[level], headcount, capacity, max(self.peak, headcount), self.active_id))
                _log(con, "INCIDENT", f"Incident #{self.active_id} escalated to {SEVERITY[level]} ({headcount}/{capacity})")
        self.active_level = level
        save_evidence(self.active_id, snapshot, clip)
        _bump()

    def _close_episode(self, now):
        iid = self.active_id
        with _db() as con:
            inc = con.execute("SELECT level, status FROM incidents WHERE id=?", (iid,)).fetchone()
            con.execute("UPDATE incidents SET cleared_at=? WHERE id=?", (now, iid))
            if inc and inc["level"] == 1 and inc["status"] == "Open":
                con.execute("UPDATE incidents SET status='Resolved', resolved_at=?, note=? WHERE id=?",
                            (now, "Auto-closed: occupancy returned to normal", iid))
                _log(con, "INCIDENT", f"Incident #{iid} auto-closed (Info level, crowd normalised)")
            else:
                _log(con, "INCIDENT", f"Incident #{iid} condition cleared - awaiting operator review")
        self.active_id = None
        self.active_level = 0
        self.normal_since = None
        self.last_close = now
        _bump()


engine = IncidentEngine()


# ------------------------------------------------------------------ queries
def _to_dict(row, with_frames=False):
    d = dict(row)
    d["severity"] = SEVERITY[d["level"]]
    unit = _unit(d["unit_id"]) if d["unit_id"] else None
    d["unit_name"] = unit["name"] if unit else None
    d["condition"] = "Cleared" if d["cleared_at"] else "Active"
    if d["status"] == "Dispatched":
        d["unit_state"] = "On scene" if d["arrived_at"] else "En route"
    else:
        d["unit_state"] = None
    v = int(d["evidence_ts"] or 0)
    d["snapshot_url"] = f"/evidence/{d['id']}/snapshot.jpg?v={v}" if d["evidence_ts"] else None
    if with_frames:
        d["frames"] = [f"/evidence/{d['id']}/f{i:02d}.jpg?v={v}" for i in range(d["frame_count"])]
    return d


def stats():
    with _db() as con:
        rows = con.execute("SELECT status, level, COUNT(*) AS n FROM incidents GROUP BY status, level").fetchall()
    s = {"total": 0, "open": 0, "critical_open": 0, "dispatched": 0, "resolved": 0}
    for r in rows:
        s["total"] += r["n"]
        if r["status"] in ("Open", "Dispatched"):
            s["open"] += r["n"]
            if r["level"] == 3:
                s["critical_open"] += r["n"]
        if r["status"] == "Dispatched":
            s["dispatched"] += r["n"]
        if r["status"] in ("Resolved", "Dismissed"):
            s["resolved"] += r["n"]
    return s


def list_incidents(limit=100):
    with _db() as con:
        rows = con.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_to_dict(r) for r in rows]


def get_incident(incident_id):
    with _db() as con:
        row = con.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
    if not row:
        raise LookupError("Incident not found")
    return _to_dict(row, with_frames=True)


def units_status():
    """Every unit with its current state and its distance/ETA to the incident location."""
    with _db() as con:
        busy = {r["unit_id"]: r for r in con.execute(
            "SELECT unit_id, id, arrived_at FROM incidents WHERE status='Dispatched'")}
    out = []
    for u in UNITS:
        b = busy.get(u["id"])
        state = "Available" if not b else ("On scene" if b["arrived_at"] else "En route")
        out.append({**u, "state": state, "busy_incident": b["id"] if b else None,
                    "distance": round(_distance(u), 1), "eta_s": _eta(u)})
    return out


def list_events(limit=200):
    with _db() as con:
        rows = con.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------ operator actions
def dispatch(incident_id, unit_id=None):
    """Assign a unit. With no unit_id the NEAREST available unit is chosen."""
    with _db() as con:
        inc = con.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not inc:
            raise LookupError("Incident not found")
        if inc["status"] != "Open":
            raise ValueError(f"Incident is already {inc['status'].lower()}")
        busy = {r["unit_id"] for r in con.execute("SELECT unit_id FROM incidents WHERE status='Dispatched'")}
        free = [u for u in UNITS if u["id"] not in busy]
        if not free:
            raise ValueError("No units available")
        if unit_id:
            unit = next((u for u in free if u["id"] == unit_id), None)
            if not unit:
                raise ValueError("That unit is not available")
        else:
            unit = min(free, key=_distance)
        eta = _eta(unit)
        con.execute("UPDATE incidents SET status='Dispatched', unit_id=?, dispatched_at=?, eta_s=?, arrived_at=NULL "
                    "WHERE id=?", (unit["id"], time.time(), eta, incident_id))
        how = "selected by operator" if unit_id else "nearest available"
        _log(con, "DISPATCH", f"{unit['name']} dispatched to incident #{incident_id} ({how}, ETA {eta}s)")
    _bump()
    return get_incident(incident_id)


def _close(incident_id, status, note):
    with _db() as con:
        inc = con.execute("SELECT status FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not inc:
            raise LookupError("Incident not found")
        if inc["status"] not in ("Open", "Dispatched"):
            raise ValueError(f"Incident is already {inc['status'].lower()}")
        con.execute("UPDATE incidents SET status=?, resolved_at=?, note=? WHERE id=?",
                    (status, time.time(), note, incident_id))
        _log(con, "OPERATOR", f"Incident #{incident_id} {status.lower()} by operator")
    _bump()
    return get_incident(incident_id)


def resolve(incident_id):
    return _close(incident_id, "Resolved", "Resolved by operator")


def dismiss(incident_id):
    return _close(incident_id, "Dismissed", "Dismissed by operator as false alarm")


def tick_dispatch(now=None):
    """Called about once a second: marks dispatched units as 'on scene' when their ETA has passed."""
    now = now or time.time()
    changed = False
    with _db() as con:
        rows = con.execute("SELECT id, unit_id, dispatched_at, eta_s FROM incidents "
                           "WHERE status='Dispatched' AND arrived_at IS NULL").fetchall()
        for r in rows:
            if now >= r["dispatched_at"] + r["eta_s"]:
                con.execute("UPDATE incidents SET arrived_at=? WHERE id=?", (now, r["id"]))
                _log(con, "DISPATCH", f"{_unit(r['unit_id'])['name']} is on scene at incident #{r['id']}")
                changed = True
    if changed:
        _bump()


def reset_all():
    """Wipe incidents, audit events and evidence (use right before a demo)."""
    with _db() as con:
        con.execute("DELETE FROM incidents")
        con.execute("DELETE FROM events")
        con.execute("DELETE FROM sqlite_sequence WHERE name IN ('incidents','events')")
    if EVIDENCE_DIR.exists():
        for child in EVIDENCE_DIR.iterdir():
            shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink()
    engine.reset()
    _bump()
