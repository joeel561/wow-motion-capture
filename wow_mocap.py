# -*- coding: utf-8 -*-
"""
WoW Motion Capture
==================
Steuert deinen Spielcharakter mit Körperposen über die Webcam.

* Nur der Körper wird ausgewertet (Schultern, Arme, Hände, Hüfte, Beine, Füße).
  Die Kopf-/Gesichtspunkte (MediaPipe-Punkte 0-10) werden komplett ignoriert.
* Posen werden selbst kalibriert: Name + Taste eingeben, Aufnahme starten,
  Pose ein paar Sekunden halten - fertig.
* Posen gehören zu einem Körperbereich (Ganzer Körper, Rumpf, Arme, Beine).
  Jeder Bereich wird unabhängig erkannt, so kannst du z. B. gleichzeitig
  nach vorn lehnen (laufen) und mit dem Arm einen Zauber auslösen.
* Tasten werden per SendInput mit Scancodes gesendet (funktioniert in Spielen).
"""

import ctypes
import json
import math
import os
import queue
import threading
import time
import urllib.request
import winsound
from collections import Counter, namedtuple
from ctypes import wintypes

import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageTk

APP_TITLE = "WoW Motion Capture"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "posen.json")
MODEL_DIR = os.path.join(BASE_DIR, "modelle")
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}

DEFAULT_SETTINGS = {
    "kamera": 0,
    "modell": "full",
    "toleranz": 0.35,      # max. Abstand zur kalibrierten Pose (in Rumpflängen)
    "stabil": 4,           # so viele Frames muss eine Pose stabil sein
    "k": 5,                # k-nächste-Nachbarn
    "countdown": 5,        # Sekunden bis zur Aufnahme
    "dauer": 3,            # Sekunden Aufnahme
    "fenster": "World of Warcraft",  # Tasten nur an Fenster mit diesem Titel
    "toggle": "F12",       # globale Ein/Aus-Taste
}

# --------------------------------------------------------------------------
# Körperpunkte (ohne Kopf)
# --------------------------------------------------------------------------
# MediaPipe Pose: 0-10 = Kopf/Gesicht (ignoriert), 11-32 = Körper
BODY_IDX = list(range(11, 33))
L_SH, R_SH, L_HIP, R_HIP = 11 - 11, 12 - 11, 23 - 11, 24 - 11
FEATURE_WEIGHTS = np.array([1.0, 1.0, 0.5], dtype=np.float32)  # x, y, Tiefe

# Bereich -> (Anzeigename, Landmark-Indizes, Bezugspunkt)
ZONES = {
    "koerper": ("Ganzer Körper", list(range(11, 33)), "huefte"),
    "rumpf": ("Rumpf (Neigen)", [11, 12, 23, 24], "huefte"),
    "arme": ("Arme & Hände", list(range(11, 23)), "schulter"),
    "beine": ("Beine & Füße", list(range(23, 33)), "huefte"),
}
REST_ZONE = "alle"
ZONE_LABELS = {k: v[0] for k, v in ZONES.items()}
ZONE_LABELS[REST_ZONE] = "Ruhepose (alle Bereiche)"
ZONE_FROM_LABEL = {v: k for k, v in ZONE_LABELS.items()}

MODE_LABELS = {"antippen": "Antippen (einmal)", "halten": "Halten (solange aktiv)"}
MODE_SHORT = {"antippen": "Antippen", "halten": "Halten"}
MODE_FROM_LABEL = {v: k for k, v in MODE_LABELS.items()}
ART_LABELS = {"haltung": "Haltung (Pose stillhalten)",
              "bewegung": "Bewegung (z. B. auf der Stelle gehen)"}
ART_SHORT = {"haltung": "Haltung", "bewegung": "Bewegung"}
ART_FROM_LABEL = {v: k for k, v in ART_LABELS.items()}
POSE_KEYS = ("name", "zone", "keys", "mode", "art", "empf")

TOGGLE_KEYS = {"F12": 0x7B, "F11": 0x7A, "F10": 0x79, "F9": 0x78,
               "Pause": 0x13, "Rollen": 0x91, "Einfg": 0x2D}

CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (15, 17), (15, 19),
               (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23),
               (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
               (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)]

PoseInfo = namedtuple("PoseInfo", "name zone keys combo mode")
ZoneResult = namedtuple("ZoneResult", "name dist ranking")
FieldInfo = namedtuple("FieldInfo", "name rect part info")

# Körperteile für Kamera-Felder. Das Bild wird gespiegelt ausgewertet, deshalb
# sind MediaPipes "linke" Punkte (15, 17, ...) deine echte rechte Seite.
BODY_PARTS = {
    "haende": ("Eine Hand (egal welche)", list(range(15, 23))),
    "rechte_hand": ("Rechte Hand", [15, 17, 19, 21]),
    "linke_hand": ("Linke Hand", [16, 18, 20, 22]),
    "fuesse": ("Ein Fuß (egal welcher)", list(range(27, 33))),
    "rechter_fuss": ("Rechter Fuß", [27, 29, 31]),
    "linker_fuss": ("Linker Fuß", [28, 30, 32]),
    "mitte": ("Körpermitte", None),
    "koerper": ("Irgendein Körperteil", list(range(11, 33))),
}
PART_LABELS = {k: v[0] for k, v in BODY_PARTS.items()}
PART_FROM_LABEL = {v: k for k, v in PART_LABELS.items()}


def ascii_text(s):
    """OpenCV kann keine Umlaute zeichnen."""
    s = s.translate(str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae",
                                   "Ö": "Oe", "Ü": "Ue", "ß": "ss"}))
    return s.encode("ascii", "replace").decode("ascii")

# --------------------------------------------------------------------------
# Windows-Eingabe (SendInput mit Scancodes)
# --------------------------------------------------------------------------
user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE = 0x0001, 0x0002, 0x0008


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t))


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t))


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD))


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (("type", wintypes.DWORD), ("u", _INPUTUNION))


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT
user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
user32.MapVirtualKeyW.restype = wintypes.UINT
user32.VkKeyScanW.argtypes = (wintypes.WCHAR,)
user32.VkKeyScanW.restype = ctypes.c_short
user32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)

VK_NAMES = {
    "space": 0x20, "leertaste": 0x20, "tab": 0x09, "enter": 0x0D, "eingabe": 0x0D,
    "esc": 0x1B, "backspace": 0x08, "shift": 0xA0, "umschalt": 0xA0,
    "ctrl": 0xA2, "strg": 0xA2, "alt": 0xA4,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "hoch": 0x26, "runter": 0x28, "links": 0x25, "rechts": 0x27,
    "insert": 0x2D, "einfg": 0x2D, "delete": 0x2E, "entf": 0x2E,
    "home": 0x24, "pos1": 0x24, "end": 0x23, "ende": 0x23,
    "pageup": 0x21, "bildauf": 0x21, "pagedown": 0x22, "bildab": 0x22,
    "numplus": 0x6B, "numminus": 0x6D, "nummal": 0x6A, "numdurch": 0x6F,
}
VK_NAMES.update({f"f{i}": 0x6F + i for i in range(1, 13)})
VK_NAMES.update({f"num{i}": 0x60 + i for i in range(10)})
EXTENDED_VKS = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x6F}

# Name -> (Flag runter, Flag hoch, mouseData)
MOUSE_BUTTONS = {
    "lmb": (0x0002, 0x0004, 0), "rmb": (0x0008, 0x0010, 0), "mmb": (0x0020, 0x0040, 0),
    "maus4": (0x0080, 0x0100, 1), "maus5": (0x0080, 0x0100, 2),
}


def _send(*inputs):
    arr = (INPUT * len(inputs))(*inputs)
    user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


def send_key(vk, up):
    scan = user32.MapVirtualKeyW(vk, 0)  # MAPVK_VK_TO_VSC, nutzt aktuelles Tastaturlayout
    inp = INPUT(type=INPUT_KEYBOARD)
    if scan:
        flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
        if vk in EXTENDED_VKS:
            flags |= KEYEVENTF_EXTENDEDKEY
        inp.ki = KEYBDINPUT(0, scan, flags, 0, 0)
    else:
        inp.ki = KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP if up else 0, 0, 0)
    _send(inp)


def send_mouse(name, up):
    down_flag, up_flag, data = MOUSE_BUTTONS[name]
    inp = INPUT(type=INPUT_MOUSE)
    inp.mi = MOUSEINPUT(0, 0, data, up_flag if up else down_flag, 0, 0)
    _send(inp)


def parse_combo(text):
    """'shift+1' -> ((k, VK_LSHIFT), (k, VK_1)). Leerer Text -> ()."""
    text = (text or "").strip().lower()
    if not text:
        return ()
    keys = []
    for part in text.split("+"):
        p = part.strip()
        if not p:
            raise ValueError("Leerer Tastenteil. Für die Plus-Taste bitte 'numplus' schreiben.")
        if p in MOUSE_BUTTONS:
            keys.append(("m", p))
        elif p in VK_NAMES:
            keys.append(("k", VK_NAMES[p]))
        elif len(p) == 1:
            r = user32.VkKeyScanW(p)
            if r == -1:
                raise ValueError(f"Die Taste '{p}' gibt es auf diesem Tastaturlayout nicht.")
            keys.append(("k", r & 0xFF))
        else:
            raise ValueError(f"Unbekannte Taste: '{p}'")
    return tuple(keys)


def foreground_title():
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def beep(*seq):
    """seq: (frequenz, ms), ... - spielt im Hintergrund."""
    def run():
        for f, ms in seq:
            try:
                winsound.Beep(f, ms)
            except RuntimeError:
                pass
    threading.Thread(target=run, daemon=True).start()


class KeySender(threading.Thread):
    """Führt Tastenaktionen nacheinander in eigenem Thread aus und zählt gehaltene Tasten."""

    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.held = {}

    def down(self, combo):
        self.q.put(("down", combo))

    def up(self, combo):
        self.q.put(("up", combo))

    def tap(self, combo):
        self.q.put(("tap", combo))

    def release_all(self):
        self.q.put(("release_all",))

    def stop(self):
        self.q.put(None)

    @staticmethod
    def _press(key, up):
        if key[0] == "k":
            send_key(key[1], up)
        else:
            send_mouse(key[1], up)

    def run(self):
        while True:
            op = self.q.get()
            if op is None:
                self._release_all()
                return
            try:
                getattr(self, "_" + op[0])(*op[1:])
            except Exception as exc:  # noqa: BLE001
                print("Eingabefehler:", exc)

    def _down(self, combo):
        for key in combo:
            c = self.held.get(key, 0)
            if c == 0:
                self._press(key, False)
            self.held[key] = c + 1

    def _up(self, combo):
        for key in reversed(combo):
            c = self.held.get(key, 0)
            if c <= 1:
                self.held.pop(key, None)
                if c == 1:
                    self._press(key, True)
            else:
                self.held[key] = c - 1

    def _tap(self, combo):
        pressed = [k for k in combo if self.held.get(k, 0) == 0]
        for key in pressed:
            self._press(key, False)
        time.sleep(0.05)
        for key in reversed(pressed):
            self._press(key, True)

    def _release_all(self):
        for key in list(self.held):
            self._press(key, True)
        self.held.clear()


class ActionEngine:
    """Ein aktiver Zustand pro Körperbereich."""

    def __init__(self, sender):
        self.sender = sender
        self.active = {}  # zone -> PoseInfo (nur Halten-Posen)

    def set_pose(self, zone, info):
        old = self.active.pop(zone, None)
        if old is not None:
            self.sender.up(old.combo)
        if info is None or not info.combo:
            return
        if info.mode == "halten":
            self.sender.down(info.combo)
            self.active[zone] = info
        else:
            self.sender.tap(info.combo)

    def release_all(self):
        for zone in list(self.active):
            self.set_pose(zone, None)

    def held_text(self):
        return ", ".join(f"{i.keys} ({i.name})" for i in self.active.values())


# --------------------------------------------------------------------------
# Posenerkennung
# --------------------------------------------------------------------------
def zone_features(raws, zone):
    """raws: (N, 22, 4) = x, y, z, Sichtbarkeit -> normierte Punkte (N, K, 3), Sichtbarkeit (N, K)."""
    p = raws[..., :3] * FEATURE_WEIGHTS
    sh = (p[:, L_SH] + p[:, R_SH]) / 2
    hip = (p[:, L_HIP] + p[:, R_HIP]) / 2
    torso = np.linalg.norm((sh - hip)[:, :2], axis=1)
    shoulder_w = np.linalg.norm((p[:, L_SH] - p[:, R_SH])[:, :2], axis=1)
    # Mischung aus Rumpflänge und Schulterbreite: unabhängig vom Kameraabstand,
    # aber Vor-/Zurücklehnen bleibt sichtbar.
    scale = np.maximum(0.5 * torso + 0.65 * shoulder_w, 1e-3)
    _, ids, ref = ZONES[zone]
    center = sh if ref == "schulter" else hip
    idx = [i - 11 for i in ids]
    pts = (p[:, idx] - center[:, None, :]) / scale[:, None, None]
    vis = np.clip(raws[:, idx, 3], 0.0, 1.0)
    return pts, vis


def sample_distances(pa, va, pb, vb):
    """Abstand jeder Aufnahme aus A zu jeder aus B (Na x Nb), gewichtet wie im Klassifikator."""
    w = np.minimum(vb[None], va[:, None])
    d2 = ((pb[None] - pa[:, None]) ** 2).sum(axis=3)
    return np.sqrt((w * d2).sum(axis=2) / (w.sum(axis=2) + 1e-6))


def similar_poses(poses, name, limit=0.35, close=0.25, close_frac=0.15):
    """Welche Posen im selben Bereich kommen der Pose `name` zu nahe?
    -> [(Name, typischer Abstand, Anteil sehr naher Aufnahmen)]"""
    pose = next((p for p in poses if p["name"] == name), None)
    if not pose or pose["zone"] == REST_ZONE or pose.get("art") == "bewegung" or not pose["samples"]:
        return []
    zone = pose["zone"]
    pa, va = zone_features(np.asarray(pose["samples"], np.float32), zone)
    out = []
    for other in poses:
        if other is pose or not other["samples"] or other.get("art") == "bewegung":
            continue
        if other["zone"] not in (zone, REST_ZONE):
            continue
        pb, vb = zone_features(np.asarray(other["samples"], np.float32), zone)
        d = sample_distances(pa, va, pb, vb).min(axis=1)
        med, frac = float(np.median(d)), float((d < close).mean())
        if med < limit or frac > close_frac:
            out.append((other["name"], med, frac))
    return sorted(out, key=lambda t: t[1])


def similarity_warning(poses, name):
    hits = similar_poses(poses, name)
    if not hits:
        return None
    lines = []
    for other, med, frac in hits:
        if med < 0.35:
            lines.append(f"• „{name}“ sieht fast aus wie „{other}“ (Abstand {med:.2f}).")
        else:
            lines.append(f"• {frac:.0%} der Aufnahmen von „{name}“ sehen aus wie „{other}“ "
                         "(wahrscheinlich Bewegung oder Übergang aufgenommen).")
    return ("\n".join(lines) + "\n\nDadurch kann die Pose auslösen, obwohl du sie nicht machst. "
            "Tipp: Pose deutlicher machen und während der Aufnahme ruhig halten, dann die alte "
            "Pose löschen und neu aufnehmen. Für Bewegungen (z. B. Gehen) „Erkennung: Bewegung“ wählen.")


class ZoneModel:
    """k-nächste-Nachbarn über alle kalibrierten Samples eines Bereichs."""

    # Eine Haltung zählt nur, wenn sie deutlich näher ist als die Ruhepose.
    REST_MARGIN = 0.8

    # Aufnahmen einer Pose, die näher als dieser Anteil ihres typischen Abstands an
    # der Ruhepose liegen, gelten als Übergang (Hinein-/Herausgehen) und werden ignoriert.
    TRANSITION_CUT = 0.5

    def __init__(self, zone, poses):
        raws, labels = [], []
        for pose in poses:
            for s in pose["samples"]:
                raws.append(s)
                labels.append(pose["name"])
        self.zone = zone
        self.labels = np.array(labels)
        self.names = sorted(set(labels))
        self.rest_names = {p["name"] for p in poses if p["zone"] == REST_ZONE}
        self.sens = {p["name"]: float(p.get("empf", 1.0)) for p in poses}
        self.pts, self.vis = zone_features(np.asarray(raws, dtype=np.float32), zone)
        self.removed = {}
        self._drop_transitions()

    def _drop_transitions(self):
        rest_mask = np.isin(self.labels, list(self.rest_names))
        if not rest_mask.any():
            return
        keep = np.ones(len(self.labels), dtype=bool)
        for name in self.names:
            if name in self.rest_names:
                continue
            idx = np.where(self.labels == name)[0]
            d = sample_distances(self.pts[idx], self.vis[idx],
                                 self.pts[rest_mask], self.vis[rest_mask]).min(axis=1)
            drop = d < self.TRANSITION_CUT * np.median(d)
            if (~drop).sum() < 5:  # immer genug Aufnahmen übrig lassen
                continue
            keep[idx[drop]] = False
            self.removed[name] = int(drop.sum())
        self.labels, self.pts, self.vis = self.labels[keep], self.pts[keep], self.vis[keep]

    def classify(self, raw, k, threshold):
        pts, vis = zone_features(raw[None], self.zone)
        pts, vis = pts[0], vis[0]
        if vis.mean() < 0.35:  # Bereich nicht im Bild (z. B. Beine beim Sitzen)
            return ZoneResult(None, None, [])
        w = np.minimum(self.vis, vis[None, :])
        d2 = ((self.pts - pts[None]) ** 2).sum(axis=2)
        dist = np.sqrt((w * d2).sum(axis=1) / (w.sum(axis=1) + 1e-6))
        ranking = sorted(((n, float(dist[self.labels == n].min())) for n in self.names),
                         key=lambda t: t[1])
        kk = max(1, min(k, len(dist)))
        nearest = np.argpartition(dist, kk - 1)[:kk]
        best = Counter(self.labels[nearest].tolist()).most_common(1)[0][0]
        d = dict(ranking)
        best_dist = d[best]
        # Empfindlichkeit pro Pose: < 1 = strenger, > 1 = lockerer
        if best_dist > threshold * self.sens.get(best, 1.0):
            best = None
        elif best not in self.rest_names and self.rest_names:
            rest_dist = min(d[n] for n in self.rest_names)
            if best_dist > rest_dist * self.REST_MARGIN:
                best = None
        return ZoneResult(best, best_dist, ranking)


# --------------------------------------------------------------------------
# Bewegungserkennung (z. B. auf der Stelle gehen)
# --------------------------------------------------------------------------
MOTION_WINDOW_S = 0.7      # über diese Zeit wird die Bewegungsstärke gemittelt
DEFAULT_REST_ENERGY = 0.2  # typisches Messrauschen im Stillstand (Rumpflängen/s)
MotionInfo = namedtuple("MotionInfo", "name zone threshold energy rest_energy info")


def _frame_steps(pts, vis):
    """Bewegung von Frame zu Frame (nur x/y, Tiefe ist zu verrauscht)."""
    xy = pts[..., :2]
    wv = np.minimum(vis[1:], vis[:-1])
    d2 = ((xy[1:] - xy[:-1]) ** 2).sum(axis=2)
    return np.sqrt((wv * d2).sum(axis=1) / (wv.sum(axis=1) + 1e-6))


def recorded_energy(pose, zone):
    """Typische Bewegungsstärke einer Aufnahme in Rumpflängen pro Sekunde."""
    if len(pose["samples"]) < 10:
        return 0.0
    pts, vis = zone_features(np.asarray(pose["samples"], np.float32), zone)
    steps = _frame_steps(pts, vis)
    fps = float(pose.get("fps", 30.0))
    win = max(3, min(len(steps), int(round(MOTION_WINDOW_S * fps))))
    energy = np.convolve(steps, np.ones(win) / win, mode="valid") * fps
    return float(np.median(energy))


def motion_threshold(pose_e, rest_e, sens):
    """Schwelle zwischen Stillstand und kalibrierter Bewegung. sens > 1 = empfindlicher."""
    thr = rest_e + (pose_e - rest_e) * 0.35 / max(sens, 0.1)
    return max(thr, rest_e * 1.6)


class MotionTracker:
    """Misst laufend die Bewegungsstärke eines Körperbereichs."""

    def __init__(self, zone):
        self.zone = zone
        self.hist = []  # (Zeit, Punkte, Sichtbarkeit)

    def update(self, raw, now):
        if raw is None:
            self.hist.clear()
            return None
        pts, vis = zone_features(raw[None], self.zone)
        if vis[0].mean() < 0.35:
            self.hist.clear()
            return None
        self.hist.append((now, pts[0], vis[0]))
        while self.hist and now - self.hist[0][0] > MOTION_WINDOW_S:
            self.hist.pop(0)
        span = self.hist[-1][0] - self.hist[0][0] if len(self.hist) > 1 else 0.0
        if span < MOTION_WINDOW_S * 0.6:
            return None
        pts = np.stack([h[1] for h in self.hist])
        vis = np.stack([h[2] for h in self.hist])
        return float(_frame_steps(pts, vis).sum() / span)


def build_models(poses):
    infos = {}
    for p in poses:
        try:
            combo = parse_combo(p.get("keys", "")) if p["zone"] != REST_ZONE else ()
        except ValueError:
            combo = ()
        infos[p["name"]] = PoseInfo(p["name"], p["zone"], p.get("keys", ""), combo,
                                    p.get("mode", "antippen"))
    models = {}
    for zone in ZONES:
        members = [p for p in poses if p["samples"] and p["zone"] in (zone, REST_ZONE)
                   and p.get("art", "haltung") != "bewegung"]
        if any(p["zone"] == zone for p in members):
            models[zone] = ZoneModel(zone, members)

    motions = []
    rest = [p for p in poses if p["zone"] == REST_ZONE and p["samples"]]
    for p in poses:
        if p.get("art") != "bewegung" or p["zone"] == REST_ZONE or not p["samples"]:
            continue
        rest_e = (float(np.median([recorded_energy(r, p["zone"]) for r in rest]))
                  if rest else DEFAULT_REST_ENERGY)
        rest_e = max(rest_e, 0.05)
        pose_e = recorded_energy(p, p["zone"])
        thr = motion_threshold(pose_e, rest_e, float(p.get("empf", 1.0)))
        motions.append(MotionInfo(p["name"], p["zone"], thr, pose_e, rest_e, infos[p["name"]]))
    return models, infos, motions


def build_fields(felder):
    out = []
    for f in felder:
        try:
            combo = parse_combo(f.get("keys", ""))
        except ValueError:
            combo = ()
        info = PoseInfo(f["name"], "feld", f.get("keys", ""), combo, f.get("mode", "antippen"))
        out.append(FieldInfo(f["name"], tuple(f["rect"]), f.get("part", "haende"), info))
    return out


def field_hit(raw, aspect, field):
    """Ist das gewählte Körperteil im Rechteck (Koordinaten 0..1 im gespiegelten Bild)?"""
    x0, y0, x1, y1 = field.rect
    ids = BODY_PARTS.get(field.part, BODY_PARTS["haende"])[1]
    if ids is None:  # Körpermitte = Mittelpunkt von Schultern und Hüfte
        p = raw[[L_SH, R_SH, L_HIP, R_HIP]]
        if p[:, 3].min() < 0.3:
            return False
        x, y = p[:, 0].mean() / aspect, p[:, 1].mean()
        return bool(x0 <= x <= x1 and y0 <= y <= y1)
    p = raw[[i - 11 for i in ids]]
    xs, ys = p[:, 0] / aspect, p[:, 1]
    return bool(np.any((p[:, 3] > 0.5) & (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)))


def ensure_model(kind, report):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"pose_landmarker_{kind}.task")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    report(f"Lade Pose-Modell '{kind}' herunter (einmalig) ...")
    tmp = path + ".part"
    urllib.request.urlretrieve(MODEL_URLS[kind], tmp)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------
# Zeichnen
# --------------------------------------------------------------------------
def _color(i):
    if 13 <= i <= 22:
        return (0, 165, 255)   # Arme: orange (BGR)
    if 25 <= i <= 32:
        return (255, 200, 0)   # Beine: hellblau
    return (240, 240, 240)     # Rumpf


def draw_skeleton(img, raw, aspect):
    h, w = img.shape[:2]
    pts = {i: (int(raw[i - 11, 0] / aspect * w), int(raw[i - 11, 1] * h)) for i in BODY_IDX}
    vis = {i: raw[i - 11, 3] for i in BODY_IDX}
    th = max(2, w // 320)
    for a, b in CONNECTIONS:
        if vis[a] > 0.4 and vis[b] > 0.4:
            cv2.line(img, pts[a], pts[b], _color(b), th, cv2.LINE_AA)
    for i in BODY_IDX:
        if vis[i] > 0.4:
            cv2.circle(img, pts[i], th + 2, _color(i), -1, cv2.LINE_AA)


def _text(img, txt, org, scale, color, thick):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_fields(img, fields, active, selected):
    h, w = img.shape[:2]
    boxes = [(f, (int(f.rect[0] * w), int(f.rect[1] * h)), (int(f.rect[2] * w), int(f.rect[3] * h)))
             for f in fields]
    if any(f.name in active for f, _, _ in boxes):
        fill = img.copy()
        for f, a, b in boxes:
            if f.name in active:
                cv2.rectangle(fill, a, b, (80, 220, 80), -1)
        cv2.addWeighted(fill, 0.3, img, 0.7, 0, dst=img)
    for f, a, b in boxes:
        color = (80, 220, 80) if f.name in active else (0, 215, 255)
        cv2.rectangle(img, a, b, color, 4 if f.name == selected else 2, cv2.LINE_AA)
        _text(img, ascii_text(f.name), (a[0] + 6, a[1] + 26), 0.7, color, 2)


def draw_overlay(img, ov):
    h, w = img.shape[:2]
    if ov[0] == "countdown":
        txt = str(ov[1])
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 7, 14)
        _text(img, txt, ((w - tw) // 2, (h + tht) // 2), 7, (255, 255, 255), 14)
        _text(img, "POSE EINNEHMEN", (20, 50), 1.2, (0, 220, 255), 3)
    elif ov[0] == "rec":
        cv2.circle(img, (35, 40), 14, (0, 0, 255), -1, cv2.LINE_AA)
        _text(img, f"AUFNAHME  ({ov[2]} Samples)", (60, 52), 1.1, (0, 0, 255), 3)
        cv2.rectangle(img, (0, h - 14), (int(w * ov[1]), h), (0, 0, 255), -1)


# --------------------------------------------------------------------------
# Kamera-/Erkennungs-Thread
# --------------------------------------------------------------------------

class Tracker(threading.Thread):
    def __init__(self, settings):
        super().__init__(daemon=True)
        self.settings = settings
        self.lock = threading.Lock()
        self.frame = None
        self.status = {}
        self.events = queue.Queue()
        self.models, self.infos = {}, {}
        self.fields = []
        self.motions = []
        self.motion_trackers = {}  # Bereich -> MotionTracker
        self.selected_field = None
        self.enabled = False
        self.rec = None
        self.sender = KeySender()
        self.sender.start()
        self.engine = ActionEngine(self.sender)
        self._stop_evt = threading.Event()
        self._reopen = True
        self._models_changed = False
        self._toggle_was_down = False
        self.cand, self.stable = {}, {}
        self.last_action = ""
        self.foreign_title = ""
        self.info = "Starte ..."
        self.error = ""

    # --- Aufrufe aus der Oberfläche ---
    def set_models(self, models, infos, motions=()):
        with self.lock:
            self.models, self.infos, self.motions = models, infos, list(motions)
            self._models_changed = True

    def set_fields(self, fields):
        with self.lock:
            self.fields = fields
            self._models_changed = True

    def request_reopen(self):
        self._reopen = True

    def stop(self):
        self._stop_evt.set()

    def snapshot(self):
        with self.lock:
            return self.frame, self.status

    def start_recording(self, target, countdown, duration):
        now = time.monotonic()
        self.rec = {"target": target, "start": now + countdown, "end": now + countdown + duration,
                    "samples": [], "phase": "countdown", "last_sec": None}

    def cancel_recording(self):
        self.rec = None

    # --- intern ---
    def _check_toggle(self):
        vk = TOGGLE_KEYS.get(self.settings.get("toggle"), 0x7B)
        down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
        if down and not self._toggle_was_down:
            self.enabled = not self.enabled
            beep((1200, 90), (1500, 90)) if self.enabled else beep((700, 90), (500, 120))
            self.events.put(("toggled", self.enabled))
        self._toggle_was_down = down

    def _handle_recording(self, raw, now):
        rec = self.rec
        if rec is None:
            return None
        if now < rec["start"]:
            sec = math.ceil(rec["start"] - now)
            if sec != rec["last_sec"]:
                rec["last_sec"] = sec
                beep((660, 110))
            return ("countdown", sec)
        if now < rec["end"]:
            if rec["phase"] == "countdown":
                rec["phase"] = "rec"
                beep((1000, 300))
            if raw is not None:
                rec["samples"].append(np.round(raw, 4).tolist())
            frac = (now - rec["start"]) / (rec["end"] - rec["start"])
            return ("rec", frac, len(rec["samples"]))
        self.rec = None
        beep((1300, 100), (1600, 160))
        rec["target"]["fps"] = round(len(rec["samples"]) / max(rec["end"] - rec["start"], 0.1), 1)
        self.events.put(("recorded", rec["target"], rec["samples"]))
        return None

    def _open(self):
        self.error = ""
        self.info = "Starte Kamera ..."
        model_path = ensure_model(self.settings.get("modell", "full"),
                                  lambda msg: setattr(self, "info", msg))
        with open(model_path, "rb") as f:
            buf = f.read()
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=buf),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
        wanted = int(self.settings.get("kamera", 0))
        # Erst die gewählte Kamera, sonst automatisch die nächste funktionierende suchen.
        for idx in [wanted] + [i for i in range(6) if i != wanted]:
            self.info = f"Öffne Kamera {idx} ..."
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ok, _ = cap.read()
                if ok:
                    if idx != wanted:
                        self.settings["kamera"] = idx
                        self.events.put(("camera", idx))
                    self.info = ""
                    return cap, landmarker
            cap.release()
        landmarker.close()
        raise RuntimeError("Keine Kamera gefunden. Ist die Webcam angeschlossen und "
                           "nicht von einem anderen Programm belegt?")

    def _publish(self, frame_rgb, status):
        status.update(info=self.info, error=self.error)
        with self.lock:
            if frame_rgb is not None:
                self.frame = frame_rgb
            self.status = status

    def run(self):
        cap = landmarker = None
        last_ts, fails, fps, t_prev = 0, 0, 0.0, time.monotonic()
        try:
            while not self._stop_evt.is_set():
                if self._reopen:
                    self._reopen = False
                    self.engine.release_all()
                    if cap is not None:
                        cap.release()
                    if landmarker is not None:
                        landmarker.close()
                    cap = landmarker = None
                    try:
                        cap, landmarker = self._open()
                    except Exception as exc:  # noqa: BLE001
                        self.error = str(exc)
                        self.info = ""
                if cap is None:
                    self._publish(None, {})
                    self._stop_evt.wait(0.2)
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    fails += 1
                    if fails > 60:
                        self.error = "Kamera liefert keine Bilder (andere Kameranummer probieren?)"
                        self._publish(None, {})
                    time.sleep(0.03)
                    continue
                fails = 0
                self.error = ""
                try:
                    frame_rgb, status = self._process(frame, landmarker, last_ts)
                    last_ts = status.pop("_ts")
                except Exception as exc:  # noqa: BLE001
                    self.error = f"Fehler bei der Verarbeitung: {exc}"
                    self._publish(None, {})
                    time.sleep(0.1)
                    continue
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
                t_prev = now
                status["fps"] = fps
                self._publish(frame_rgb, status)
        finally:
            self.engine.release_all()
            self.sender.release_all()
            if cap is not None:
                cap.release()
            if landmarker is not None:
                landmarker.close()

    def _update_slot(self, slot, name, info, allowed, stable_n):
        """Entprellt einen Erkennungs-Slot (Körperbereich oder Feld) und löst Aktionen aus."""
        c_name, c_count = self.cand.get(slot, (None, 0))
        c_count = c_count + 1 if name == c_name else 1
        self.cand[slot] = (name, c_count)
        if allowed and c_count >= stable_n and self.stable.get(slot) != name:
            self.stable[slot] = name
            self.engine.set_pose(slot, info)
            if info is not None and info.combo:
                verb = "gehalten" if info.mode == "halten" else "angetippt"
                self.last_action = f"{info.name} → {info.keys} ({verb})"

    def _process(self, frame, landmarker, last_ts):
        frame = cv2.flip(frame, 1)  # Spiegelbild, fühlt sich natürlicher an
        h, w = frame.shape[:2]
        if w > 960:
            frame = cv2.resize(frame, (960, int(h * 960 / w)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
        aspect = w / h
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = max(int(time.monotonic() * 1000), last_ts + 1)
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        raw = None
        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            raw = np.array([[lms[i].x * aspect, lms[i].y, (lms[i].z or 0.0) * aspect,
                             1.0 if lms[i].visibility is None else lms[i].visibility]
                            for i in BODY_IDX], dtype=np.float32)

        now = time.monotonic()
        self._check_toggle()
        with self.lock:
            models, infos, fields, motions = self.models, self.infos, self.fields, self.motions
            changed, self._models_changed = self._models_changed, False
        if changed:
            self.engine.release_all()
            self.cand.clear()
            self.stable.clear()

        overlay = self._handle_recording(raw, now)

        k = int(self.settings.get("k", 5))
        threshold = float(self.settings.get("toleranz", 0.35))
        stable_n = max(1, int(self.settings.get("stabil", 4)))
        results = {z: (m.classify(raw, k, threshold) if raw is not None else ZoneResult(None, None, []))
                   for z, m in models.items()}

        title = foreground_title()
        if title and title != APP_TITLE:
            self.foreign_title = title
        filt = str(self.settings.get("fenster", "")).strip().lower()
        window_ok = (not filt) or (filt in title.lower())
        allowed = self.enabled and self.rec is None and window_ok

        for zone in ZONES:
            res = results.get(zone)
            name = res.name if res else None
            info = infos.get(name) if name else None
            if info is not None and info.zone == REST_ZONE:
                info = None
            self._update_slot(zone, name, info, allowed, stable_n)

        # Bewegungen: Stärke pro Bereich messen, mit kalibrierter Schwelle vergleichen
        energies = {}
        for zone in {m.zone for m in motions}:
            mt = self.motion_trackers.setdefault(zone, MotionTracker(zone))
            energies[zone] = mt.update(raw, now)
        motion_status = {}
        for m in motions:
            e = energies.get(m.zone)
            moving = e is not None and e >= m.threshold
            motion_status[m.name] = (e, m.threshold, moving)
            self._update_slot("bew:" + m.name, m.name if moving else None,
                              m.info if moving else None, allowed, stable_n)

        active_fields = set()
        for f in fields:
            inside = raw is not None and field_hit(raw, aspect, f)
            if inside:
                active_fields.add(f.name)
            self._update_slot("feld:" + f.name, f.name if inside else None,
                              f.info if inside else None, allowed, stable_n)

        if not allowed and (self.engine.active or self.stable):
            self.engine.release_all()
            self.stable.clear()

        if fields:
            draw_fields(frame, fields, active_fields, self.selected_field)
        if raw is not None:
            draw_skeleton(frame, raw, aspect)
        if overlay:
            draw_overlay(frame, overlay)

        status = {
            "_ts": ts,
            "person": raw is not None,
            "enabled": self.enabled,
            "window_ok": window_ok,
            "recording": self.rec is not None,
            "zones": {z: {"pose": r.name, "dist": r.dist, "ranking": r.ranking[:4]}
                      for z, r in results.items()},
            "fields": sorted(active_fields),
            "motions": motion_status,
            "held": self.engine.held_text(),
            "last_action": self.last_action,
            "foreign_title": self.foreign_title,
        }
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), status


# --------------------------------------------------------------------------
# Daten
# --------------------------------------------------------------------------
def load_data():
    data = {"version": 1, "settings": {}, "poses": [], "felder": []}
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8-sig") as f:  # verträgt auch Notepad-BOM
                data.update(json.load(f))
        except (OSError, ValueError) as exc:
            messagebox.showwarning(APP_TITLE, f"posen.json konnte nicht gelesen werden:\n{exc}")
    settings = dict(DEFAULT_SETTINGS)
    settings.update(data.get("settings") or {})
    data["settings"] = settings
    return data


def save_data(data):
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, DATA_FILE)


# --------------------------------------------------------------------------
# Oberfläche
# --------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.data = load_data()
        self.settings = self.data["settings"]
        self.tracker = Tracker(self.settings)
        self._photo = None
        self._save_job = None
        self._closing = False
        self._disp = None          # (x-Versatz, y-Versatz, Breite, Höhe) des angezeigten Bildes
        self._draw_mode = False
        self._drag_start = None
        self._drag_rect = None
        self._pending_rect = None  # aufgezogenes, noch nicht gespeichertes Rechteck

        root.title(APP_TITLE)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1500, sw - 60)}x{min(900, sh - 100)}+20+20")
        root.minsize(900, 600)

        self._build_ui()
        self._refresh_tree()
        self._refresh_ftree()
        self._rebuild()
        self.tracker.start()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(30, self._tick)

    # ---------- Aufbau ----------
    def _build_ui(self):
        r = self.root
        r.columnconfigure(0, weight=1)
        r.rowconfigure(0, weight=1)

        left = ttk.Frame(r, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.video = tk.Label(left, bg="#111", fg="#bbb", text="Kamera startet ...",
                              bd=0, highlightthickness=0, font=("Segoe UI", 12))
        self.video.grid(row=0, column=0, sticky="nsew")
        self.video.bind("<ButtonPress-1>", self._mouse_down)
        self.video.bind("<B1-Motion>", self._mouse_move)
        self.video.bind("<ButtonRelease-1>", self._mouse_up)
        self.rank_lbl = ttk.Label(left, text="", font=("Consolas", 10), justify="left")
        self.rank_lbl.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(left, foreground="#777",
                  text="Der Kopf wird nicht ausgewertet – nur Schultern, Arme, Hände, Hüfte, Beine, Füße."
                  ).grid(row=2, column=0, sticky="w")

        right = ttk.Frame(r, padding=(0, 8, 8, 8), width=430)
        right.grid(row=0, column=1, sticky="ns")

        # Status
        st = ttk.LabelFrame(right, text="Status", padding=8)
        st.pack(fill="x")
        self.state_lbl = tk.Label(st, text="", font=("Segoe UI", 12, "bold"), anchor="w",
                                  justify="left", wraplength=400)
        self.state_lbl.pack(fill="x")
        self.enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(st, text="Steuerung aktiv (sendet Tasten ans Spiel)",
                        variable=self.enabled_var, command=self._on_enable).pack(anchor="w", pady=(4, 2))
        self.zone_lbl = ttk.Label(st, text="", justify="left")
        self.zone_lbl.pack(anchor="w")
        self.action_lbl = ttk.Label(st, text="", justify="left", wraplength=400)
        self.action_lbl.pack(anchor="w", pady=(4, 0))
        self.window_lbl = ttk.Label(st, text="", foreground="#666", wraplength=400, justify="left")
        self.window_lbl.pack(anchor="w", pady=(4, 0))

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True, pady=(8, 0))
        poses_tab = ttk.Frame(nb, padding=8)
        fields_tab = ttk.Frame(nb, padding=8)
        settings_tab = ttk.Frame(nb, padding=8)
        nb.add(poses_tab, text="Posen")
        nb.add(fields_tab, text="Kamera-Felder")
        nb.add(settings_tab, text="Einstellungen")
        self._build_poses_tab(poses_tab)
        self._build_fields_tab(fields_tab)
        self._build_settings_tab(settings_tab)

    def _build_fields_tab(self, f):
        cols = ("teil", "taste", "modus")
        self.ftree = ttk.Treeview(f, columns=cols, height=6, selectmode="browse")
        self.ftree.heading("#0", text="Feld")
        self.ftree.heading("teil", text="Körperteil")
        self.ftree.heading("taste", text="Taste")
        self.ftree.heading("modus", text="Modus")
        self.ftree.column("#0", width=120)
        self.ftree.column("teil", width=150)
        self.ftree.column("taste", width=70)
        self.ftree.column("modus", width=70)
        self.ftree.pack(fill="x")
        self.ftree.bind("<<TreeviewSelect>>", self._on_field_select)
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 10))
        ttk.Button(row, text="Löschen", command=self._delete_field).pack(side="left")

        ed = ttk.LabelFrame(f, text="Feld anlegen / bearbeiten", padding=8)
        ed.pack(fill="x")
        ed.columnconfigure(1, weight=1)
        ttk.Button(ed, text="▭ Rechteck im Kamerabild aufziehen", command=self._start_draw
                   ).grid(row=0, column=0, columnspan=2, sticky="w")
        self.draw_lbl = ttk.Label(ed, text="", foreground="#1f618d", wraplength=380, justify="left")
        self.draw_lbl.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 6))
        self.fname_var = tk.StringVar()
        self.fpart_var = tk.StringVar(value=PART_LABELS["haende"])
        self.fkey_var = tk.StringVar()
        self.fmode_var = tk.StringVar(value=MODE_LABELS["antippen"])
        ttk.Label(ed, text="Name").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(ed, textvariable=self.fname_var).grid(row=2, column=1, sticky="ew", pady=2)
        ttk.Label(ed, text="Körperteil").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Combobox(ed, textvariable=self.fpart_var, state="readonly",
                     values=list(PART_LABELS.values())).grid(row=3, column=1, sticky="ew", pady=2)
        ttk.Label(ed, text="Taste").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Entry(ed, textvariable=self.fkey_var).grid(row=4, column=1, sticky="ew", pady=2)
        ttk.Label(ed, text="Modus").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Combobox(ed, textvariable=self.fmode_var, state="readonly",
                     values=list(MODE_LABELS.values())).grid(row=5, column=1, sticky="ew", pady=2)
        b = ttk.Frame(ed)
        b.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(b, text="Neues Feld speichern", command=self._save_field).pack(side="left")
        ttk.Button(b, text="Änderungen übernehmen", command=self._apply_field).pack(side="left", padx=4)

        ttk.Label(f, foreground="#555", wraplength=400, justify="left",
                  text="Ein Feld löst aus, sobald das gewählte Körperteil im Rechteck ist – "
                       "z. B. Hand nach oben rechts strecken oder seitlich in ein Feld treten. "
                       "Das Kamerabild ist gespiegelt wie ein Spiegel. Felder funktionieren "
                       "gleichzeitig mit den Posen. Zum Verschieben: Feld auswählen, neues "
                       "Rechteck aufziehen, „Änderungen übernehmen“."
                  ).pack(fill="x", pady=(10, 0))

    def _build_poses_tab(self, f):
        cols = ("bereich", "art", "taste", "modus", "samples")
        self.tree = ttk.Treeview(f, columns=cols, height=7, selectmode="browse")
        self.tree.heading("#0", text="Pose")
        self.tree.heading("bereich", text="Bereich")
        self.tree.heading("taste", text="Taste")
        self.tree.heading("modus", text="Modus")
        self.tree.heading("art", text="Erkennung")
        self.tree.heading("samples", text="Samples")
        self.tree.column("#0", width=110)
        self.tree.column("bereich", width=100)
        self.tree.column("art", width=70)
        self.tree.column("taste", width=50)
        self.tree.column("modus", width=65)
        self.tree.column("samples", width=55, anchor="e")
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 10))
        ttk.Button(row, text="Samples hinzufügen", command=self._record_more).pack(side="left")
        ttk.Button(row, text="Löschen", command=self._delete).pack(side="left", padx=4)

        ed = ttk.LabelFrame(f, text="Pose anlegen / bearbeiten", padding=8)
        ed.pack(fill="x")
        ed.columnconfigure(1, weight=1)
        self.name_var = tk.StringVar()
        self.zone_var = tk.StringVar(value=ZONE_LABELS["arme"])
        self.key_var = tk.StringVar()
        self.mode_var = tk.StringVar(value=MODE_LABELS["antippen"])
        ttk.Label(ed, text="Name").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(ed, textvariable=self.name_var).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(ed, text="Bereich").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(ed, textvariable=self.zone_var, state="readonly",
                     values=[ZONE_LABELS[z] for z in ZONES]).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(ed, text="Taste").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(ed, textvariable=self.key_var).grid(row=2, column=1, sticky="ew", pady=2)
        ttk.Label(ed, foreground="#777", wraplength=340, justify="left",
                  text="z. B. w · 1 · space · shift+2 · f1 · lmb / rmb (Maus) · num5 · "
                       "links/rechts/hoch/runter. Leer = keine Aktion."
                  ).grid(row=3, column=1, sticky="w")
        ttk.Label(ed, text="Modus").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Combobox(ed, textvariable=self.mode_var, state="readonly",
                     values=list(MODE_LABELS.values())).grid(row=4, column=1, sticky="ew", pady=2)

        self.art_var = tk.StringVar(value=ART_LABELS["haltung"])
        ttk.Label(ed, text="Erkennung").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Combobox(ed, textvariable=self.art_var, state="readonly",
                     values=list(ART_LABELS.values())).grid(row=5, column=1, sticky="ew", pady=2)
        ttk.Label(ed, text="Empfindlichk.").grid(row=6, column=0, sticky="w", pady=2)
        emp = ttk.Frame(ed)
        emp.grid(row=6, column=1, sticky="ew", pady=2)
        self.empf_var = tk.DoubleVar(value=1.0)
        self.empf_lbl = ttk.Label(emp, text="1.0", width=4)
        ttk.Scale(emp, from_=0.3, to=2.0, variable=self.empf_var,
                  command=lambda _=None: self.empf_lbl.config(text=f"{self.empf_var.get():.1f}")
                  ).pack(side="left", fill="x", expand=True)
        self.empf_lbl.pack(side="left", padx=(6, 0))
        ttk.Label(ed, foreground="#777", wraplength=340, justify="left",
                  text="Kleiner = löst schwerer aus, größer = leichter. "
                       "Nach Änderung „Änderungen übernehmen“ klicken."
                  ).grid(row=7, column=1, sticky="w")

        b = ttk.Frame(ed)
        b.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(b, text="● Neue Pose aufnehmen", command=self._record_new).pack(side="left")
        ttk.Button(b, text="Änderungen übernehmen", command=self._apply_edit).pack(side="left", padx=4)
        b2 = ttk.Frame(ed)
        b2.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(b2, text="Ruhepose aufnehmen", command=self._record_rest).pack(side="left")
        ttk.Button(b2, text="Aufnahme abbrechen", command=self._cancel_rec).pack(side="left", padx=4)

        ttk.Label(f, foreground="#555", wraplength=400, justify="left",
                  text="Tipp: Nimm zuerst die Ruhepose auf (normal stehen/sitzen). "
                       "Bewege dich während der Aufnahme leicht innerhalb der Pose – "
                       "das macht die Erkennung robuster."
                  ).pack(fill="x", pady=(10, 0))

    def _build_settings_tab(self, f):
        f.columnconfigure(1, weight=1)
        s = self.settings
        r = 0

        ttk.Label(f, text="Kamera-Nr.").grid(row=r, column=0, sticky="w", pady=3)
        self.cam_var = tk.IntVar(value=int(s["kamera"]))
        ttk.Spinbox(f, from_=0, to=9, width=5, textvariable=self.cam_var).grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Label(f, text="Pose-Modell").grid(row=r, column=0, sticky="w", pady=3)
        self.model_var = tk.StringVar(value=s["modell"])
        ttk.Combobox(f, textvariable=self.model_var, state="readonly", width=8,
                     values=["full", "lite"]).grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Button(f, text="Kamera / Modell anwenden", command=self._apply_camera
                   ).grid(row=r, column=1, sticky="w", pady=(2, 10))
        r += 1

        ttk.Label(f, text="Toleranz").grid(row=r, column=0, sticky="w", pady=3)
        tol = ttk.Frame(f)
        tol.grid(row=r, column=1, sticky="ew")
        self.tol_var = tk.DoubleVar(value=float(s["toleranz"]))
        self.tol_lbl = ttk.Label(tol, text=f"{s['toleranz']:.2f}", width=5)
        ttk.Scale(tol, from_=0.05, to=1.0, variable=self.tol_var, command=self._on_tol
                  ).pack(side="left", fill="x", expand=True)
        self.tol_lbl.pack(side="left", padx=(6, 0))
        r += 1
        ttk.Label(f, foreground="#777", wraplength=300, justify="left",
                  text="Kleiner = strenger. Die Zahlen unter dem Kamerabild zeigen, "
                       "wie nah du jeder Pose gerade bist.").grid(row=r, column=1, sticky="w")
        r += 1

        self.stable_var = tk.IntVar(value=int(s["stabil"]))
        self.countdown_var = tk.IntVar(value=int(s["countdown"]))
        self.dur_var = tk.IntVar(value=int(s["dauer"]))
        for label, var, lo, hi in (("Stabilität (Frames)", self.stable_var, 1, 15),
                                   ("Countdown (s)", self.countdown_var, 0, 20),
                                   ("Aufnahmedauer (s)", self.dur_var, 1, 15)):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Spinbox(f, from_=lo, to=hi, width=5, textvariable=var).grid(row=r, column=1, sticky="w")
            r += 1
        self._bind_setting(self.stable_var, "stabil", int)
        self._bind_setting(self.countdown_var, "countdown", int)
        self._bind_setting(self.dur_var, "dauer", int)

        ttk.Label(f, text="Spielfenster-Titel").grid(row=r, column=0, sticky="w", pady=(10, 3))
        self.win_var = tk.StringVar(value=s["fenster"])
        ttk.Entry(f, textvariable=self.win_var).grid(row=r, column=1, sticky="ew", pady=(10, 3))
        self._bind_setting(self.win_var, "fenster", str)
        r += 1
        ttk.Label(f, foreground="#777", wraplength=300, justify="left",
                  text="Tasten werden nur gesendet, wenn das aktive Fenster diesen Text im Titel hat. "
                       "Leer = an jedes Fenster (Vorsicht).").grid(row=r, column=1, sticky="w")
        r += 1
        ttk.Label(f, text="Ein/Aus-Taste").grid(row=r, column=0, sticky="w", pady=(10, 3))
        self.toggle_var = tk.StringVar(value=s["toggle"])
        ttk.Combobox(f, textvariable=self.toggle_var, state="readonly", width=8,
                     values=list(TOGGLE_KEYS)).grid(row=r, column=1, sticky="w", pady=(10, 3))
        self._bind_setting(self.toggle_var, "toggle", str)

    def _bind_setting(self, var, key, conv):
        def cb(*_):
            try:
                self.settings[key] = conv(var.get())
            except (tk.TclError, ValueError):
                return
            self._schedule_save()
        var.trace_add("write", cb)

    # ---------- Daten ----------
    def _schedule_save(self):
        if self._save_job:
            self.root.after_cancel(self._save_job)
        self._save_job = self.root.after(600, self._save_now)

    def _save_now(self):
        self._save_job = None
        try:
            save_data(self.data)
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"Speichern fehlgeschlagen:\n{exc}")

    def _find(self, name):
        for p in self.data["poses"]:
            if p["name"].lower() == name.lower():
                return p
        return None

    def _rebuild(self):
        models, infos, motions = build_models(self.data["poses"])
        self.tracker.set_models(models, infos, motions)
        self.tracker.set_fields(build_fields(self.data["felder"]))

    # ---------- Kamera-Felder ----------
    def _refresh_ftree(self, select=None):
        self.ftree.delete(*self.ftree.get_children())
        for fd in self.data["felder"]:
            self.ftree.insert("", "end", iid=fd["name"], text=fd["name"], values=(
                PART_LABELS.get(fd.get("part"), "?"), fd.get("keys") or "–",
                MODE_SHORT.get(fd.get("mode"), "")))
        if select and self.ftree.exists(select):
            self.ftree.selection_set(select)
            self.ftree.see(select)

    def _find_field(self, name):
        return next((fd for fd in self.data["felder"] if fd["name"].lower() == name.lower()), None)

    def _selected_field(self):
        sel = self.ftree.selection()
        return self._find_field(sel[0]) if sel else None

    def _on_field_select(self, _=None):
        fd = self._selected_field()
        if not fd:
            return
        self.fname_var.set(fd["name"])
        self.fpart_var.set(PART_LABELS.get(fd.get("part"), PART_LABELS["haende"]))
        self.fkey_var.set(fd.get("keys", ""))
        self.fmode_var.set(MODE_LABELS.get(fd.get("mode"), MODE_LABELS["antippen"]))
        self.tracker.selected_field = fd["name"]
        self._pending_rect = None
        self.draw_lbl.config(text="")

    def _start_draw(self):
        self._draw_mode = True
        self._drag_start = self._drag_rect = None
        self.video.config(cursor="crosshair")
        self.draw_lbl.config(text="Jetzt im Kamerabild mit gedrückter Maustaste ein Rechteck ziehen.")

    def _to_norm(self, ev):
        if not self._disp:
            return None
        ox, oy, iw, ih = self._disp
        return (min(max((ev.x - ox) / iw, 0.0), 1.0), min(max((ev.y - oy) / ih, 0.0), 1.0))

    def _mouse_down(self, ev):
        if not self._draw_mode:
            return
        p = self._to_norm(ev)
        if p is not None:
            self._drag_start = p
            self._drag_rect = (p[0], p[1], p[0], p[1])

    def _mouse_move(self, ev):
        if not self._draw_mode or self._drag_start is None:
            return
        p = self._to_norm(ev)
        if p is None:
            return
        (ax, ay), (bx, by) = self._drag_start, p
        self._drag_rect = (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))

    def _mouse_up(self, ev):
        if not self._draw_mode or self._drag_start is None:
            return
        self._mouse_move(ev)
        r, self._drag_start, self._drag_rect = self._drag_rect, None, None
        if r[2] - r[0] < 0.03 or r[3] - r[1] < 0.03:
            self.draw_lbl.config(text="Das Rechteck ist zu klein – bitte noch einmal ziehen.")
            return
        self._pending_rect = tuple(round(v, 4) for v in r)
        self._draw_mode = False
        self.video.config(cursor="")
        if self._selected_field():
            self.draw_lbl.config(text="Neues Rechteck gesetzt (weiß). „Änderungen übernehmen“ verschiebt "
                                      "das gewählte Feld hierher, „Neues Feld speichern“ legt ein neues an.")
        else:
            self.draw_lbl.config(text="Rechteck gesetzt (weiß). Jetzt Name, Körperteil und Taste "
                                      "eintragen und „Neues Feld speichern“.")

    def _field_form(self):
        name = self.fname_var.get().strip()
        if not name:
            messagebox.showerror(APP_TITLE, "Bitte einen Namen für das Feld eingeben.")
            return None
        keys = self.fkey_var.get().strip().lower()
        try:
            parse_combo(keys)
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return None
        return {"name": name, "part": PART_FROM_LABEL.get(self.fpart_var.get(), "haende"),
                "keys": keys, "mode": MODE_FROM_LABEL.get(self.fmode_var.get(), "antippen")}

    def _fields_changed(self, select=None):
        self._save_now()
        self._refresh_ftree(select=select)
        self._rebuild()

    def _save_field(self):
        form = self._field_form()
        if not form:
            return
        if self._pending_rect is None:
            messagebox.showinfo(APP_TITLE, "Bitte zuerst mit „▭ Rechteck im Kamerabild aufziehen“ "
                                           "ein Feld im Kamerabild markieren.")
            return
        if self._find_field(form["name"]):
            messagebox.showerror(APP_TITLE, f"Es gibt schon ein Feld „{form['name']}“.")
            return
        form["rect"] = list(self._pending_rect)
        self.data["felder"].append(form)
        self._pending_rect = None
        self.draw_lbl.config(text="")
        self.tracker.selected_field = form["name"]
        self._fields_changed(select=form["name"])

    def _apply_field(self):
        fd = self._selected_field()
        if not fd:
            messagebox.showinfo(APP_TITLE, "Bitte zuerst ein Feld in der Liste auswählen.")
            return
        form = self._field_form()
        if not form:
            return
        other = self._find_field(form["name"])
        if other is not None and other is not fd:
            messagebox.showerror(APP_TITLE, f"Es gibt schon ein Feld „{other['name']}“.")
            return
        fd.update(form)
        if self._pending_rect is not None:
            fd["rect"] = list(self._pending_rect)
            self._pending_rect = None
        self.draw_lbl.config(text="")
        self.tracker.selected_field = fd["name"]
        self._fields_changed(select=fd["name"])

    def _delete_field(self):
        fd = self._selected_field()
        if fd and messagebox.askyesno(APP_TITLE, f"Feld „{fd['name']}“ löschen?"):
            self.data["felder"].remove(fd)
            self.tracker.selected_field = None
            self._fields_changed()

    def _refresh_tree(self, select=None):
        self.tree.delete(*self.tree.get_children())
        for p in self.data["poses"]:
            rest = p["zone"] == REST_ZONE
            self.tree.insert("", "end", iid=p["name"], text=p["name"], values=(
                "Ruhepose" if rest else ZONE_LABELS[p["zone"]],
                "–" if rest else ART_SHORT.get(p.get("art", "haltung"), ""),
                "–" if rest or not p.get("keys") else p["keys"],
                "–" if rest else MODE_SHORT.get(p.get("mode"), ""),
                len(p["samples"])))
        if select and self.tree.exists(select):
            self.tree.selection_set(select)
            self.tree.see(select)

    def _selected(self):
        sel = self.tree.selection()
        return self._find(sel[0]) if sel else None

    # ---------- Aktionen ----------
    def _on_enable(self):
        self.tracker.enabled = self.enabled_var.get()

    def _on_tol(self, _=None):
        v = round(self.tol_var.get(), 2)
        self.settings["toleranz"] = v
        self.tol_lbl.config(text=f"{v:.2f}")
        self._schedule_save()

    def _apply_camera(self):
        try:
            self.settings["kamera"] = int(self.cam_var.get())
        except (tk.TclError, ValueError):
            return
        self.settings["modell"] = self.model_var.get()
        self._save_now()
        self.tracker.request_reopen()

    def _on_select(self, _=None):
        p = self._selected()
        if not p:
            return
        self.name_var.set(p["name"])
        if p["zone"] != REST_ZONE:
            self.zone_var.set(ZONE_LABELS[p["zone"]])
            self.mode_var.set(MODE_LABELS.get(p.get("mode"), MODE_LABELS["antippen"]))
            self.art_var.set(ART_LABELS.get(p.get("art", "haltung"), ART_LABELS["haltung"]))
            self.empf_var.set(float(p.get("empf", 1.0)))
            self.empf_lbl.config(text=f"{float(p.get('empf', 1.0)):.1f}")
        self.key_var.set(p.get("keys", ""))

    def _form(self):
        """Liest das Formular, gibt dict oder None (mit Fehlermeldung) zurück."""
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror(APP_TITLE, "Bitte einen Namen für die Pose eingeben.")
            return None
        zone = ZONE_FROM_LABEL.get(self.zone_var.get(), "arme")
        keys = self.key_var.get().strip().lower()
        try:
            parse_combo(keys)
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return None
        mode = MODE_FROM_LABEL.get(self.mode_var.get(), "antippen")
        art = ART_FROM_LABEL.get(self.art_var.get(), "haltung")
        empf = round(float(self.empf_var.get()), 2)
        return {"name": name, "zone": zone, "keys": keys, "mode": mode, "art": art, "empf": empf}

    def _busy(self):
        if self.tracker.rec is not None:
            messagebox.showinfo(APP_TITLE, "Es läuft bereits eine Aufnahme.")
            return True
        return False

    def _start_rec(self, target):
        if self._busy():
            return
        self.tracker.start_recording(target, int(self.settings["countdown"]),
                                     int(self.settings["dauer"]))

    def _record_new(self):
        form = self._form()
        if not form:
            return
        existing = self._find(form["name"])
        if existing:
            if not messagebox.askyesno(APP_TITLE, f"„{existing['name']}“ gibt es schon. "
                                                  "Weitere Samples zu dieser Pose aufnehmen?"):
                return
            form = {k: existing.get(k) for k in POSE_KEYS if k in existing}
        self._start_rec(form)

    def _record_more(self):
        p = self._selected()
        if not p:
            messagebox.showinfo(APP_TITLE, "Bitte zuerst eine Pose in der Liste auswählen.")
            return
        self._start_rec({k: p[k] for k in POSE_KEYS if k in p})

    def _record_rest(self):
        rest = next((p for p in self.data["poses"] if p["zone"] == REST_ZONE), None)
        name = rest["name"] if rest else "Ruhepose"
        self._start_rec({"name": name, "zone": REST_ZONE, "keys": "", "mode": "antippen"})

    def _cancel_rec(self):
        self.tracker.cancel_recording()

    def _on_recorded(self, target, samples):
        if len(samples) < 5:
            messagebox.showwarning(
                APP_TITLE, "Während der Aufnahme wurde kaum ein Körper erkannt.\n\n"
                           "Stell dich so hin, dass die Kamera dich mindestens von den Schultern "
                           "bis zur Hüfte sieht (für Bein-Posen ganz), und versuche es erneut.")
            return
        pose = self._find(target["name"])
        if pose is None:
            pose = dict(target)
            pose["samples"] = []
            self.data["poses"].append(pose)
        pose.setdefault("fps", target.get("fps", 30.0))
        pose["samples"].extend(samples)
        self._save_now()
        self._refresh_tree(select=pose["name"])
        self._rebuild()
        if pose["zone"] == REST_ZONE:  # neue Ruhepose: alle Posen dagegen prüfen
            self._warn_similar([p["name"] for p in self.data["poses"] if p is not pose])
        else:
            self._warn_similar([pose["name"]])

    def _warn_similar(self, names):
        msgs = [m for m in (similarity_warning(self.data["poses"], n) for n in names) if m]
        if msgs:
            messagebox.showwarning(APP_TITLE, "Achtung, Verwechslungsgefahr:\n\n" + "\n\n".join(msgs))

    def _apply_edit(self):
        p = self._selected()
        if not p:
            messagebox.showinfo(APP_TITLE, "Bitte zuerst eine Pose in der Liste auswählen.")
            return
        form = self._form()
        if not form:
            return
        other = self._find(form["name"])
        if other is not None and other is not p:
            messagebox.showerror(APP_TITLE, f"Es gibt schon eine Pose „{other['name']}“.")
            return
        p["name"] = form["name"]
        if p["zone"] != REST_ZONE:
            p.update(zone=form["zone"], keys=form["keys"], mode=form["mode"],
                     art=form["art"], empf=form["empf"])
        self._save_now()
        self._refresh_tree(select=p["name"])
        self._rebuild()
        self._warn_similar([p["name"]])

    def _delete(self):
        p = self._selected()
        if not p:
            return
        if messagebox.askyesno(APP_TITLE, f"Pose „{p['name']}“ mit {len(p['samples'])} Samples löschen?"):
            self.data["poses"].remove(p)
            self._save_now()
            self._refresh_tree()
            self._rebuild()

    # ---------- Laufende Aktualisierung ----------
    def _tick(self):
        if self._closing:
            return
        try:
            frame, st = self.tracker.snapshot()
            if frame is not None:
                lw, lh = self.video.winfo_width() - 4, self.video.winfo_height() - 4
                if lw > 20 and lh > 20:
                    fh, fw = frame.shape[:2]
                    scale = min(lw / fw, lh / fh)
                    iw, ih = max(1, int(fw * scale)), max(1, int(fh * scale))
                    img = Image.fromarray(frame).resize((iw, ih), Image.BILINEAR)
                    self._disp = ((self.video.winfo_width() - iw) // 2,
                                  (self.video.winfo_height() - ih) // 2, iw, ih)
                    rect = self._drag_rect or self._pending_rect
                    if rect:
                        ImageDraw.Draw(img).rectangle(
                            [rect[0] * iw, rect[1] * ih, rect[2] * iw, rect[3] * ih],
                            outline=(255, 255, 255), width=3)
                    self._photo = ImageTk.PhotoImage(img)
                    self.video.configure(image=self._photo, text="")
            self._update_status(st or {})
            while True:
                try:
                    ev = self.tracker.events.get_nowait()
                except queue.Empty:
                    break
                if ev[0] == "toggled":
                    self.enabled_var.set(ev[1])
                elif ev[0] == "recorded":
                    self._on_recorded(ev[1], ev[2])
                elif ev[0] == "camera":
                    self.cam_var.set(ev[1])
                    self._schedule_save()
        finally:
            self.root.after(30, self._tick)

    def _update_status(self, st):
        toggle = self.settings.get("toggle", "F12")
        if st.get("error"):
            self.state_lbl.config(text="⚠ " + st["error"], fg="#c0392b")
        elif st.get("info"):
            self.state_lbl.config(text=st["info"], fg="#555")
        elif st.get("recording"):
            self.state_lbl.config(text="● Aufnahme – Pose einnehmen und halten", fg="#c0392b")
        elif st.get("enabled"):
            if st.get("window_ok"):
                self.state_lbl.config(text=f"STEUERUNG AKTIV   ({toggle} = aus)", fg="#1e8449")
            else:
                self.state_lbl.config(text=f"Aktiv – wartet auf das Spielfenster   ({toggle} = aus)",
                                      fg="#ca6f1e")
        else:
            self.state_lbl.config(text=f"Steuerung aus   ({toggle} = an)", fg="#777")

        zones = st.get("zones", {})
        lines = []
        if not st.get("person", False) and "zones" in st:
            lines.append("Keine Person im Bild.")
        else:
            for z, info in zones.items():
                name = info["pose"] or "–"
                d = f"  ({info['dist']:.2f})" if info["dist"] is not None and info["pose"] else ""
                lines.append(f"{ZONE_LABELS[z]}: {name}{d}")
            if not zones and not self.data["felder"]:
                lines.append("Noch keine Posen kalibriert." if not self.data["poses"]
                             else "Nimm noch eine Pose mit Bereich (nicht nur die Ruhepose) auf.")
        for name, (e, thr, moving) in st.get("motions", {}).items():
            val = "–" if e is None else f"{e:.2f}"
            lines.append(f"{name} (Bewegung): {val} / Schwelle {thr:.2f}" + ("  ✔ aktiv" if moving else ""))
        if self.data["felder"]:
            act = st.get("fields", [])
            lines.append("Kamera-Felder: " + (", ".join(act) if act else "–"))
        self.zone_lbl.config(text="\n".join(lines))

        parts = []
        if st.get("held"):
            parts.append(f"Gehalten: {st['held']}")
        if st.get("last_action"):
            parts.append(f"Zuletzt: {st['last_action']}")
        self.action_lbl.config(text="\n".join(parts))
        if st.get("foreign_title"):
            self.window_lbl.config(text=f"Zuletzt aktives Fenster: „{st['foreign_title']}“")

        rank_lines = []
        for z, info in zones.items():
            if info["ranking"]:
                items = "  ".join(f"{n} {d:.2f}" for n, d in info["ranking"])
                rank_lines.append(f"{ZONE_LABELS[z]:<16} {items}")
        if "fps" in st:
            rank_lines.append(f"{st['fps']:.0f} FPS")
        self.rank_lbl.config(text="\n".join(rank_lines))

    def on_close(self):
        if self._closing:
            return
        self._closing = True
        self._close_deadline = time.monotonic() + 6
        self.tracker.stop()
        self.root.withdraw()
        self._finish_close()

    def _finish_close(self):
        # Nicht mit join() blockieren: der Kameratreiber schickt beim Freigeben
        # Fensternachrichten, die die Oberfläche weiter verarbeiten muss.
        if self.tracker.is_alive() and time.monotonic() < self._close_deadline:
            self.root.after(50, self._finish_close)
            return
        self.tracker.sender.stop()
        self.tracker.sender.join(timeout=1)
        if self._save_job:
            self.root.after_cancel(self._save_job)
        self._save_now()
        self.root.destroy()


def main():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
    # Tasten sind losgelassen, Daten gespeichert. Sicherheitshalber hart beenden,
    # damit kein hängender Treiber-Thread den Prozess am Leben hält.
    k32 = ctypes.windll.kernel32
    k32.TerminateProcess(k32.GetCurrentProcess(), 0)


if __name__ == "__main__":
    main()
