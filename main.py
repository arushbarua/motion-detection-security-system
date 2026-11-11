"""
Motion Detection & Security Surveillance System
================================================
A webcam-based security dashboard built with Python, OpenCV and Tkinter.

Features
--------
* Live camera feed with moving regions highlighted by bounding boxes
* Adaptive background model (handles slow lighting drift) + sudden-light-change guard
* Automatic timestamped snapshots (with cooldown) saved to  captured_images/
* Automatic short video clips on motion saved to            recordings/
* Event history (persisted to event_log.csv) and a snapshot gallery with preview
* Adjustable sensitivity, minimum object size and snapshot cooldown (live)
* Sound alert, optional e-mail alert with the snapshot attached
* Keyboard shortcuts: SPACE = start/stop, S = snapshot, Q = quit

Run:  python main.py
"""

import csv
import os
import queue
import smtplib
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
SNAP_DIR = BASE_DIR / "captured_images"
REC_DIR = BASE_DIR / "recordings"
LOG_FILE = BASE_DIR / "event_log.csv"
SNAP_DIR.mkdir(exist_ok=True)
REC_DIR.mkdir(exist_ok=True)

PROC_WIDTH = 640          # frames are resized to this width for speed
WARMUP_FRAMES = 30        # frames used to learn the empty background
POST_ROLL_SECONDS = 5     # keep an event "open" this long after last movement
RECORD_FPS = 20

# Dark theme colours
BG = "#0f1419"
PANEL = "#171d25"
CARD = "#1e2631"
TEXT = "#e6edf3"
MUTED = "#8b98a9"
GREEN = "#1fa65a"
RED = "#e5484d"
GREY = "#3b4452"
ACCENT = "#3b82f6"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class Settings:
    """Values shared between the GUI thread and the camera thread."""

    sensitivity = 6      # 1 (low) .. 10 (high)
    min_area = 1500      # smallest moving region (pixels) that counts
    cooldown = 3         # seconds between snapshots during one event
    record = True
    sound = True
    email = False
    ai = True            # verify motion with the YOLO object detector
    conf = 0.45          # minimum AI confidence (0.25 .. 0.90)


# Things worth raising an alarm for (COCO class names understood by YOLO).
TARGET_NAMES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "bird", "cat", "dog", "horse", "sheep", "cow", "bear",
}
MODEL_NAME = os.environ.get("MOTION_MODEL", "yolov8n.pt")  # try yolov8s.pt for higher accuracy
MIN_MOTION_OVERLAP = 0.04   # share of a detected object's box that must be moving
CONFIRM_WINDOW = 3          # AI must agree in 2 of the last 3 checks before alarming


def log_event(kind: str, detail: str = "") -> str:
    """Append one row to the CSV log and return the timestamp used."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_file = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "event", "detail"])
        writer.writerow([stamp, kind, detail])
    return stamp


def read_log(limit: int = 300):
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    return rows[-limit:]


def open_path(path) -> None:
    path = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as exc:  # pragma: no cover
        messagebox.showerror("Open failed", str(exc))


def open_capture(index: int):
    """Open a camera, trying several Windows backends. Returns (capture, backend) or (None, None)."""
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            for _ in range(5):          # some cameras return empty frames while warming up
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap, backend
        cap.release()
    return None, None


def find_cameras(max_index: int = 5):
    """Scan camera numbers; return [(index, thumbnail_bgr, 'WxH'), ...] for the ones that work."""
    found = []
    for index in range(max_index):
        cap, _ = open_capture(index)
        if cap is None:
            continue
        frame = None
        for _ in range(8):
            ok, img = cap.read()
            if ok and img is not None:
                frame = img
        cap.release()
        if frame is not None:
            h, w = frame.shape[:2]
            found.append((index, cv2.resize(frame, (200, max(int(200 * h / w), 1))), f"{w}x{h}"))
    return found


def beep() -> None:
    if sys.platform.startswith("win"):
        import winsound

        winsound.Beep(1200, 250)
        winsound.Beep(900, 250)
    else:
        print("\a", end="", flush=True)


def send_email_alert(image_path: Path, when: str) -> None:
    """Send a snapshot by e-mail. Configure with environment variables:
    MOTION_SMTP_USER, MOTION_SMTP_PASS (Gmail app password), MOTION_ALERT_TO (optional)."""
    user = os.environ.get("MOTION_SMTP_USER")
    password = os.environ.get("MOTION_SMTP_PASS")
    to = os.environ.get("MOTION_ALERT_TO", user)
    if not (user and password and to):
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = f"Security alert: motion detected at {when}"
        msg["From"] = user
        msg["To"] = to
        msg.set_content(f"Motion was detected at {when}. Snapshot attached.")
        msg.add_attachment(
            image_path.read_bytes(), maintype="image", subtype="jpeg", filename=image_path.name
        )
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
            server.login(user, password)
            server.send_message(msg)
    except Exception as exc:
        print("E-mail alert failed:", exc)


# --------------------------------------------------------------------------- #
# Motion detection
# --------------------------------------------------------------------------- #
class MotionDetector:
    """Running-average background subtraction (more robust than raw frame differencing)."""

    def __init__(self) -> None:
        self.background = None
        self.last_mask = None   # binary motion mask of the latest frame

    def reset(self) -> None:
        self.background = None
        self.last_mask = None

    def process(self, frame, sensitivity: int, min_area: int):
        """Return a list of (x, y, w, h) boxes for regions that moved."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.background is None:
            self.background = gray.astype("float")
            self.last_mask = None
            return [], False

        cv2.accumulateWeighted(gray, self.background, 0.05)
        delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))

        threshold = max(8, 45 - int(sensitivity) * 4)
        mask = cv2.threshold(delta, threshold, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.dilate(mask, None, iterations=2)
        self.last_mask = mask

        # Sudden global change (lights switched on/off): relearn instead of alarming.
        if cv2.countNonZero(mask) > 0.6 * mask.size:
            self.background = gray.astype("float")
            self.last_mask = None
            return [], True

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
        return boxes, False


# --------------------------------------------------------------------------- #
# AI object detection (YOLO)
# --------------------------------------------------------------------------- #
class ObjectDetector:
    """Finds people, vehicles and animals so fans, doors and curtains are ignored."""

    def __init__(self) -> None:
        from ultralytics import YOLO  # imported here so the app still starts without it

        self.model = YOLO(MODEL_NAME)  # first run downloads the weights automatically
        names = self.model.names
        self.class_ids = [i for i, n in names.items() if n in TARGET_NAMES]

    def detect(self, frame, conf: float):
        """Return a list of (x, y, w, h, label, confidence)."""
        result = self.model.predict(frame, conf=conf, classes=self.class_ids,
                                    imgsz=640, verbose=False)[0]
        found = []
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            label = self.model.names[int(box.cls[0])]
            found.append((x1, y1, x2 - x1, y2 - y1, label, float(box.conf[0])))
        return found


def moving_share(mask, box) -> float:
    """Fraction of a box's area that contains motion pixels."""
    if mask is None:
        return 0.0
    x, y, w, h = box
    x, y = max(x, 0), max(y, 0)
    region = mask[y:y + h, x:x + w]
    return float(cv2.countNonZero(region)) / region.size if region.size else 0.0


# --------------------------------------------------------------------------- #
# Camera thread
# --------------------------------------------------------------------------- #
class CameraWorker(threading.Thread):
    def __init__(self, cam_index: int, frame_q: queue.Queue, event_q: queue.Queue):
        super().__init__(daemon=True)
        self.cam_index = cam_index
        self.frame_q = frame_q
        self.event_q = event_q
        self.stop_event = threading.Event()
        self.snapshot_requested = threading.Event()

    # -- drawing -----------------------------------------------------------
    @staticmethod
    def _overlay(img, boxes, recording, fps, learning, dets=(), ai_on=False):
        for (x, y, w, h) in boxes:
            if ai_on:   # motion that the AI did not confirm: thin grey box (ignored)
                cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y, w, h, label, conf) in dets:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            tag = f"{label.upper()} {conf:.0%}"
            cv2.rectangle(img, (x, max(y - 20, 0)), (x + 9 * len(tag), max(y, 20)), (0, 0, 255), -1)
            cv2.putText(img, tag, (x + 3, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if ai_on:
            cv2.putText(img, "AI VERIFIED", (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 220, 120), 1, cv2.LINE_AA)
        stamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(img, stamp, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{fps:4.1f} FPS", (img.shape[1] - 90, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        if learning:
            cv2.putText(img, "Calibrating background...", (8, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        if recording:
            cv2.circle(img, (img.shape[1] // 2 - 30, 13), 6, (0, 0, 255), -1)
            cv2.putText(img, "REC", (img.shape[1] // 2 - 18, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    def _push(self, frame):
        try:
            self.frame_q.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_q.put_nowait(frame)
            except queue.Full:
                pass

    # -- main loop ---------------------------------------------------------
    def run(self) -> None:
        cap, _ = open_capture(self.cam_index)
        if cap is None:
            self.event_q.put(("error", f"Could not open camera {self.cam_index}.\n"
                                       "Close other apps using the camera, check Windows camera privacy "
                                       "settings, or pick another camera in Settings."))
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        detector = MotionDetector()

        # Load the AI model (first run downloads ~6 MB of weights, so it needs internet once).
        ai = None
        if Settings.ai:
            self.event_q.put(("info", "LOADING AI MODEL..."))
            try:
                ai = ObjectDetector()
            except Exception as exc:
                self.event_q.put(("ai_unavailable", f"{type(exc).__name__}: {exc}"))
        cached = []
        hits = []

        writer = None
        in_event = False
        last_motion = 0.0
        last_snap = 0.0
        frames_seen = 0
        fps, t_prev = 0.0, time.time()
        self.event_q.put(("started",))

        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.event_q.put(("error", "Lost connection to the camera."))
                    break

                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (PROC_WIDTH, int(h * PROC_WIDTH / w)))
                frames_seen += 1

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
                t_prev = now

                boxes, light_change = detector.process(frame, Settings.sensitivity, Settings.min_area)
                learning = frames_seen <= WARMUP_FRAMES
                if learning:
                    boxes = []

                use_ai = ai is not None and Settings.ai
                dets = []
                if use_ai:
                    if boxes or in_event:
                        if frames_seen % 2 == 0:   # run the AI on every 2nd frame to keep FPS up
                            found = ai.detect(frame, Settings.conf)
                            # keep only objects that are actually moving (ignores parked cars, posters)
                            cached = [d for d in found
                                      if moving_share(detector.last_mask, d[:4]) >= MIN_MOTION_OVERLAP]
                            hits = (hits + [bool(cached)])[-CONFIRM_WINDOW:]
                    else:
                        cached, hits = [], []
                    confirmed = bool(cached) and sum(hits) >= 2
                    dets = cached if confirmed else []
                    moving = confirmed
                else:
                    moving = bool(boxes)

                if moving:
                    last_motion = now
                    if not in_event:
                        in_event = True
                        what = ", ".join(sorted({d[4] for d in dets})) or "motion"
                        self.event_q.put(("motion_start", what))

                if in_event and now - last_motion > POST_ROLL_SECONDS:
                    in_event = False
                    cached, hits = [], []
                    self.event_q.put(("motion_end",))

                view = frame.copy()
                self._overlay(view, boxes, writer is not None, fps, learning, dets, use_ai)

                # Snapshots (automatic during motion, or on demand)
                manual = self.snapshot_requested.is_set()
                if (moving and now - last_snap >= Settings.cooldown) or manual:
                    self.snapshot_requested.clear()
                    name = f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    if manual:
                        name = name.replace("motion_", "manual_")
                    path = SNAP_DIR / name
                    cv2.imwrite(str(path), view)
                    first = last_snap < last_motion - POST_ROLL_SECONDS or last_snap == 0.0
                    last_snap = now
                    self.event_q.put(("snapshot", f"captured_images/{name}", first and not manual))

                # Video clip recording while an event is open
                if Settings.record and in_event:
                    if writer is None:
                        clip = f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(REC_DIR / clip), fourcc, RECORD_FPS,
                                                 (view.shape[1], view.shape[0]))
                        self.event_q.put(("recording", f"recordings/{clip}"))
                    writer.write(view)
                elif writer is not None:
                    writer.release()
                    writer = None

                self._push((view, in_event))
        finally:
            if writer is not None:
                writer.release()
            cap.release()
            self.event_q.put(("stopped",))


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Motion Detection & Security System")
        root.configure(bg=BG)
        root.geometry("1180x720")
        root.minsize(1000, 640)

        self.frame_q: queue.Queue = queue.Queue(maxsize=1)
        self.event_q: queue.Queue = queue.Queue()
        self.worker = None
        self.motion_count = 0
        self.started_at = None
        self.photo = None
        self.preview_photo = None
        self.running = False
        self.cam_choices = {}                  # "Camera 1 (1280x720)" -> 1
        self.scan_q: queue.Queue = queue.Queue()
        self.scanning = False

        self._style()
        self._build()
        self._load_history()
        self._refresh_gallery()
        self._show_placeholder("Camera is off - press Start Monitoring")
        self._set_status("off")

        root.bind("<space>", lambda e: self.toggle())
        root.bind("<s>", lambda e: self.manual_snapshot())
        root.bind("<q>", lambda e: self.on_close())
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll()
        self.root.after(400, self.scan_cameras)   # look for cameras right after the window opens

    # -- UI construction -----------------------------------------------------
    def _style(self) -> None:
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=PANEL, foreground=TEXT, fieldbackground=CARD)
        s.configure("TFrame", background=PANEL)
        s.configure("Card.TFrame", background=CARD)
        s.configure("TLabel", background=PANEL, foreground=TEXT)
        s.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        s.configure("Card.TLabel", background=CARD, foreground=TEXT)
        s.configure("Big.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 20, "bold"))
        s.configure("TButton", background=CARD, foreground=TEXT, padding=8, borderwidth=0)
        s.map("TButton", background=[("active", GREY)])
        s.configure("Start.TButton", background=GREEN, foreground="white", font=("Segoe UI", 10, "bold"))
        s.map("Start.TButton", background=[("active", "#17884a")])
        s.configure("Stop.TButton", background=RED, foreground="white", font=("Segoe UI", 10, "bold"))
        s.map("Stop.TButton", background=[("active", "#c23b40")])
        s.configure("TCheckbutton", background=PANEL, foreground=TEXT)
        s.configure("TNotebook", background=PANEL, borderwidth=0)
        s.configure("TNotebook.Tab", background=CARD, foreground=MUTED, padding=(14, 8))
        s.map("TNotebook.Tab", background=[("selected", ACCENT)], foreground=[("selected", "white")])
        s.configure("Treeview", background=CARD, fieldbackground=CARD, foreground=TEXT, rowheight=24, borderwidth=0)
        s.configure("Treeview.Heading", background=GREY, foreground=TEXT)
        s.configure("Horizontal.TScale", background=PANEL)

    def _build(self) -> None:
        # Header
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=16, pady=(12, 6))
        tk.Label(header, text="SECURITY MONITOR", bg=BG, fg=TEXT,
                 font=("Segoe UI", 16, "bold")).pack(side="left")
        tk.Label(header, text="  Motion Detection & Surveillance System", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(side="left", pady=(6, 0))
        self.clock = tk.Label(header, text="", bg=BG, fg=MUTED, font=("Segoe UI", 10))
        self.clock.pack(side="right")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # Left: status + video
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.status = tk.Label(left, text="", font=("Segoe UI", 14, "bold"), fg="white", pady=10)
        self.status.pack(fill="x")
        self.video = tk.Label(left, bg="black")
        self.video.pack(fill="both", expand=True, pady=(8, 0))

        # Right: stats + notebook
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")

        stats = ttk.Frame(right)
        stats.pack(fill="x", padx=10, pady=10)
        stats.columnconfigure((0, 1), weight=1)
        self.count_var = tk.StringVar(value="0")
        self.uptime_var = tk.StringVar(value="00:00:00")
        for col, (title, var) in enumerate((("Motion events", self.count_var), ("Uptime", self.uptime_var))):
            card = ttk.Frame(stats, style="Card.TFrame")
            card.grid(row=0, column=col, sticky="ew", padx=4)
            ttk.Label(card, text=title, style="Card.TLabel", foreground=MUTED).pack(anchor="w", padx=12, pady=(8, 0))
            ttk.Label(card, textvariable=var, style="Big.TLabel").pack(anchor="w", padx=12, pady=(0, 8))

        btns = ttk.Frame(right)
        btns.pack(fill="x", padx=10)
        self.toggle_btn = ttk.Button(btns, text="Start Monitoring", style="Start.TButton", command=self.toggle)
        self.toggle_btn.pack(side="left", fill="x", expand=True, padx=(4, 4))
        ttk.Button(btns, text="Snapshot", command=self.manual_snapshot).pack(side="left", padx=4)

        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        self._tab_settings(notebook)
        self._tab_history(notebook)
        self._tab_gallery(notebook)

    def _tab_settings(self, nb) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="Settings")
        pad = {"padx": 14, "pady": (10, 0), "anchor": "w"}

        ttk.Label(tab, text="Sensitivity  (higher = detects smaller changes)").pack(**pad)
        self.sens = tk.DoubleVar(value=Settings.sensitivity)
        ttk.Scale(tab, from_=1, to=10, variable=self.sens, command=self._apply).pack(fill="x", padx=14)

        ttk.Label(tab, text="Minimum object size  (higher = ignores small movement)").pack(**pad)
        self.area = tk.DoubleVar(value=Settings.min_area)
        ttk.Scale(tab, from_=500, to=20000, variable=self.area, command=self._apply).pack(fill="x", padx=14)

        self.ai_var = tk.BooleanVar(value=Settings.ai)
        ttk.Checkbutton(tab, text="AI verification: only alarm for people, vehicles, animals",
                        variable=self.ai_var, command=self._apply).pack(**pad)
        ttk.Label(tab, text="AI confidence  (higher = fewer false alarms)").pack(**pad)
        self.conf_var = tk.DoubleVar(value=Settings.conf)
        ttk.Scale(tab, from_=0.25, to=0.90, variable=self.conf_var, command=self._apply).pack(fill="x", padx=14)

        ttk.Label(tab, text="Snapshot cooldown (seconds)").pack(**pad)
        self.cool = tk.IntVar(value=Settings.cooldown)
        ttk.Spinbox(tab, from_=1, to=60, textvariable=self.cool, width=6, command=self._apply).pack(**pad)

        ttk.Label(tab, text="Camera  (internal / external)").pack(**pad)
        cam_row = ttk.Frame(tab)
        cam_row.pack(anchor="w", padx=14, pady=(4, 0))
        self.cam_label = tk.StringVar(value="Camera 0")
        self.cam_combo = ttk.Combobox(cam_row, textvariable=self.cam_label, state="readonly",
                                      values=["Camera 0"], width=24)
        self.cam_combo.pack(side="left")
        ttk.Button(cam_row, text="Scan", command=self.scan_cameras).pack(side="left", padx=6)

        self.rec_var = tk.BooleanVar(value=True)
        self.snd_var = tk.BooleanVar(value=True)
        self.mail_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Record video clip on motion", variable=self.rec_var,
                        command=self._apply).pack(**pad)
        ttk.Checkbutton(tab, text="Sound alert", variable=self.snd_var, command=self._apply).pack(**pad)
        ttk.Checkbutton(tab, text="E-mail alert with snapshot", variable=self.mail_var,
                        command=self._mail_toggle).pack(**pad)

        row = ttk.Frame(tab)
        row.pack(fill="x", padx=14, pady=14)
        ttk.Button(row, text="Open snapshots", command=lambda: open_path(SNAP_DIR)).pack(side="left", padx=(0, 6))
        ttk.Button(row, text="Open recordings", command=lambda: open_path(REC_DIR)).pack(side="left")

    def _tab_history(self, nb) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="Event History")
        cols = ("time", "event", "detail")
        self.tree = ttk.Treeview(tab, columns=cols, show="headings")
        for c, wdt in zip(cols, (140, 80, 200)):
            self.tree.heading(c, text=c.title())
            self.tree.column(c, width=wdt, anchor="w")
        scroll = ttk.Scrollbar(tab, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scroll.pack(side="right", fill="y", pady=8, padx=(0, 4))
        self.tree.bind("<Double-1>", self._open_history_item)

    def _tab_gallery(self, nb) -> None:
        tab = ttk.Frame(nb)
        nb.add(tab, text="Gallery")
        top = ttk.Frame(tab)
        top.pack(fill="x", padx=8, pady=8)
        self.listbox = tk.Listbox(top, height=6, bg=CARD, fg=TEXT, selectbackground=ACCENT,
                                  highlightthickness=0, borderwidth=0, activestyle="none")
        self.listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(top, orient="vertical", command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", self._preview)
        self.listbox.bind("<Double-1>", lambda e: self._open_selected())
        self.preview = tk.Label(tab, bg="black", text="Select a snapshot", fg=MUTED)
        self.preview.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # -- display helpers -----------------------------------------------------
    def _show_placeholder(self, text: str) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(img, text, ((640 - size[0]) // 2, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (150, 160, 175), 2, cv2.LINE_AA)
        self._show_frame(img)

    def _show_frame(self, bgr) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        max_w = max(self.video.winfo_width(), 320)
        max_h = max(self.video.winfo_height(), 240)
        img.thumbnail((max_w, max_h))
        self.photo = ImageTk.PhotoImage(img)
        self.video.configure(image=self.photo)

    def _set_status(self, mode: str) -> None:
        text, colour = {
            "off": ("MONITORING OFF", GREY),
            "secure": ("AREA SECURE", GREEN),
            "motion": ("MOTION DETECTED", RED),
        }[mode]
        self.status.configure(text=text, bg=colour)

    # -- settings ------------------------------------------------------------
    def _apply(self, *_):
        try:
            Settings.sensitivity = float(self.sens.get())
            Settings.min_area = int(float(self.area.get()))
            Settings.cooldown = max(1, int(self.cool.get()))
        except (tk.TclError, ValueError):
            pass
        Settings.record = self.rec_var.get()
        Settings.sound = self.snd_var.get()
        Settings.ai = self.ai_var.get()
        try:
            Settings.conf = float(self.conf_var.get())
        except (tk.TclError, ValueError):
            pass

    def _mail_toggle(self) -> None:
        if self.mail_var.get() and not (os.environ.get("MOTION_SMTP_USER") and os.environ.get("MOTION_SMTP_PASS")):
            messagebox.showinfo(
                "E-mail not configured",
                "Set the environment variables MOTION_SMTP_USER and MOTION_SMTP_PASS "
                "(a Gmail App Password) before starting the app. See README.md.")
            self.mail_var.set(False)
        Settings.email = self.mail_var.get()

    # -- actions -------------------------------------------------------------
    def toggle(self) -> None:
        self.stop() if self.running else self.start()

    def start(self) -> None:
        if self.running:
            return
        self._apply()
        cam_index = self.cam_choices.get(self.cam_label.get(), 0)
        self.worker = CameraWorker(cam_index, self.frame_q, self.event_q)
        self.worker.start()
        self.running = True
        self.started_at = time.time()
        self.toggle_btn.configure(text="Stop Monitoring", style="Stop.TButton")
        self._set_status("secure")
        log_event("START", "Monitoring started")
        self._add_history_row(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "START", "Monitoring started")

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        if self.worker:
            self.worker.stop_event.set()
            self.worker.join(timeout=3)
            self.worker = None
        self.toggle_btn.configure(text="Start Monitoring", style="Start.TButton")
        self._set_status("off")
        self._show_placeholder("Camera is off - press Start Monitoring")
        stamp = log_event("STOP", "Monitoring stopped")
        self._add_history_row(stamp, "STOP", "Monitoring stopped")

    # -- camera discovery ----------------------------------------------------
    def scan_cameras(self) -> None:
        if self.scanning or self.running:
            if self.running:
                messagebox.showinfo("Scan cameras", "Stop monitoring first, then scan again.")
            return
        self.scanning = True
        self.status.configure(text="SCANNING CAMERAS...", bg=ACCENT)
        threading.Thread(target=lambda: self.scan_q.put(find_cameras()), daemon=True).start()

    def _on_scan_done(self, cameras) -> None:
        self.scanning = False
        if not self.running:
            self._set_status("off")
        if not cameras:
            messagebox.showwarning(
                "No camera found",
                "No working camera was detected.\n\n- Close apps that use the camera (Zoom, Teams, Camera)\n"
                "- Windows Settings > Privacy > Camera: allow desktop apps\n"
                "- Plug in the external camera, then press Scan")
            return
        self.cam_choices = {f"Camera {i}  ({res})": i for i, _thumb, res in cameras}
        labels = list(self.cam_choices)
        self.cam_combo.configure(values=labels)
        self.cam_label.set(labels[0])
        if len(cameras) > 1:
            self._camera_chooser(cameras)

    def _camera_chooser(self, cameras) -> None:
        """Startup dialog: pick which camera to use (shows a preview of each)."""
        win = tk.Toplevel(self.root)
        win.title("Choose camera")
        win.configure(bg=PANEL)
        win.transient(self.root)
        win.resizable(False, False)
        tk.Label(win, text="Which camera do you want to use?", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(padx=20, pady=(16, 2))
        tk.Label(win, text="Pick the preview that shows your view. The built-in webcam is usually "
                           "Camera 0; a USB camera is usually Camera 1.",
                 bg=PANEL, fg=MUTED, wraplength=460, justify="left").pack(padx=20, pady=(0, 10))

        choice = tk.StringVar(value=list(self.cam_choices)[0])
        row = tk.Frame(win, bg=PANEL)
        row.pack(padx=16, pady=4)
        win._thumbs = []   # keep references so Tk does not drop the images
        for (index, thumb, res), label in zip(cameras, self.cam_choices):
            card = tk.Frame(row, bg=CARD, padx=8, pady=8)
            card.pack(side="left", padx=6)
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)))
            win._thumbs.append(photo)
            tk.Label(card, image=photo, bg=CARD).pack()
            tk.Radiobutton(card, text=f"Camera {index}  ({res})", variable=choice, value=label,
                           bg=CARD, fg=TEXT, selectcolor=GREY, activebackground=CARD,
                           activeforeground=TEXT).pack(pady=(6, 0))

        def confirm() -> None:
            self.cam_label.set(choice.get())
            win.destroy()

        ttk.Button(win, text="Use selected camera", style="Start.TButton", command=confirm).pack(pady=14)
        win.protocol("WM_DELETE_WINDOW", confirm)
        win.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - win.winfo_height()) // 3
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        win.grab_set()

    def manual_snapshot(self) -> None:
        if self.running and self.worker:
            self.worker.snapshot_requested.set()

    def on_close(self) -> None:
        self.stop()
        self.root.destroy()

    # -- history & gallery ---------------------------------------------------
    def _load_history(self) -> None:
        for row in read_log():
            if len(row) >= 3:
                self.tree.insert("", 0, values=row[:3])

    def _add_history_row(self, stamp, kind, detail) -> None:
        self.tree.insert("", 0, values=(stamp, kind, detail))

    def _open_history_item(self, _event) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        detail = self.tree.item(sel[0], "values")[2]
        target = BASE_DIR / detail
        if detail.startswith(("captured_images/", "recordings/")) and target.exists():
            open_path(target)

    def _refresh_gallery(self) -> None:
        self.listbox.delete(0, "end")
        files = sorted(SNAP_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)[:300]
        self.gallery_files = files
        for f in files:
            self.listbox.insert("end", f.name)

    def _preview(self, _event=None) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        try:
            img = Image.open(self.gallery_files[sel[0]])
            img.thumbnail((420, 300))
            self.preview_photo = ImageTk.PhotoImage(img)
            self.preview.configure(image=self.preview_photo, text="")
        except Exception:
            self.preview.configure(image="", text="Cannot preview this file")

    def _open_selected(self) -> None:
        sel = self.listbox.curselection()
        if sel:
            open_path(self.gallery_files[sel[0]])

    # -- main polling loop ---------------------------------------------------
    def _poll(self) -> None:
        self.clock.configure(text=datetime.now().strftime("%a %d %b %Y   %H:%M:%S"))
        if self.running and self.started_at:
            secs = int(time.time() - self.started_at)
            self.uptime_var.set(f"{secs // 3600:02d}:{secs % 3600 // 60:02d}:{secs % 60:02d}")

        # newest video frame
        try:
            latest = None
            while True:
                latest = self.frame_q.get_nowait()
        except queue.Empty:
            pass
        if latest is not None and self.running:
            frame, in_event = latest
            self._show_frame(frame)
            self._set_status("motion" if in_event else "secure")

        # camera scan results
        try:
            self._on_scan_done(self.scan_q.get_nowait())
        except queue.Empty:
            pass

        # worker events
        try:
            while True:
                self._handle_event(self.event_q.get_nowait())
        except queue.Empty:
            pass

        self.root.after(30, self._poll)

    def _handle_event(self, event) -> None:
        kind = event[0]
        if kind == "info":
            self.status.configure(text=event[1], bg=ACCENT)
        elif kind == "ai_unavailable":
            self.ai_var.set(False)
            Settings.ai = False
            messagebox.showwarning(
                "AI model not available",
                "Running in basic motion mode (it cannot tell people from fans or doors).\n\n"
                "This app is running with THIS Python:\n"
                f"{sys.executable}\n\n"
                "Install the AI library into that exact Python by running this in PowerShell, "
                "then restart the app:\n\n"
                f'& "{sys.executable}" -m pip install ultralytics\n\n'
                "(Be online the first time: it also downloads the model.)\n\n"
                f"Details: {event[1]}")
        elif kind == "motion_start":
            self.motion_count += 1
            self.count_var.set(str(self.motion_count))
            what = event[1] if len(event) > 1 else "motion"
            detail = f"Event #{self.motion_count}: {what}"
            stamp = log_event("MOTION", detail)
            self._add_history_row(stamp, "MOTION", detail)
            if Settings.sound:
                threading.Thread(target=beep, daemon=True).start()
        elif kind == "motion_end":
            stamp = log_event("CLEAR", "Area clear")
            self._add_history_row(stamp, "CLEAR", "Area clear")
        elif kind == "snapshot":
            rel, first = event[1], event[2]
            stamp = log_event("SNAPSHOT", rel)
            self._add_history_row(stamp, "SNAPSHOT", rel)
            self._refresh_gallery()
            if first and Settings.email:
                threading.Thread(target=send_email_alert, args=(BASE_DIR / rel, stamp), daemon=True).start()
        elif kind == "recording":
            stamp = log_event("VIDEO", event[1])
            self._add_history_row(stamp, "VIDEO", event[1])
        elif kind == "error":
            messagebox.showerror("Camera error", event[1])
            self.running = True  # let stop() run its cleanup path
            self.stop()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
