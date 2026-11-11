# Motion Detection & Security System

A security camera app for Windows, made with Python. It uses your webcam to watch a room. When a person, vehicle or
animal moves in front of the camera, it shows an alert, saves a photo, records a short video and writes the event
in a log. It ignores things like fans, curtains and doors.

**Built with:** Python, OpenCV, YOLOv8 (Ultralytics), NumPy, Tkinter, Pillow

## What it can do
- Live camera view with boxes around what is moving
- AI check (YOLOv8): only people, vehicles and animals trigger the alarm. Fans, doors and curtains are ignored
- Red box with a label on confirmed objects, like `PERSON 87%`
- Status banner: MONITORING OFF / AREA SECURE / MOTION DETECTED
- Automatic photos with date and time, saved in `captured_images/`
- Automatic short video clips while something is moving, saved in `recordings/`
- Event history (saved in `event_log.csv`) and a photo gallery with preview
- Camera chooser: pick the built-in or the external webcam, with a preview of each
- Settings you can change while it runs: sensitivity, minimum object size, AI confidence, photo cooldown
- Light-change guard: switching a light on or off does not cause a false alarm
- Sound alert, and an optional email alert with the photo
- Works in basic motion mode if the AI library is not installed

## What you need
- Windows 10 or 11
- **Python 3.10, 3.11 or 3.12** (newer versions may not work with PyTorch)
- A webcam (built-in or USB)
- Internet the first time (to install libraries and download the AI model)
- VS Code is optional, you can also use PowerShell

## How to run it
1. Install Python from python.org. Tick **"Add Python to PATH"** during install.
2. Put `main.py`, `requirements.txt` and `README.md` in one folder, for example `Motion_Detection_System`.
3. Open PowerShell in that folder (in File Explorer, click the address bar, type `powershell`, press Enter).
4. Install the libraries (this is big, wait a few minutes):
   ```
   python -m pip install -r requirements.txt
   ```
5. Start the app:
   ```
   python main.py
   ```
6. If you have more than one camera, a window asks which one to use. Pick the preview that shows your view.
7. Click **Start Monitoring**. The first time, it downloads the AI model (about 6 MB). Wait until
   "LOADING AI MODEL..." goes away. Then walk in front of the camera.

If `python` is not recognized, use `py` instead of `python` in the commands above.

## Controls
| Key | What it does |
| --- | --- |
| Space | Start or stop monitoring |
| S | Save a photo now |
| Q | Quit |

## Settings tab
| Setting | What it does |
| --- | --- |
| Sensitivity | Higher = smaller changes count as movement |
| Minimum object size | Higher = small movement is ignored |
| AI verification | On = only people, vehicles and animals alarm. Off = any movement alarms |
| AI confidence | Higher = fewer false alarms, but it may miss some real ones |
| Snapshot cooldown | Seconds to wait between saved photos |
| Camera / Scan | Choose the camera. Press Scan after plugging in a USB camera |
| Record video clip | Turns video recording on or off |
| Sound alert | Turns the beep on or off |
| E-mail alert | Sends the photo to your email (needs setup, see below) |

## How it works
1. The camera gives frames (pictures) again and again.
2. Each frame is made smaller, turned grey and blurred a little.
3. The program keeps a background picture of the room and updates it slowly. Each new frame is compared with it,
   and the differences show what moved.
4. Moving areas that are big enough get a box.
5. YOLOv8 then checks what is in the frame. An alarm happens only if a person, vehicle or animal is inside the
   moving area, and the AI agrees in 2 of the last 3 checks.
6. On an alarm it saves a photo, records a video, plays a sound and writes the event to the log.

## Optional: email alerts (Gmail)
1. In your Google account, turn on 2-Step Verification and make an **App Password**.
2. In PowerShell, run these (use your own details):
   ```
   setx MOTION_SMTP_USER "yourname@gmail.com"
   setx MOTION_SMTP_PASS "your-16-letter-app-password"
   setx MOTION_ALERT_TO "email-to-receive-alerts@gmail.com"
   ```
3. Close PowerShell, open it again, run `python main.py`, and tick **E-mail alert** in Settings.

## Better accuracy (optional)
The app uses the small AI model `yolov8n.pt`, which is fast. On a faster computer you can use a bigger one.
Before starting the app, in PowerShell run:
```
$env:MOTION_MODEL="yolov8s.pt"
python main.py
```
It is more accurate but slower.

## Problems and fixes
| Problem | Fix |
| --- | --- |
| "AI model not available" pop-up | The AI library is in a different Python. Run the command shown in the pop-up, then restart the app |
| `pip` is not recognized | Use `python -m pip install ...` or `py -m pip install ...` |
| Install fails with a PyTorch error | Your Python is too new. Install Python 3.12 and use `py -3.12 -m pip install -r requirements.txt`, then `py -3.12 main.py` |
| Camera does not open | Close Zoom, Teams and the Camera app. Check Windows Settings > Privacy > Camera and allow desktop apps. Then press Scan |
| Built-in camera missing, USB works | Press Scan in Settings. Check the same privacy setting above |
| Too many false alarms | Raise AI confidence or Minimum object size, or lower Sensitivity |
| Misses real movement | Lower AI confidence or Minimum object size, or raise Sensitivity |
| App is slow | Use the default `yolov8n.pt` model and close other heavy programs |

## Files
- `main.py` - the whole program
- `requirements.txt` - libraries to install
- `README.md` - this file
- `captured_images/`, `recordings/`, `event_log.csv` - made automatically when you run it

## Limits
- It can miss people in the dark, far away, or partly hidden.
- A person standing completely still does not trigger it, because it needs movement.
- The camera must stay still. If the camera moves, everything looks like movement.
- It is a learning and small-setup project, not a replacement for a professional security system.

## Future ideas
Face recognition for known people, many cameras at once, phone notifications, cloud storage for photos,
a password for the app, and night-vision support.
