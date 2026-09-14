# Caught

A desk camera that learns your bad habits and, at the end of the week, shows you the chart.

It runs all day on your Mac's built-in camera. It never uploads anything and never stores video.
It notices when you pick up your phone, slouch, stare at your lap, chew your nails, rub your eyes,
yawn, doze, fidget, eat, drink, look away, or leave. Each one becomes an episode with a start, an
end, and a confidence. A local dashboard shows today's timeline, the week's heatmap, streaks, and
one uncomfortable bar chart. It does not nag.

Under the product is the actual project: a rule-based detector stack you can read in one sitting,
a personal temporal model that trains on *your* footage, and an active-learning loop that asks you
about the episodes it is least sure of and retrains overnight.

## Requirements

* macOS on Apple silicon (tested on an M5), Python 3.12 via [uv](https://docs.astral.sh/uv/)
* The built-in camera, or an iPhone via Continuity Camera

## Setup

```bash
cd caught
uv sync                      # installs MediaPipe, YOLO, PyTorch (Metal), FastAPI
uv run caught cameras        # macOS will ask for camera permission the first time
uv run caught calibrate      # sit upright for 10 seconds: your posture baseline
uv run caught run --open     # watch + dashboard at http://127.0.0.1:8787
```

Model weights for MediaPipe and YOLO11n live in `models/` and are downloaded once.

## Daily use

Open the dashboard, press **25 min** (or 50, or 90), and study. When you stop, it shows a focus
score for that stretch: phone pickups, minutes on the phone, longest clean run, posture, and the one
thing to fix first. The Home tab keeps today's score and your recent sessions in view. Everything
else (timeline, week heatmap, review queue) is one tab over when you want it.

## Commands

| Command | What it does |
|---|---|
| `caught run [--open] [--window] [--camera N]` | Watch the camera and serve the dashboard. `--window` adds a live preview window. |
| `caught serve` | Dashboard only, no camera. Browse history from any day. |
| `caught calibrate` | Record your upright posture baseline (needed for slouch detection). |
| `caught train` | Train the personal model on stored features and your labels. |
| `caught report [--roast] [--open]` | Build the weekly report. `--roast` asks Claude for the paragraph, sending weekly totals only. |
| `caught simulate --days 7` | Generate synthetic history so you can explore the dashboard before wearing it in. |
| `caught menubar` | Run as a menu bar app with pause, calibrate, and train. |
| `caught status`, `caught wipe --yes` | Print a summary, or delete everything. |
| `caught run --video clip.mp4` | Replay a recorded video through the pipeline for development. |
| `caught record phone --seconds 20` | Record a labeled clip of yourself doing a habit into `bench/` for backtesting. |
| `caught bench [bench/] [--trace]` | Backtest the detectors on every labeled clip and photo; `--trace` writes per-frame CSVs. |

Data lives in `~/.caught` (override with `--data DIR` or `CAUGHT_DATA`).

## How it works

```
camera ──► MediaPipe face/hand/pose landmarks ──┐
       └─► YOLO11n: phone, cup, bottle, food ───┤
                                                 ▼
                       FrameFeatures (52 numbers per frame: head pose, hand-to-mouth
                       distance in face-heights, eye closure, phone-in-hand, motion…)
                                                 │
            ┌────────────────────────────────────┼─────────────────────────┐
            ▼                                    ▼                         ▼
   12 rule detectors               personal temporal model          feature history
   (instantaneous 0..1)            (Conv1d→GRU over 8s windows)     (SQLite, 2 Hz)
            └──────────────── blend ─────────────┘                         │
                                 ▼                                         │
                 per-habit state machine: EMA smoothing, hysteresis,       │
                 minimum dwell, minimum duration, gap merging              │
                                 ▼                                         │
                        episodes ──► SQLite ──► dashboard / report         │
                                 │                                         │
                    uncertain ones get 6 tiny thumbnails                   │
                                 ▼                                         │
                     you label yes/no in the Review tab ───► nightly retrain
```

**Detectors** (`src/caught/detectors/rules.py`) are small pure functions over the feature vector.
Every one has explicit thresholds you can tune from `config.json` under `detector_params`.

**Temporal logic** (`src/caught/episodes.py`) is where false positives go to die: a phone has to be
in your hand for two seconds before an episode starts, four seconds out of view before it ends,
and two sessions fifteen seconds apart are one session.

**The personal model** (`src/caught/model/`) trains on windows of the stored feature history.
Rule episodes are weak positives, your confirmations are strong positives, your rejections are
strong negatives, and time far from any episode is a weak negative. Its scores blend with the
rules at a weight you control in Settings, and grow more trusted as you label more.

**Active learning** (`src/caught/model/active.py`) queues an episode for review when its confidence
is in the uncertain band, when the rules and the model disagree, or as a small random sample for
quality control. A nightly job retrains when there are new labels.

**Power** (`src/caught/power.py`) drops to one frame per second when nobody is there and three
when you are still, and runs the hand landmarker only when the body model says a wrist is up.

## Backtesting

`bench/all/` holds about twenty freely licensed stock clips and photos of real people at desks
(Mixkit and Wikimedia Commons; sources in `bench/commons/sources.json`) with a labeled
`manifest.json`. `caught bench bench/all` runs the full perception and detector stack on each and
judges what the product would report: committed episodes. Every strict item passes; the optional
ones document known limits (side views, faces too small, a single bite in a long clip).

Findings that came out of the bench and are now built in:

* Head pitch from the face transform is positive when looking down (verified on footage).
* When the person sits far away, the face landmarker is re-run on an upscaled crop around the
  body model's head landmarks.
* Hands over the face hide the fingers from the hand model; the wrist from the body model
  is used as a fallback for hand-at-face, eye rubbing, and phone calls (wrist at the ear).
* A cup at the mouth is exactly when the object detector stops seeing it, so an object seen in the
  hand stays "held" until the hand drops. A sip with no cup visible is recognised from pursed lips,
  head tipped back, and a hand at the mouth.
* The apple on the desk is not eating: food only counts near the mouth or hand. Chewing is small
  jaw motion plus repeated hand-to-mouth visits; a yawn is excluded.
* Eye closure from the blendshapes rises when the gaze drops (reading, eating), so drowsiness
  requires the gaze to be level, no yawn, and no hand near the eyes.
* A square, low-confidence "phone" is usually a mug.
* Episodes arbitrate: when a drink, eat, or phone episode starts, a hand-at-face episode that
  began seconds earlier is retracted. It was the same gesture.
* Dwell timers use a leaky credit so one dropped frame does not restart the clock.

Record your own clips with `caught record <habit>` and they join the bench.

## Privacy

* Frames live in memory and are discarded after each step.
* Only uncertain episodes keep a strip of up to six 160-pixel thumbnails, locally, for the review
  queue. Turn that off in Settings and nothing image-like is ever written.
* The server binds to 127.0.0.1.
* The optional weekly roast sends weekly aggregate numbers to the Claude API and nothing else, and
  only when you enable it.

## Tuning

* Head pitch is read from MediaPipe's face transform. Positive pitch means chin toward chest. If
  "looking down" fires when you look up, check the Live tab's pitch readout and flip the sign in
  `features.py`.
* Slouch is a robust z-score against your calibration across five geometric cues. Recalibrate if
  you move the camera or the chair.
* Every threshold is in `detector_params`, e.g. `{"phone": {"conf_lo": 0.4, "min_on": 3.0}}`.

## Tests

```bash
uv run pytest -q
```

The suite covers the state machine, every detector, feature geometry, the store, training and
inference on simulated data, the report, and the HTTP API.
