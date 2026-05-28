# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

A learning repository for an end-to-end "Smart Observer" video pipeline: YOLO object detection on a video stream → PostgreSQL event log → FastAPI dashboard. Work is organised by `Chapter_*` git branches that each add a stage of the curriculum (numpy basics → image processing → video → docker/Postgres → YOLO training → multi-thread pipeline → FastAPI UI). The `master` branch holds the foundational pre-YOLO material; the current `Chapter_9` branch contains the full pipeline + dashboard and is what this file documents.

## Environment

- Python 3.11 via micromamba env `ai_project` (`/home/yustasdev/micromamba/envs/ai_project`). The local `.venv/` is a thin wrapper that points at this env (`base-prefix` in `pyvenv.cfg`).
- GPU inference is assumed: `core/detector.py` hardcodes `device="cuda:0"`. Runs on WSL2/Linux.
- PostgreSQL runs in Docker on `localhost:5432`. `.env` holds `POSTGRES_USER=myuser`, `POSTGRES_PASSWORD=mypassword`, `POSTGRES_DB=mydatabase`; the same creds are hardcoded in `db/database.py`.
- Legacy training dataset lives **outside** the repo at `/home/yustasdev/datasets/cow_dataset` with the standard YOLO layout (`images/{train,val,test}` + `labels/{train,val,test}`). It has its own `data.yaml`; `07_train_local.py` points there directly.

## Architecture

Three concurrent threads connected by two bounded queues, plus a separate FastAPI dashboard process:

```
source ──▶ VideoStreamer ──▶ frame_queue ──▶ AIDetector ──▶ event_queue ──▶ DBLogger ──┐
        (core/streamer.py)    (maxsize=30)  (core/detector.py)  (maxsize=100)          │
                                                                                       ▼
                                                            data/saved_events/*.jpg + detections row
                                                                                       │
                                                            FastAPI dashboard reads ◀──┘
                                                            (api/main_api.py)
```

- `main.py` — orchestrator. Parses `--source`, sets up file logging (`logs/info.log` / `warning.log` / `error.log`), starts threads in consumer→producer order (DBLogger → AIDetector → VideoStreamer), runs a 3-sec health check, prints queue depths every 5 sec, handles Ctrl+C with graceful shutdown.
- `core/streamer.py` — reads frames via `cv2.VideoCapture`; on queue-full, drops the **oldest** frame (real-time priority).
- `core/detector.py` — runs YOLO inference on each frame; filters by `target_classes`; produces **one event per detected target object**, but all events from the same frame share one annotated image (to avoid the cumulative-boxes bug).
- `core/logger.py` — consumes events: writes `event_<class>_<utc_ts>.jpg` to `data/saved_events/` and inserts a row into the `detections` table. Drains the queue on stop so no events are lost during shutdown.
- `db/database.py` + `db/models.py` — `Detection` ORM model (`id, timestamp, object_class, confidence, bounding_box, image_path`). **The `detections` table is created by `Base.metadata.create_all()` inside `DBLogger.run()`, not by `init/01-init.sql`** — the SQL init file only seeds an unrelated `users` table left over from earlier chapters.
- `api/main_api.py` + `api/templates/index.html` — FastAPI dashboard. `GET /` renders the latest 20 events as Bootstrap cards; `GET /api/events?limit=N` returns JSON; `/images/*` serves the saved JPEGs. Paths are anchored to the file (`Path(__file__).resolve().parent.parent`), so it runs regardless of CWD.

## Commands

Bring up Postgres:
```bash
docker compose up -d
```

Run the detection pipeline (writes to DB + disk):
```bash
python main.py --source 0                  # webcam (default)
python main.py --source test_video1.mp4    # local video file
python main.py --source rtsp://...         # network stream
```

Run the FastAPI dashboard (separate process):
```bash
uvicorn api.main_api:app --reload
# → http://localhost:8000
```

Legacy training/inference (kept for reference; dataset lives outside the repo):
```bash
python 07_train_local.py     # train; output under ai_runs/cow_learning*/
python 07_parse_results.py   # inference on the hardcoded image in the script
```

## Model and classes

- `best.pt` is the active checkpoint: a 2-class **person + car** detector from Roboflow Universe. Class IDs are referenced in `main.py` via `TARGET_CLASSES = [0, 1]`. **Always sanity-check the actual mapping** from the first log line `Классы модели: ...` printed by `AIDetector` on startup — if the model loads with `0=car, 1=person`, that order propagates into the `object_class` column.
- `best.pt.kaggle-trained` is an older cow/person checkpoint kept as a backup.
- Pretrained Ultralytics weights (`yolo11s.pt`, `yolo11m.pt`, `yolo26n.pt`) at the repo root are referenced by the legacy training scripts and are auto-downloaded by Ultralytics elsewhere — don't delete them lightly.

## Repo-specific gotchas

- **Two `data.yaml` files exist.** The one at the repo root is a teaching example with the placeholder `path: /home/user/datasets/cow_dataset` and is *not* what training uses. `07_train_local.py` passes the absolute path to the dataset's own `data.yaml`. If you change class definitions, edit the one in the dataset directory.
- **`image_path` in the DB stores an absolute path** (because `main.py` passes `str(SAVE_DIR)` where `SAVE_DIR` is absolute). `index.html` only uses `image_path.split('/')[-1]` to build the `/images/...` URL, so the dashboard tolerates either absolute or relative paths.
- `init/01-init.sql` is a Postgres bootstrap from the Chapter_6 docker experiments and is unrelated to the detections schema. The `detections` table is owned by SQLAlchemy, not by SQL init.
- `AGENTS.md` is stale relative to Chapter_9 (references a `111111data.yaml` filename that no longer exists and pre-pipeline training params). Treat `main.py` + `core/*.py` + the dataset's `data.yaml` as the source of truth.
- Code comments are in **Russian**. Preserve the language when editing existing comments to stay consistent with the rest of the codebase.

## Working across chapter branches

Each `Chapter_*` branch is largely additive. When asked to work on a specific chapter, check out that branch first — files like `videostream.py`, `traffic.mp4`, `main.py`, `core/`, `db/`, `api/` only exist on the relevant chapters. `git log --all --oneline` is the fastest way to see what's where.
