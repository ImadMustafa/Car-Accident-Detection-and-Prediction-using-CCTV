import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from accident_detector_predictor import run_cnn
from accident_detector_predictor import run_cnn_yolo
from accident_detector_predictor import run_yolo
from models.cnn_model import load_cnnmodel as load_cnn
from models.yolo_model import load_yolomodel as load_yolo


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
# Single filename so each run overwrites the previous output instead of filling the folder.
OUTPUT_VIDEO_BASENAME = "result.mp4"
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB


print("[INFO] Loading models...")
load_cnn()
load_yolo()
print("[INFO] Models loaded successfully")


PROCESSOR_FN = Callable[..., str]
PROCESSORS: Dict[str, PROCESSOR_FN] = {
    "cnn": run_cnn,
    "yolo": run_yolo,
    "hybrid": run_cnn_yolo,
}

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
MAX_TIMELINE_POINTS = 5000


def _allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _make_browser_compatible_mp4(source_path: Path) -> Path:
    """
    Re-encode output to an HTML5-friendly MP4 when ffmpeg is available.
    Many OpenCV writers default to codecs browsers cannot decode.
    On success, replaces source_path in place so you keep a single file (e.g. result.mp4).
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return source_path

    encoder = _pick_h264_encoder(ffmpeg_bin)
    if not encoder:
        return source_path

    fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=str(source_path.parent))
    os.close(fd)
    temp_path = Path(tmp_name)

    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(source_path),
        "-c:v",
        encoder,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(temp_path),
    ]

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        if process.returncode == 0 and temp_path.exists() and temp_path.stat().st_size > 0:
            try:
                os.replace(str(temp_path), str(source_path))
            except OSError:
                return source_path
            return source_path
    except Exception:
        pass
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return source_path


def _pick_h264_encoder(ffmpeg_bin: str) -> str:
    """
    Pick first available H.264 encoder from preferred list.
    """
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        if result.returncode != 0:
            return ""
        output = result.stdout.lower()
    except Exception:
        return ""

    preferred = ["libx264", "h264_nvenc", "libopenh264", "h264_qsv", "h264_amf"]
    for name in preferred:
        if name in output:
            return name
    return ""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


@app.route("/process", methods=["POST"])
def process_video():
    try:
        if "video" not in request.files:
            return jsonify({"error": "No video file was provided."}), 400

        video = request.files["video"]
        mode = request.form.get("mode", "hybrid").strip().lower()

        if video.filename == "":
            return jsonify({"error": "Please select a video file."}), 400

        if mode not in PROCESSORS:
            return jsonify({"error": "Invalid processing mode."}), 400

        if not _allowed_file(video.filename):
            return jsonify({"error": "Unsupported file format."}), 400

        request_id = uuid.uuid4().hex[:10]
        original_name = secure_filename(video.filename)
        ext = original_name.rsplit(".", 1)[1].lower()

        input_filename = f"{request_id}_input.{ext}"

        input_path = UPLOAD_DIR / input_filename
        output_path = OUTPUT_DIR / OUTPUT_VIDEO_BASENAME

        video.save(str(input_path))

        job_id = request_id
        now = time.time()

        with JOBS_LOCK:
            JOBS[job_id] = {
                "job_id": job_id,
                "mode": mode,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "output_url": "",
                "error": "",
                "cancel_requested": False,
                "metrics": {
                    "frame": 0,
                    "mode": mode,
                    "cnn_score": None,
                    "cnn_accident": None,
                    "yolo_score": None,
                    "yolo_collision": None,
                    "vehicles": 0,
                    "prediction_pairs": 0,
                    "prediction_warning": False,
                    "accident_detected": False,
                    "state": "normal",
                },
                "metrics_timeline": [],
            }

        processor = PROCESSORS[mode]

        def _update_metrics(payload: Dict[str, Any]) -> None:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    return
                current = dict(job.get("metrics", {}))
                current.update(payload)
                job["metrics"] = current
                timeline = job.setdefault("metrics_timeline", [])
                timeline.append(dict(current))
                if len(timeline) > MAX_TIMELINE_POINTS:
                    timeline.pop(0)
                job["updated_at"] = time.time()

        def _run_job() -> None:
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["status"] = "processing"
                    JOBS[job_id]["updated_at"] = time.time()

            def _should_stop() -> bool:
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                    return bool(job and job.get("cancel_requested"))

            try:
                processor(
                    str(input_path),
                    str(output_path),
                    metrics_callback=_update_metrics,
                    should_stop=_should_stop,
                )

                if not output_path.exists():
                    raise RuntimeError("Processing finished but output file was not created.")

                playable_path = _make_browser_compatible_mp4(output_path)
                output_url = f"/outputs/{playable_path.name}"

                with JOBS_LOCK:
                    if job_id in JOBS:
                        if JOBS[job_id].get("cancel_requested"):
                            JOBS[job_id]["status"] = "cancelled"
                            JOBS[job_id]["error"] = "Processing was cancelled by user."
                            JOBS[job_id]["updated_at"] = time.time()
                            return
                        JOBS[job_id]["status"] = "done"
                        JOBS[job_id]["output_url"] = output_url
                        JOBS[job_id]["output_filename"] = playable_path.name
                        JOBS[job_id]["updated_at"] = time.time()
            except Exception as exc:
                with JOBS_LOCK:
                    if job_id in JOBS:
                        if JOBS[job_id].get("cancel_requested"):
                            JOBS[job_id]["status"] = "cancelled"
                            JOBS[job_id]["error"] = "Processing was cancelled by user."
                        else:
                            JOBS[job_id]["status"] = "error"
                            JOBS[job_id]["error"] = str(exc)
                        JOBS[job_id]["updated_at"] = time.time()

        threading.Thread(target=_run_job, daemon=True).start()

        return jsonify(
            {
                "ok": True,
                "job_id": job_id,
                "mode": mode,
                "status": "queued",
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/jobs/<job_id>", methods=["GET"])
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found."}), 404
        response = {
            "ok": True,
            "job_id": job["job_id"],
            "mode": job["mode"],
            "status": job["status"],
            "output_url": job.get("output_url", ""),
            "output_filename": job.get("output_filename", ""),
            "error": job.get("error", ""),
        }
        return jsonify(response)


@app.route("/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found."}), 404
        if job["status"] in {"done", "error", "cancelled"}:
            return jsonify(
                {
                    "ok": True,
                    "job_id": job_id,
                    "status": job["status"],
                    "message": "Job already finished.",
                }
            )
        job["cancel_requested"] = True
        job["updated_at"] = time.time()
    return jsonify({"ok": True, "job_id": job_id, "status": "cancelling"})


@app.route("/metrics/<job_id>", methods=["GET"])
def get_metrics(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found."}), 404
        response = {
            "ok": True,
            "job_id": job["job_id"],
            "mode": job["mode"],
            "status": job["status"],
            "metrics": job.get("metrics", {}),
            "error": job.get("error", ""),
        }
        return jsonify(response)


@app.route("/outputs/<path:filename>", methods=["GET"])
def get_output(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, mimetype="video/mp4", as_attachment=False)


@app.route("/timeline/<job_id>", methods=["GET"])
def get_timeline(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"error": "Job not found."}), 404
        timeline: List[Dict[str, Any]] = list(job.get("metrics_timeline", []))
        response = {
            "ok": True,
            "job_id": job["job_id"],
            "mode": job["mode"],
            "status": job["status"],
            "timeline": timeline,
            "error": job.get("error", ""),
        }
        return jsonify(response)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)