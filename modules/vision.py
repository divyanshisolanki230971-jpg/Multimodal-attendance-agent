import cv2
from collections import defaultdict
import numpy as np
import tempfile
import os

def analyze_video(video_file, sample_every_n_frames=8, min_presence_seconds=5.0):
    """
    Detect faces in sampled frames and associate detections with anonymous IDs
    using nearest-centroid matching. This is a prototype tracker, not biometric
    identification; IDs are session-local and may switch after occlusion.
    """
    # Streamlit UploadedFile is a file-like object; OpenCV needs a filesystem path.
    suffix = os.path.splitext(getattr(video_file, "name", "input.mp4"))[1] or ".mp4"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp.write(video_file.getvalue() if hasattr(video_file, "getvalue") else video_file.read())
        temp.close()
        cap = cv2.VideoCapture(temp.name)
    except Exception:
        temp.close()
        os.unlink(temp.name)
        raise
    if not cap.isOpened():
        raise ValueError("Could not open the supplied video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames else 0
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        cap.release()
        raise RuntimeError("OpenCV face detector could not be loaded.")

    tracks = {}
    counts = defaultdict(int)
    next_id = 1
    frame_index = 0
    timeline = []
    preview = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % max(1, sample_every_n_frames) == 0:
            small = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1]))) if frame.shape[1] > 640 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))
            centers = [(int(x + w/2), int(y + h/2)) for x, y, w, h in faces]
            assigned = set()
            new_tracks = {}
            for center in centers:
                best_id, best_dist = None, 90
                for tid, old in tracks.items():
                    if tid in assigned:
                        continue
                    dist = float(np.linalg.norm(np.array(center) - np.array(old)))
                    if dist < best_dist:
                        best_id, best_dist = tid, dist
                if best_id is None:
                    best_id = next_id
                    next_id += 1
                assigned.add(best_id)
                new_tracks[best_id] = center
                counts[best_id] += 1
            tracks = new_tracks
            timestamp = frame_index / fps
            timeline.append({"time_sec": round(timestamp, 1), "faces_detected": len(faces)})
            annotated = small.copy()
            for (x, y, w, h), center in zip(faces, centers):
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (30, 180, 90), 2)
            preview = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_index += 1

    cap.release()
    os.unlink(temp.name)
    # Each sampled observation represents sample_every_n_frames / fps seconds.
    sample_interval = max(1, sample_every_n_frames) / max(fps, 0.001)
    records = []
    for tid, count in sorted(counts.items()):
        observed = count * sample_interval
        records.append({
            "Participant": f"Participant {tid}",
            "Observed presence (sec)": round(observed, 1),
            "Status": "Present" if observed >= min_presence_seconds else "Below threshold"
        })
    return {
        "attendance": records,
        "timeline": timeline,
        "duration_sec": round(duration, 1),
        "sampled_frames": len(timeline),
        "preview": preview,
        "note": "Anonymous session-local IDs from face detections; not verified identities. Tracking may split or merge IDs."
    }
