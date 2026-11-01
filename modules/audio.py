import librosa
import numpy as np

def analyze_audio(audio_file):
    """Extract coarse audio activity indicators from an uploaded audio file."""
    y, sr = librosa.load(audio_file, sr=16000, mono=True)
    if len(y) == 0:
        raise ValueError("The audio file is empty.")
    hop = 16000
    frame_length = 16000
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    threshold = max(float(np.percentile(rms, 35)), 0.008)
    active = rms > threshold
    rows = [
        {"time_sec": round(float(t), 1), "rms": float(r), "zcr": float(z), "activity": bool(a)}
        for t, r, z, a in zip(times, rms, zcr, active)
    ]
    return {
        "timeline": rows,
        "duration_sec": round(len(y) / sr, 1),
        "active_ratio": float(np.mean(active)),
        "mean_rms": float(np.mean(rms)),
        "note": "Audio activity is not speaker identification or speech transcription. Noise and speech can overlap."
    }
