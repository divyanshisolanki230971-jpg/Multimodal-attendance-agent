from collections import defaultdict

def attendance_from_tracks(track_frame_counts, sampled_frames, fps, min_presence_seconds=5.0):
    """Convert anonymous track frame counts into sustained-presence attendance."""
    seconds_per_sample = 1.0 / max(fps, 0.001)
    records = []
    for track_id, count in sorted(track_frame_counts.items()):
        duration = count * seconds_per_sample
        records.append({
            "Participant": f"Participant {track_id}",
            "Observed presence (sec)": round(duration, 1),
            "Status": "Present" if duration >= min_presence_seconds else "Below threshold"
        })
    return records
