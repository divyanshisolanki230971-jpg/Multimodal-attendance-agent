def fuse_session(video_result=None, audio_result=None):
    """
    Explainable heuristic score based on observable proxies.
    It must not be interpreted as a psychological measurement of attention.
    """
    visual = None
    if video_result and video_result.get("timeline"):
        vals = [x["faces_detected"] for x in video_result["timeline"]]
        visual = min(100.0, (sum(vals) / max(1, len(vals))) * 20.0)
    audio = None
    if audio_result:
        audio = min(100.0, audio_result["active_ratio"] * 100.0)

    available = [x for x in [visual, audio] if x is not None]
    score = round(sum(available) / len(available), 1) if available else None
    return {
        "visual_presence_proxy": round(visual, 1) if visual is not None else None,
        "audio_activity_proxy": round(audio, 1) if audio is not None else None,
        "engagement_indicator": score,
        "interpretation": "A heuristic session activity indicator, not a validated measure of individual attention."
    }

def teaching_suggestions(agent_result, audio_result=None):
    suggestions = []
    score = agent_result.get("engagement_indicator")
    if score is None:
        suggestions.append("Process a video or audio sample to generate session insights.")
    elif score < 35:
        suggestions.append("Consider a short interactive check-in, question, or activity to re-engage the class.")
    elif score < 65:
        suggestions.append("Try alternating explanation with a brief discussion or knowledge check.")
    else:
        suggestions.append("The measured activity indicators are relatively high; identify which teaching segment coincided with this period.")
    if audio_result and audio_result["active_ratio"] > 0.75:
        suggestions.append("Audio activity is high. Review the session context to distinguish productive discussion from background noise.")
    suggestions.append("Treat all suggestions as prompts for instructor review, not automated judgments about learners.")
    return suggestions
