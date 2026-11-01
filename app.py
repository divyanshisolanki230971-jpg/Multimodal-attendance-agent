import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from modules.vision import analyze_video
from modules.audio import analyze_audio
from modules.agent import fuse_session, teaching_suggestions

st.set_page_config(page_title="Multimodal Attendance & Engagement Agent", page_icon="🎓", layout="wide")

st.title("🎓 Multimodal Attendance & Engagement Agent")
st.caption("Privacy-conscious classroom activity analytics | Prototype")

st.info(
    "This prototype uses anonymous face-detection tracks and coarse audio activity features. "
    "It does not identify people, infer emotions, or reliably measure individual attention. "
    "Use only with informed consent and institution-approved data."
)

mode = st.sidebar.radio("Workspace", ["Explore demo session", "Analyze local media"])
st.sidebar.markdown("---")
st.sidebar.caption("Online demo uses synthetic data. Local media analysis runs in the current app environment.")

def show_dashboard(attendance, timeline, agent, suggestions, audio_timeline=None):
    a, b, c, d = st.columns(4)
    present = sum(x["Status"] == "Present" for x in attendance)
    a.metric("Tracked participants", len(attendance))
    b.metric("Present (threshold met)", present)
    c.metric("Below threshold", len(attendance) - present)
    d.metric("Session activity indicator", f'{agent["engagement_indicator"]:.1f}/100' if agent["engagement_indicator"] is not None else "N/A")

    st.subheader("Attendance")
    st.dataframe(pd.DataFrame(attendance), use_container_width=True, hide_index=True)
    st.subheader("Visual activity over time")
    if timeline:
        st.plotly_chart(px.line(pd.DataFrame(timeline), x="time_sec", y="faces_detected",
                                labels={"time_sec":"Time (seconds)", "faces_detected":"Faces detected"},
                                title="Face detections per sampled interval"), use_container_width=True)
    if audio_timeline:
        st.subheader("Audio activity")
        adf = pd.DataFrame(audio_timeline)
        st.plotly_chart(px.line(adf, x="time_sec", y="rms", title="Audio RMS energy over time"), use_container_width=True)
    st.subheader("Instructor suggestions")
    for s in suggestions:
        st.write("• " + s)
    st.caption(agent["interpretation"])

if mode == "Explore demo session":
    st.subheader("Sample classroom session")
    rng = np.random.default_rng(14)
    times = np.arange(0, 1800, 30)
    faces = np.clip(18 + 3*np.sin(times/170) + rng.normal(0, 1.7, len(times)), 5, 28).astype(int)
    timeline = [{"time_sec": int(t), "faces_detected": int(v)} for t, v in zip(times, faces)]
    attendance = []
    for i in range(1, 26):
        seconds = int(rng.integers(420, 1750))
        attendance.append({"Participant": f"Participant {i:02d}", "Observed presence (sec)": seconds,
                           "Status": "Present" if seconds >= 300 else "Below threshold"})
    audio_timeline = [{"time_sec": int(t), "rms": float(max(.01, .06 + .025*np.sin(t/95) + rng.normal(0,.008)))} for t in times]
    agent = {"visual_presence_proxy": 72.0, "audio_activity_proxy": 61.0, "engagement_indicator": 66.5,
             "interpretation": "Synthetic sample indicators for demonstrating the dashboard. Not real classroom measurements."}
    suggestions = ["Consider a short knowledge check around the mid-session activity dip.",
                   "Review high audio-activity intervals to distinguish discussion from background noise.",
                   "Use these indicators as prompts for instructor review, not learner grading."]
    show_dashboard(attendance, timeline, agent, suggestions, audio_timeline)
    st.download_button("Download sample attendance CSV", pd.DataFrame(attendance).to_csv(index=False),
                       file_name="sample_attendance.csv", mime="text/csv")

else:
    st.subheader("Analyze a local video and/or audio file")
    st.warning("For privacy, use consented recordings. Uploaded media is processed by this app instance; this hosted demo is not an offline environment. For strictly offline processing, run the project locally.")
    video_file = st.file_uploader("Upload classroom video", type=["mp4", "mov", "avi", "mkv"])
    audio_file = st.file_uploader("Upload audio file (optional)", type=["wav", "mp3", "flac", "ogg"])
    threshold = st.slider("Minimum presence duration (seconds)", 1, 60, 5)
    if st.button("Analyze media", type="primary"):
        video_result = None
        audio_result = None
        with st.spinner("Analyzing media..."):
            if video_file is not None:
                video_result = analyze_video(video_file, min_presence_seconds=threshold)
            if audio_file is not None:
                audio_result = analyze_audio(audio_file)
        if video_result is None and audio_result is None:
            st.error("Upload at least one media file.")
        else:
            if video_result and video_result["preview"] is not None:
                st.image(video_result["preview"], caption="Sampled frame with face-detection boxes")
                st.caption(video_result["note"])
            attendance = video_result["attendance"] if video_result else []
            timeline = video_result["timeline"] if video_result else []
            agent = fuse_session(video_result, audio_result)
            suggestions = teaching_suggestions(agent, audio_result)
            show_dashboard(attendance, timeline, agent, suggestions,
                           audio_result["timeline"] if audio_result else None)
            if attendance:
                st.download_button("Download attendance CSV", pd.DataFrame(attendance).to_csv(index=False),
                                   file_name="attendance_report.csv", mime="text/csv")
            if audio_result:
                st.caption(audio_result["note"])

st.markdown("---")
st.caption("Prototype limitations: face detections are not identity verification; anonymous IDs can switch; audio activity is not speech recognition; heuristic scores require validation before operational use.")
