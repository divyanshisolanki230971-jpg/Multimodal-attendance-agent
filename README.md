# Multimodal Attendance & Engagement Agent

A privacy-conscious classroom analytics prototype combining anonymous face detection, audio activity analysis, explainable signal fusion, attendance summaries, and instructor suggestions.

## Demo

Deploy this repository using [Streamlit Community Cloud](https://share.streamlit.io/) with `app.py` as the entry point.

## Features

- **Demo session:** Synthetic sample data for a fully interactive dashboard.
- **Local media analysis:** Upload a consented video and optional audio file.
- **Attendance prototype:** Sustained detection duration threshold.
- **Audio analysis:** RMS energy and zero-crossing activity features using Librosa.
- **Multimodal agent:** Transparent heuristic fusion of visual-presence and audio-activity proxies.
- **Instructor suggestions:** Rule-based prompts, not automated decisions.
- **CSV export:** Download attendance results.

## Run locally

Python 3.9+ is recommended (Python 3.10 or 3.11 is a practical choice).

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Deploy online

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app from your repository.
4. Set the entry point to `app.py`.
5. Deploy.

The cloud app defaults to synthetic sample data. The media-upload feature, if enabled, processes uploads in the cloud environment and is therefore **not offline**. Use local execution for privacy-sensitive recordings.

## Architecture

```text
Video file ──> OpenCV face detection ──> anonymous track counts ──> attendance
                                                                  │
Audio file ──> Librosa RMS/ZCR ──> activity features ─────────────┤
                                                                  v
                                                     heuristic signal fusion
                                                                  │
                                          instructor suggestions + dashboard
```

## Important limitations and responsible use

- Face detection is not face recognition. The prototype does not identify students by name.
- Anonymous track IDs are temporary and may split or merge due to occlusion, camera movement, or missed detections.
- Face presence is not proof of attention, comprehension, or participation.
- Audio activity does not reliably distinguish speech from noise and does not identify speakers.
- The combined activity indicator is a heuristic demonstration, not a validated psychological or educational measurement.
- Obtain informed consent, follow institutional policies and applicable privacy law, minimize retention, and avoid using scores for grading or disciplinary decisions.
- Do not commit identifiable classroom video/audio, student records, API keys, or credentials to GitHub.

## Suggested future improvements

- Replace centroid association with a validated multi-object tracker.
- Add optional MediaPipe head-pose landmarks with clear validation.
- Add voice activity detection and classroom-level speech-turn statistics without speaker identification.
- Calibrate thresholds on consented, representative data.
- Add tests, model cards, data-retention controls, and role-based access.

## Tech stack

Python, Streamlit, OpenCV, Librosa, NumPy, Pandas, Plotly.
