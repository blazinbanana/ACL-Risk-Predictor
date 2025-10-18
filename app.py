import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from tensorflow.keras.models import load_model
from video_overlay_analysis import analyze_video_with_overlay

# --- Environment-safe output directory setup ---
RUNNING_IN_HF = os.getenv("SYSTEM") == "spaces"  # Detect Hugging Face Spaces
OUTPUT_DIR = "/tmp" if RUNNING_IN_HF else "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Page setup ---
st.set_page_config(page_title="ACL Risk Assessment", layout="wide", page_icon="🔬")

st.markdown("<h1 style='text-align:center; color:#2E86C1;'>ACL Injury Risk Assessment Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a video of a lunge exercise to assess ACL injury risk using AI-based motion analysis.</p>", unsafe_allow_html=True)

# --- Load model ---
model = load_model("model/acl_risk_model_v2.h5")

# --- Sidebar ---
st.sidebar.header("About this App")
st.sidebar.write("""
This tool uses **AI-driven biomechanics** to assess potential 
ACL injury risk from movement patterns in lunge videos.
""")
st.sidebar.info("Developed for sports and rehabilitation analytics.")

# --- Video upload ---
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])

# --- Process the uploaded video ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Video Analysis..")

        # Define safe output path
        output_path = os.path.join(OUTPUT_DIR, "output_video.mp4")

        with st.spinner("Analyzing your video..."):
            output_path, risk_level, confidence = analyze_video_with_overlay(video_path, model, output_path)

        # --- Check and display processed video ---
        if os.path.exists(output_path):
            st.write("File size (MB):", round(os.path.getsize(output_path) / 1e6, 2))
            st.video(output_path)
        else:
            st.error("Processed video not found — analysis may have failed.")

    # --- Risk analysis column ---
    with col2:
        st.subheader("Risk Analysis")

        with st.spinner('Evaluating movement patterns...'):
            if risk_level == 1:  # High risk
                st.error("**High Risk Detected**")
                st.progress(float(confidence))
                st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")

                st.warning("**Key Issues Detected:**")
                st.write("- Knee valgus (knee caving in)")
                st.write("- Limited knee flexion")
                st.write("- Excessive trunk lean")

                st.info("**Recommendations:**")
                st.write("- Focus on hip strengthening exercises")
                st.write("- Practice proper lunge form using mirror feedback")
                st.write("- Consider consulting a physiotherapist/coach for detailed guidance")

            else:  # Low risk
                st.success("**Low Risk**")
                st.progress(float(confidence))
                st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")

                st.info("**Good Movement Patterns:**")
                st.write("- Proper knee alignment")
                st.write("- Adequate knee flexion")
                st.write("- Stable trunk position")

    # --- Cleanup temporary file ---
    st.success("Analysis complete")
    try:
        os.unlink(video_path)
    except Exception as e:
        st.warning(f"Cleanup skipped: {e}")
