import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from tensorflow.keras.models import load_model
from video_overlay_analysis import analyze_video_with_overlay


# Ensure output directory exists
os.makedirs("output", exist_ok=True)


#page set up
st.set_page_config(page_title="ACL Risk Assessment", layout="wide", page_icon="🔬")

st.markdown("<h1 style='text-align:center; color:#2E86C1;'>ACL Injury Risk Assessment Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a video of a lunge exercise to assess ACL injury risk using AI-based motion analysis.</p>", unsafe_allow_html=True)

#load model
model = load_model("model/acl_risk_model_v2.h5")



#side bar
st.sidebar.header("About this App")
st.sidebar.write("""
This tool uses **AI-driven biomechanics** to assess potential 
ACL injury risk from movement patterns in lunge videos.
""")
st.sidebar.info("Developed for sports and rehabilitation analytics.")



#video upload
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])

#process the uploaded video
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Video Analysis..")

        output_path, risk_level, confidence = analyze_video_with_overlay(video_path, model)

        with st.spinner("Analyzing your video..."):
            output_path, risk_level, confidence = analyze_video_with_overlay(video_path, model)

        # Make sure output path is absolute
        output_abs = os.path.abspath(output_path)

        # Display file size 
        st.write("File size (MB):", round(os.path.getsize(output_abs)/1e6, 2))

        # Display analyzed video 
        with open(output_abs, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)


    #Risk analysis column
    with col2:
        st.subheader("Risk Analysis")

        with st.spinner('Evaluating movement patterns...'):

            if risk_level == 1:  # High risk
                st.error("**High Risk Detected**")
                st.progress(float(confidence))
                st.metric(label="Confidence", value=f"{confidence*100:.1f}%")

                st.warning("**Key Issues Detected:**")
                st.write("- Knee valgus (knee caving in)")
                st.write("- Limited knee flexion")
                st.write("- Excessive trunk lean")

                st.info("**Recommendations:**")
                st.write("- Focus on hip strengthening exercises")
                st.write("- Practice proper lunge form using mirror feedback")
                st.write("- Consider consulting a physiotherapist/coach to get actionable tips")

            else:  # Low risk
                st.success("**Low Risk**")
                st.progress(float(confidence))
                st.metric(label="Confidence", value=f"{confidence*100:.1f}%")

                st.info("**Good Movement Patterns:**")
                st.write("- Proper knee alignment")
                st.write("- Adequate knee flexion")
                st.write("- Stable trunk position")

    #clean up
    st.success("Analysis complete")
    

    try:
        os.unlink(video_path)
    except:
        pass
