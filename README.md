---
title: ACL Risk Predictor
emoji: 🦵
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
license: mit
---

# ACL Injury Risk Assessment using Deep Learning

<p align="center">
  <img src="docs/assets/demo.gif" alt="ACL Demo" width="600"/>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/blazinbanana/ACL-Risk-Predictor"><img src="https://img.shields.io/badge/🤗-HuggingFace%20Space-blue?style=for-the-badge" alt="Hugging Face"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue?style=for-the-badge" alt="Python"/></a>
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.8+-FF6F00?style=for-the-badge" alt="TensorFlow"/></a>
  <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-4.5+-green?style=for-the-badge" alt="OpenCV"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/></a>
  <a href="https://github.com/blazinbanana/ACL-Risk-Predictor/stargazers"><img src="https://img.shields.io/github/stars/blazinbanana/ACL-Risk-Predictor?style=for-the-badge" alt="Stars"/></a>
  <a href="https://github.com/blazinbanana/ACL-Risk-Predictor/issues"><img src="https://img.shields.io/github/issues/blazinbanana/ACL-Risk-Predictor?style=for-the-badge" alt="Issues"/></a>
</p>

---


## Live Demo

[**Try the ACL Risk Predictor on Hugging Face Spaces**](https://huggingface.co/spaces/blazinbanana/ACL-Risk-Predictor)

> Upload your lunge video and get AI-powered ACL injury risk analysis instantly.
> - **Tips**: Record your video in Landscape view. Ensure there is only one person per video for accurate analysis. You can use Front View and then Side View for a more comprehensive and accurate assessment


---

## Overview

> Real-time ACL injury risk assessment from lunge movements using computer vision and deep learning.  
> Designed for athletes, rehab clinics, and biomechanics research.


## Business & Market Potential

The ACL Injury Risk Predictor bridges the gap between sports science, injury prevention, and accessible AI-driven analytics. Designed for both athletic performance monitoring and clinical rehabilitation, the system provides actionable movement insights using only video input.


## Target Users

> Sports Medicine Clinics – For pre-season screening and return-to-play assessments.

> Athletic Teams & Coaches – To monitor form and identify high-risk movement patterns early.

> Rehabilitation Centers – To track progress post-surgery or during physiotherapy.

> Fitness & Training Platforms – To integrate movement risk analysis for users and trainers.


## Value Proposition

> **Prevent costly injuries** – ACL tears can cost $5,000–$20,000 in surgery and recovery; early detection helps reduce risk.

> **Affordable biomechanics** insights – Delivers motion analysis comparable to lab-grade systems at a fraction of the cost.

> **Scalable deployment** – Works on ordinary cameras.

> **Data-driven decisions** – Enable objective tracking of athlete performance and joint health trends over time.


## Sample Output

![Demo](assets/demo.gif)

> Video ACL injury risk assessment from lunge movements using computer vision and deep learning.

---

## Features

- **Recorded Video Analysis**: Process videos for instant ACL risk assessment  
- **Biomechanical Feature Extraction**: Automatic detection of knee valgus, flexion angles, and trunk lean  
- **Deep Learning Model**: LSTM-based classifier trained on movement sequences  
- **Web Application**: Streamlit interface for easy use  
- **Reports**: Gives a Risk Analysis to guide the user on next steps
---

## Prerequisites

- Python 3.9+  
- TensorFlow 2.8+  
- OpenCV 4.5+  

---

## Installation

```bash
git clone https://github.com/blazinbanana/ACL-Risk-Predictor.git
cd ACL-Risk-Predictor
pip install -r requirements.txt
