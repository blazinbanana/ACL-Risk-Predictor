# ACL Injury Risk Assessment using Deep Learning

<p align="center">
  <img src="docs/assets/demo.gif" alt="ACL Demo" width="600"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue?style=for-the-badge" alt="Python"/></a>
  <a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-2.8+-FF6F00?style=for-the-badge" alt="TensorFlow"/></a>
  <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-4.5+-green?style=for-the-badge" alt="OpenCV"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/></a>
  <a href="https://github.com/blazinbanana/ACL-Risk-Predictor/stargazers"><img src="https://img.shields.io/github/stars/blazinbanana/ACL-Risk-Predictor?style=for-the-badge" alt="Stars"/></a>
  <a href="https://github.com/blazinbanana/ACL-Risk-Predictor/issues"><img src="https://img.shields.io/github/issues/blazinbanana/ACL-Risk-Predictor?style=for-the-badge" alt="Issues"/></a>
</p>

---

## 🚀 Overview

> Real-time ACL injury risk assessment from lunge movements using computer vision and deep learning.  
> Designed for athletes, rehab clinics, and biomechanics research.
---

![Demo](assets/demo.gif)

> Video ACL injury risk assessment from lunge movements using computer vision and deep learning.

---

## Features

- **Recorded Video Analysis**: Process videos for instant ACL risk assessment  
- **Biomechanical Feature Extraction**: Automatic detection of knee valgus, flexion angles, and trunk lean  
- **Deep Learning Model**: LSTM-based classifier trained on movement sequences  
- **Web Application**: Streamlit interface for easy use  
- **Reports**: Gives a Risk Analysis to guide the user on next steps
- **Tips**: Record your video in Landscape view. Ensure there is only one person per video for accurate analysis. You can use both Front View and Side View for a more comprehensive and accurate assessment

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
