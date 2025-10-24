# 🎵 MoodTune-AI-Playlist-Generator

An intelligent web application built on **Python/PyTorch** that analyzes user text input to detect their emotional state and generates a matching music playlist.

This project uses a single file (`main.py`) to run both the front-end interface (Streamlit) and the PyTorch-powered API (Flask), demonstrating robust integration in a simple environment.

---

## 🚀 Key Features & Tech Stack

| Component | Technology Used | Purpose |
| :--- | :--- | :--- |
| **Mood Detection (AI Core)** | **PyTorch + Hugging Face Transformers** | Handles the Natural Language Processing (NLP) to classify the user's emotional state. |
| **API Backend** | **Flask** | Runs in a background thread to serve the PyTorch model's predictions. |
| **Frontend/UI** | **Streamlit** | Provides the clean, interactive web application interface. |
| **Execution** | **Python Threading** | Allows the Flask API and Streamlit UI to run simultaneously from a single command. |

---

## ⚙️ How to Run Locally

### 1. Installation

First, ensure you have Git and Python installed. Then, install all required libraries using the `requirements.txt` file:

```bash
# Navigate to your project folder
# pip install -r requirements.txt

# Manually list the required packages if the file upload is slow:
pip install streamlit requests flask transformers torch numpy
