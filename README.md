# 🎯 Object Detection & Image Captioning

A powerful AI web application that detects objects in images and generates intelligent captions using state-of-the-art Hugging Face models.

![Demo](https://img.shields.io/badge/Demo-Live-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

## ✨ Features

- **Object Detection**: Identify 80+ objects using DETR model
- **Image Captioning**: Generate descriptive captions with BLIP model
- **Real-time Processing**: Instant analysis with beautiful UI
- **No Setup Required**: Automatic model downloads
- **Confidence Scoring**: Filter results by detection confidence

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/object-detection-captioning.git
cd object-detection-captioning

pip install -r requirements.txt
Run the App
streamlit run app.py
Usage
Open http://localhost:8501 in your browser

Click "Load Models" in the sidebar

Upload an image (JPG, PNG, JPEG)

Click "Analyze Image"

View detection results and AI-generated caption

🛠️ Tech Stack
Frontend: Streamlit

AI Models: Hugging Face Transformers

Computer Vision: OpenCV, PIL

Deep Learning: PyTorch

Visualization: Plotly

📦 Models Used
Object Detection: facebook/detr-resnet-50
Image Captioning: Salesforce/blip-image-captioning-base





📄 License
MIT License - feel free to use this project for personal or commercial purposes.