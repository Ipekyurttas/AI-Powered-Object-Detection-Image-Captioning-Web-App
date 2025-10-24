import streamlit as st
from transformers import pipeline
from PIL import Image
import time
import cv2
import numpy as np

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'captioner' not in st.session_state:
    st.session_state.captioner = None

def load_models():
    """Models load karta hai"""
    try:
        with st.spinner("🔄 Loading Object Detection Model..."):
            st.session_state.detector = pipeline(
                "object-detection", 
                model="facebook/detr-resnet-50"
            )
        
        with st.spinner("🔄 Loading Captioning Model..."):
            st.session_state.captioner = pipeline(
                "image-to-text", 
                model="Salesforce/blip-image-captioning-base"
            )
        
        st.session_state.models_loaded = True
        st.success("✅ Models loaded successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return False

def detect_objects(image):
    """Object detection karta hai"""
    if st.session_state.detector is None:
        raise ValueError("Detector not loaded!")
    return st.session_state.detector(image)

def generate_caption(image):
    """Caption generate karta hai"""
    if st.session_state.captioner is None:
        raise ValueError("Captioner not loaded!")
    result = st.session_state.captioner(image)
    return result[0]['generated_text']

def draw_boxes(image, detections):
    """Bounding boxes draw karta hai"""
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    for detection in detections:
        box = detection['box']
        label = detection['label']
        score = detection['score']
        
        x, y, w, h = int(box['xmin']), int(box['ymin']), int(box['xmax'] - box['xmin']), int(box['ymax'] - box['ymin'])
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        label_text = f"{label}: {score:.2f}"
        cv2.putText(img_cv, label_text, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

def main():
    st.set_page_config(
        page_title="AI Image Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤗 AI Image Analysis")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Controls")
    
    # Model loading
    st.sidebar.subheader("🔧 Model Management")
    
    if st.sidebar.button("🔄 Load Models", type="primary"):
        if load_models():
            st.rerun()
    
    # Show model status
    if st.session_state.models_loaded:
        st.sidebar.success("✅ Models Ready!")
    else:
        st.sidebar.warning("⚠️ Models Not Loaded")
    
    st.sidebar.markdown("---")
    confidence = st.sidebar.slider("Confidence", 0.1, 0.9, 0.5)
    
    # Main area
    uploaded_file = st.file_uploader(
        "📁 Upload Image", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.session_state.models_loaded:
            if st.button("🎯 Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    try:
                        start_time = time.time()
                        
                        # Detection
                        detections = detect_objects(image)
                        filtered_detections = [d for d in detections if d['score'] > confidence]
                        
                        # Captioning
                        caption = generate_caption(image)
                        
                        # Draw boxes
                        result_img = draw_boxes(image, filtered_detections)
                        
                        process_time = time.time() - start_time
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🎯 Detection Results")
                            st.image(result_img, use_container_width=True)
                        
                        with col2:
                            st.subheader("📝 AI Caption")
                            st.info(caption)
                            
                            st.metric("Objects Found", len(filtered_detections))
                            st.metric("Process Time", f"{process_time:.2f}s")
                            
                            if filtered_detections:
                                st.write("**Detections:**")
                                for det in filtered_detections:
                                    st.write(f"- {det['label']} ({det['score']:.2%})")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.warning("⚠️ Please load models first using the sidebar button!")
    
    else:
        st.info("👆 Upload an image to get started!")
        
        # Instructions
        st.markdown("""
        ### 🚀 How to Use:
        1. **Click 'Load Models'** in sidebar
        2. **Wait for models to load** (first time may take 2-3 minutes)
        3. **Upload an image**
        4. **Click 'Analyze Image'**
        5. **View results!**
        
        ### ✨ Features:
        - Object Detection with DETR
        - AI Image Captioning with BLIP
        - Real-time analysis
        - No manual downloads needed
        """)

if __name__ == "__main__":
    main()