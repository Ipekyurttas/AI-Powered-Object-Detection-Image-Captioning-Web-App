import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
import os

from models.model_loader import ModelLoader
from models.detector import ObjectDetector
from captioning.caption_generator import CaptionGenerator
from utils.visualizer import ResultVisualizer
from tracking.mlflow_tracker import MLflowTracker

def main():
    st.set_page_config(page_title="AI Image Analyzer", layout="wide")
    st.title("🚀 Multi-Model Object Detection & Captioning")
    st.markdown("---")

    st.sidebar.header("⚙️ Model Ayarları")
    model_type = st.sidebar.selectbox("Algoritma Seçin:", ["YOLO11", "DETR", "MY YOLO (PC Setup)"])
    
    if 'loaded_model_type' not in st.session_state:
        st.session_state.loaded_model_type = None

    if st.sidebar.button("Modeli Aktifleştir", type="primary"):
        with st.spinner(f"{model_type} yükleniyor..."):
            st.session_state.detector_model = ModelLoader.load_model(model_type)
            
            st.session_state.caption_gen = CaptionGenerator()
            st.session_state.tracker = MLflowTracker()
            st.session_state.loaded_model_type = model_type
            st.sidebar.success(f"✅ {model_type} Hazır!")

    uploaded_file = st.file_uploader("Resim Yükle", type=['jpg', 'png', 'jpeg'])

    if uploaded_file and st.session_state.loaded_model_type:
        image = Image.open(uploaded_file)
        img_cv_orig = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if st.button("🎯 Analizi Başlat", type="primary"):
            st.session_state.tracker.start_run(run_name=f"{model_type}_Analysis")
            start_time = time.time()
            
            with st.spinner("Yapay zeka analiz ediyor..."):
                try:
                    effective_type = "YOLO11" if "YOLO" in model_type else model_type
                    detector = ObjectDetector(st.session_state.detector_model, effective_type)
                    boxes, confs, class_ids, classes, indexes = detector.detect(image)
                    
                    ai_caption = st.session_state.caption_gen.generate_ai_caption(uploaded_file)

                    st.session_state.analysis_results = {
                        'boxes': boxes,
                        'confs': confs,
                        'class_ids': class_ids,
                        'classes': classes,
                        'indexes': indexes,
                        'ai_caption': ai_caption,
                        'img_cv': img_cv_orig.copy()
                    }
                    st.session_state.analyzed = True
                    st.session_state.process_time = time.time() - start_time
                
                except Exception as e:
                    st.error(f"Hata oluştu: {e}")

        if st.session_state.get('analyzed'):
            res = st.session_state.analysis_results
            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("📊 Analiz Özeti")
                st.info(f"**AI Yorumu:** {res['ai_caption']}")
                st.metric("İşlem Süresi", f"{st.session_state.process_time:.2f}s")
                st.metric("Toplam Nesne", len(res['boxes']))
                
                if len(res['boxes']) > 0:
                    st.write("### 📦 Nesne Detayları")
                    obj_list = ["Hepsini Göster"] + [
                        f"{i+1}. {res['classes'][res['class_ids'][idx]].capitalize()} (%{res['confs'][idx]*100:.1f})" 
                        for i, idx in enumerate(res['indexes'].flatten())
                    ]
                    
                    selected_option = st.selectbox("Odaklanılacak Nesne:", obj_list)
                else:
                    st.warning("Nesne bulunamadı.")
                    selected_option = "Hepsini Göster"

            with col1:
                st.subheader("🎯 Görselleştirme")
                viz = ResultVisualizer(res['classes'])
                display_img = res['img_cv'].copy()

                if selected_option == "Hepsini Göster":
                    final_img = viz.draw_detections(
                        display_img, 
                        res['boxes'], 
                        res['confs'], 
                        res['class_ids'], 
                        res['indexes']
                    )
                    st.image(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                else:
                    selected_idx_num = int(selected_option.split(".")[0]) - 1
                    actual_idx = res['indexes'].flatten()[selected_idx_num]
                    
                    box = res['boxes'][actual_idx] 
                    label = res['classes'][res['class_ids'][actual_idx]]
                    conf_val = res['confs'][actual_idx]
                    
                    x, y, w, h = box
                    x, y = max(0, x), max(0, y)
                    
                    label_text = f"{label.capitalize()}: %{conf_val*100:.1f}"
                    cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(display_img, label_text, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    crop_img = res['img_cv'][y:y+h, x:x+w]
                    st.write(f"🔍 **Seçili Nesne Yakın Çekim:** {label.capitalize()}")
                    st.image(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB), width=250)

            st.session_state.tracker.log_parameters(st.session_state.loaded_model_type)
            st.session_state.tracker.end_run()
    else:
        st.info("💡 Başlamak için önce yan menüden bir model aktifleştirin.")

if __name__ == "__main__":
    main()