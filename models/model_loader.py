import cv2
from ultralytics import YOLO
from transformers import pipeline
from config.settings import Config

class ModelLoader:
    @staticmethod
    def load_model(model_type, model_path=None): 
        """Kullanıcının seçtiği modele göre doğru ağırlık dosyasını yükler"""
        try:
            if model_type == "MY YOLO (PC Setup)" or model_type == "YOLO11":
                if model_path:
                    return YOLO(model_path)
                elif model_type == "MY YOLO (PC Setup)":
                    return YOLO("models/yolo11_pc.pt")
                else:
                    return YOLO("yolo11n.pt")
            

            elif model_type == "DETR":
                return pipeline("object-detection", model=Config.DETR_MODEL_NAME)
            
            elif model_type == "YOLOv3":
                net = cv2.dnn.readNet(Config.YOLO_WEIGHTS, Config.YOLO_CONFIG)
                with open(Config.COCO_NAMES, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]
                return {"net": net, "classes": classes}
                
        except Exception as e:
            print(f"❌ {model_type} yükleme hatası: {e}")
            return None