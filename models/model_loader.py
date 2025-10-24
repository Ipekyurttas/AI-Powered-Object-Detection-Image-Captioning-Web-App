import cv2
import numpy as np
from config.settings import Config

class ModelLoader:
    def __init__(self):
        self.net = None
        self.classes = []
        self.output_layers = []
    
    def load_yolo_model(self):
        """YOLO model load karta hai"""
        try:
            print("📥 Loading YOLO model...")
            self.net = cv2.dnn.readNet(Config.YOLO_WEIGHTS, Config.YOLO_CONFIG)
            
            # Load COCO classes
            with open(Config.COCO_NAMES, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Get output layers
            layer_names = self.net.getLayerNames()
            self.output_layers = [
                layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()
            ]
            
            print(f"✅ YOLO model loaded! Classes: {len(self.classes)}")
            return True
            
        except Exception as e:
            print(f"❌ YOLO model loading error: {e}")
            return False
    
    def get_model_info(self):
        return self.net, self.classes, self.output_layers