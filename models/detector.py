import cv2
import numpy as np
from config.settings import Config

class ObjectDetector:
    def __init__(self, net, classes, output_layers):
        self.net = net
        self.classes = classes
        self.output_layers = output_layers

        try:
            print("📥 OpenCV DNN: GPU (OPENCL) hedefi ayarlanıyor...")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT) 
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            print("✅ OpenCV DNN: OPENCL hedefi başarıyla ayarlandı.")
        except Exception as e:
            print(f"❌ OpenCV DNN: OPENCL ayarlanamadı (Hata: {e})")
            print("ℹ️ OpenCV DNN: İşlemler CPU üzerinde devam edecek.")

    
    def preprocess_image(self, img):
        """Image ko YOLO ke liye ready karta hai"""
        blob = cv2.dnn.blobFromImage(
            img, 0.00392, (Config.IMG_SIZE, Config.IMG_SIZE), 
            (0, 0, 0), True, crop=False
        )
        self.net.setInput(blob)
        return blob
    
    def detect_objects(self, img):
        """Objects detect karta hai"""
        self.preprocess_image(img)
        outputs = self.net.forward(self.output_layers)
        return self._process_detections(outputs, img)
    
    def _process_detections(self, outputs, img):
        """Raw detections ko process karta hai"""
        height, width = img.shape[:2]
        boxes, confs, class_ids = [], [], []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > Config.CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confs.append(float(confidence))
                    class_ids.append(class_id)
        
        return boxes, confs, class_ids
    
    def apply_nms(self, boxes, confs):
        """Non-Maximum Suppression apply karta hai"""
        indexes = cv2.dnn.NMSBoxes(boxes, confs, Config.CONFIDENCE_THRESHOLD, Config.NMS_THRESHOLD)
        return indexes