import numpy as np
import cv2

class ObjectDetector:
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type

    def detect(self, image):
        """Seçili modele göre tespit yapar ve ortak format döndürür"""
        boxes, confs, class_ids, classes = [], [], [], {}

        if self.model_type == "YOLO11":
            results = self.model(image)[0]
            classes = results.names
            for box in results.boxes:
                coords = box.xyxy[0].tolist() 
                boxes.append([int(coords[0]), int(coords[1]), int(coords[2]-coords[0]), int(coords[3]-coords[1])])
                confs.append(float(box.conf[0]))
                class_ids.append(int(box.cls[0]))
            return boxes, confs, class_ids, classes, np.arange(len(boxes))

        elif self.model_type == "DETR":
            results = self.model(image)
            for i, res in enumerate(results):
                b = res['box']
                boxes.append([b['xmin'], b['ymin'], b['xmax']-b['xmin'], b['ymax']-b['ymin']])
                confs.append(res['score'])
                class_ids.append(i)
                classes[i] = res['label']
            return boxes, confs, class_ids, classes, np.arange(len(boxes))
        
        return None, None, None, None, None