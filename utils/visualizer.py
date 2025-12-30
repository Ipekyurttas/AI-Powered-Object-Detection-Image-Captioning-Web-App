import cv2
import numpy as np
from config.settings import Config

class ResultVisualizer:
    def __init__(self, classes):
        self.classes = classes
    
    def draw_detections(self, img, boxes, confs, class_ids, indexes):
        """Image par bounding boxes draw karta hai"""
        font = cv2.FONT_HERSHEY_PLAIN
        
        if len(boxes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confs[i]
            
                color = self._get_color(label)
                
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{label} {confidence:.2f}", 
                           (x, y - 10), font, 1, color, 2)
        
        return img
    
    def _get_color(self, label):
        """Object type ke hisaab se color return karta hai"""
        if 'person' in label:
            return Config.COLORS['person']
        elif any(vehicle in label for vehicle in ['car', 'bus', 'truck', 'motorcycle']):
            return Config.COLORS['vehicle']
        elif any(animal in label for animal in ['dog', 'cat', 'bird', 'horse']):
            return Config.COLORS['animal']
        else:
            return Config.COLORS['default']
    
    def add_caption_to_image(self, img, caption):
        """Image par caption add karta hai"""
        img_with_text = img.copy()
        
        text_bg_height = 80
        cv2.rectangle(img_with_text, 
                     (0, img.shape[0] - text_bg_height), 
                     (img.shape[1], img.shape[0]), 
                     (0, 0, 0), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        
        words = caption.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) < 60: 
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        

        y_position = img.shape[0] - text_bg_height + 30
        for i, line in enumerate(lines[:2]):  
            cv2.putText(img_with_text, line, 
                       (10, y_position + i * 25), 
                       font, font_scale, font_color, 1)
        
        return img_with_text
    
    def print_detection_summary(self, boxes, confs, class_ids, inference_time):
        """Console par results print karta hai"""
        print(f"\n🎯 Detection Results:")
        print(f"   Objects detected: {len(boxes)}")
        print(f"   Inference time: {inference_time:.2f} seconds")
        
        if boxes:
            print(f"   Detected objects:")
            objects_count = {}
            for class_id in class_ids:
                obj_name = self.classes[class_id]
                objects_count[obj_name] = objects_count.get(obj_name, 0) + 1
            
            for obj, count in objects_count.items():
                print(f"     - {obj}: {count}")