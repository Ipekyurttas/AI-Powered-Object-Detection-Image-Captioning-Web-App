import mlflow
import mlflow.pytorch
from datetime import datetime
import os
import json
from config.settings import Config

class MLflowTracker:
    def __init__(self):
        self.experiment_name = Config.EXPERIMENT_NAME
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """MLflow experiment setup"""
        mlflow.set_experiment(self.experiment_name)
    
    def start_run(self, run_name=None):
        """New run start karta hai"""
        if run_name is None:
            run_name = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        mlflow.start_run(run_name=run_name)
        print(f"🚀 MLflow Run Started: {run_name}")
    
    def log_parameters(self, detection_params):
        """Detection parameters log karta hai"""
        mlflow.log_params({
            "model": "YOLOv3",
            "dataset": "COCO",
            "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
            "img_size": Config.IMG_SIZE,
            "classes_count": 80,
            "captioning_model": Config.BLIP_MODEL
        })
    
    def log_metrics(self, detection_results):
        """Detection metrics log karta hai"""
        total_objects = len(detection_results['boxes'])
        avg_confidence = sum(detection_results['confidences']) / max(1, total_objects)
        
        metrics = {
            "objects_detected": total_objects,
            "avg_confidence": avg_confidence,
            "detection_time": detection_results['inference_time']
        }
        
        mlflow.log_metrics(metrics)
        return metrics
    
    def log_captions(self, simple_caption, ai_caption):
        """Captions log karta hai"""
        captions_data = {
            "simple_caption": simple_caption,
            "ai_caption": ai_caption,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save captions to temporary file
        with open('temp_captions.json', 'w') as f:
            json.dump(captions_data, f, indent=2)
        
        mlflow.log_artifact('temp_captions.json')
        os.remove('temp_captions.json')  # Cleanup
    
    def log_artifacts(self, output_image_path, input_image_path=None):
        """Images and artifacts log karta hai"""
        # Output image log karo
        if os.path.exists(output_image_path):
            mlflow.log_artifact(output_image_path, "output")
        
        # Input image bhi log kar sakte ho
        if input_image_path and os.path.exists(input_image_path):
            mlflow.log_artifact(input_image_path, "input")
    
    def log_detection_details(self, boxes, confs, class_ids, classes):
        """Detailed detection results log karta hai"""
        detection_details = {
            "detections": []
        }
        
        for i, (box, conf, class_id) in enumerate(zip(boxes, confs, class_ids)):
            detection_details["detections"].append({
                "id": i,
                "object": classes[class_id],
                "confidence": conf,
                "bbox": box
            })
        
        with open('temp_detections.json', 'w') as f:
            json.dump(detection_details, f, indent=2)
        
        mlflow.log_artifact('temp_detections.json')
        os.remove('temp_detections.json')  # Cleanup
    
    def end_run(self):
        """Run complete karta hai"""
        mlflow.end_run()
        print("✅ MLflow Run Completed")