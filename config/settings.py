class Config:
    # YOLO Model Paths
    YOLO_WEIGHTS = 'models/yolov3.weights'
    YOLO_CONFIG = 'models/yolov3-tiny.cfg'
    COCO_NAMES = 'models/coco.names'
    
    
    # Baaki settings same
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    IMG_SIZE = 416
   
    
    # Captioning Settings
    BLIP_MODEL = 'Salesforce/blip-image-captioning-base'
    MAX_CAPTION_LENGTH = 50
    NUM_BEAMS = 5
    
    # MLflow Settings
    EXPERIMENT_NAME = 'Object_Detection_Captioning'
    
    # Colors for different classes
    COLORS = {
        'person': (255, 0, 0),    # Blue
        'vehicle': (0, 255, 0),   # Green  
        'animal': (0, 0, 255),    # Red
        'default': (255, 255, 0)  # Cyan
    }