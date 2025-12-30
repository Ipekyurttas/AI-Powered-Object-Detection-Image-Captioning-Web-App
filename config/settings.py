class Config:
    YOLO11_MODEL_PATH = "yolo11n.pt" 
    DETR_MODEL_NAME = "facebook/detr-resnet-50"
    
    YOLO_WEIGHTS = 'models/yolov3.weights'
    YOLO_CONFIG = 'models/yolov3-tiny.cfg'
    COCO_NAMES = 'models/coco.names'
    

    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    
    BLIP_MODEL = 'Salesforce/blip-image-captioning-base'
    MAX_CAPTION_LENGTH = 50
    NUM_BEAMS = 5
    
    EXPERIMENT_NAME = 'Object_Detection_Analysis'

    IMG_SIZE = 640
    
    COLORS = {
        'person': (255, 0, 0),
        'vehicle': (0, 255, 0),
        'animal': (0, 0, 255),
        'default': (255, 255, 0)
    }