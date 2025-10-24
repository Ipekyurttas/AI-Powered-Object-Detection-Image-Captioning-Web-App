import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import cv2
from config.settings import Config

class CaptionGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """BLIP model load karta hai"""
        try:
            print("📥 Loading BLIP model for captioning...")
            self.processor = BlipProcessor.from_pretrained(Config.BLIP_MODEL)
            self.model = BlipForConditionalGeneration.from_pretrained(
                Config.BLIP_MODEL,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            print("✅ BLIP model loaded successfully!")
        except Exception as e:
            print(f"❌ BLIP model loading failed: {e}")
            raise e
    
    def generate_ai_caption(self, image_path):
        """AI se creative caption generate karta hai"""
        try:
            # Image load karo (PIL format me)
            pil_image = Image.open(image_path).convert('RGB')
            
            # Preprocess
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=Config.MAX_CAPTION_LENGTH,
                            num_beams=Config.NUM_BEAMS,
                            early_stopping=True
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_length=Config.MAX_CAPTION_LENGTH,
                        num_beams=Config.NUM_BEAMS,
                        early_stopping=True
                    )
            
            # Decode caption
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            return caption
            
        except Exception as e:
            print(f"❌ Caption generation error: {e}")
            return "Unable to generate caption for this image."
    
    def generate_simple_caption(self, boxes, confs, class_ids, classes):
        """Detected objects se simple caption banata hai"""
        if not boxes:
            return "No objects detected in the image"
        
        objects_count = {}
        for class_id in class_ids:
            obj_name = classes[class_id]
            objects_count[obj_name] = objects_count.get(obj_name, 0) + 1
        
        # Simple caption generate karo
        caption_parts = []
        for obj, count in objects_count.items():
            if count == 1:
                caption_parts.append(f"a {obj}")
            else:
                caption_parts.append(f"{count} {obj}s")
        
        return "Image contains " + ", ".join(caption_parts)