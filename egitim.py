from ultralytics import YOLO
import os


model = YOLO("yolo11n.pt") 


data_path = os.path.abspath("PC/data.yaml")


results = model.train(
    data=data_path,
    epochs=50,       
    imgsz=512,       
    device="mps"     
)

print("✅ Eğitim tamamlandı!")