from Ultralytics import YOLO


model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
p = r"C:\Users\o.abdulmalik\Documents\Shadow-Mitigation\cloud-type-classification2\images"


results = model.train(data=p, epochs=100, imgsz=640)
