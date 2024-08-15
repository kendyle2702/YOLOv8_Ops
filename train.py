from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8m.pt')

results = model.train(data='data.yaml',
                      epochs=5,imgsz=640,)