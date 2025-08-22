from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')

# Train on glove-1 dataset
model.train(data=r'C:\Users\hp\submission\glove-1\data.yaml', epochs=6, imgsz=320, batch=4, device='cpu')

