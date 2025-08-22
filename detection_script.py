import os
import cv2
import json
import argparse
from ultralytics import YOLO

def draw_boxes(img, results, class_names):
    """Draw bounding boxes on the image."""
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls = int(box.cls.item())
            if cls >= len(class_names):  # Skip invalid classes
                continue
            label = f"{class_names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def save_json_log(filename, results, class_names, log_dir):
    """Save detection results to JSON."""
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf.item()
            cls = int(box.cls.item())
            if cls >= len(class_names):  # Skip invalid classes
                continue
            detections.append({
                "label": class_names[cls],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
    log = {"filename": filename, "detections": detections}
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"{filename}.json"), 'w') as f:
        json.dump(log, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Glove vs No-Glove Hand Detection")
    parser.add_argument('--input', default=r'C:\Users\hp\submission\glove-1\test\images', help='Input folder with images')
    parser.add_argument('--output', default='output', help='Output folder for annotated images')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()

    # Load fine-tuned model
    model = YOLO(r'C:\Users\hp\submission\runs\detect\train3\weights\best.pt')
    class_names = ['glove', 'no_glove']

    os.makedirs(args.output, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    for img_file in os.listdir(args.input):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(args.input, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load {img_file}")
                continue

            results = model(img, conf=args.confidence)
            annotated_img = draw_boxes(img.copy(), results, class_names)
            cv2.imwrite(os.path.join(args.output, img_file), annotated_img)
            save_json_log(img_file, results, class_names, 'logs')
            print(f"Processed {img_file}")

if __name__ == "__main__":
    main()