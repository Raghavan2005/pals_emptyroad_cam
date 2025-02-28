import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Perform object detection using YOLO
    results = model(frame)
    
    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            
            # Only detect 'bottle' (COCO class ID 39) with confidence > 0.5
            if class_id == 39 and confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                label = f"Bottle {confidence:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Define parking guideline points
    pts = np.array([
        [w // 4, h - 50], [w // 2 - 50, h // 2 + 50],
        [w // 2 + 50, h // 2 + 50], [3 * w // 4, h - 50]
    ], np.int32)

    # Draw parking guidelines (Yellow)
    cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=3)

    # Add green lines for distance markers
    cv2.line(frame, (w // 4, h - 100), (3 * w // 4, h - 100), (0, 255, 0), 2)
    cv2.line(frame, (w // 4, h - 200), (3 * w // 4, h - 200), (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow("Water Bottle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
