import cv2
import numpy as np

# Define box rotation angle
angle = 0  

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=10, maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, radius = circle

            # Draw the detected circle
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 3)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

            # Define box size
            box_size = radius * 2  
            half_size = box_size // 2

            # Define box corners (before rotation)
            box_pts = np.array([
                [-half_size, -half_size], [half_size, -half_size], 
                [half_size, half_size], [-half_size, half_size]
            ], np.float32)

            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1)

            # Rotate box
            rotated_pts = np.dot(box_pts, rotation_matrix[:, :2].T) + [x, y]

            # Convert to int for drawing
            rotated_pts = np.int32(rotated_pts)

            # Draw the spinning box
            cv2.polylines(frame, [rotated_pts], isClosed=True, color=(255, 0, 0), thickness=2)

    # Increase angle for rotation effect
    angle += 5  
    if angle >= 360:
        angle = 0  

    # Show the frame
    cv2.imshow("Spinning Box Around Object", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
