import cv2
import os
from ultralytics import YOLO

os.makedirs('outputs/task3_detection', exist_ok=True)

# =========================================================
# PART A: Haar Cascade — Face Detection (use VIDEO frame)
# =========================================================
cap = cv2.VideoCapture('data/Original_video.mp4')
ret, video_frame = cap.read()
cap.release()
cv2.imwrite('outputs/task3_detection/video_frame_for_haar.jpg', video_frame)

gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

img_haar = video_frame.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(img_haar, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img_haar, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

cv2.imwrite('outputs/task3_detection/haar_face_detection.jpg', img_haar)
print(f"[Haar Cascade] Faces detected in video frame: {len(faces)}")

# =========================================================
# PART B: YOLO on CUBE photo — Expected Failure Case
# =========================================================
cube_img = cv2.imread('data/cube_photo1.jpg')
model = YOLO('yolov8n.pt')  

results_cube = model(cube_img)
cube_detections = results_cube[0].boxes

img_yolo_cube = results_cube[0].plot()
cv2.imwrite('outputs/task3_detection/yolo_cube_result.jpg', img_yolo_cube)

print(f"\n[YOLO on Cube Photo]")
if len(cube_detections) == 0:
    print("No objects detected. This confirms YOLO's limitation: "
          "'Rubik's Cube' is not one of the 80 COCO classes it was trained on.")
else:
    for box in cube_detections:
        cls = model.names[int(box.cls)]
        conf = float(box.conf)
        print(f"Detected: {cls} (confidence: {conf:.2f}) -- NOTE: likely a misclassification "
              f"since cube is not a COCO class, so it may be wrongly labeled as something similar "
              f"(e.g., 'book', 'cell phone').")

# =========================================================
# PART C: YOLO on VIDEO frame — Expected Success Case
# =========================================================
results_video = model(video_frame)
video_detections = results_video[0].boxes

img_yolo_video = results_video[0].plot()
cv2.imwrite('outputs/task3_detection/yolo_video_result.jpg', img_yolo_video)

print(f"\n[YOLO on Video Frame]")
if len(video_detections) == 0:
    print("No objects detected in this frame.")
else:
    for box in video_detections:
        cls = model.names[int(box.cls)]
        conf = float(box.conf)
        print(f"Detected: {cls} (confidence: {conf:.2f})")

# =========================================================
# SUMMARY FOR REPORT
# =========================================================
print("\n=== COMPARISON SUMMARY ===")
print(f"{'Test Case':<25}{'Method':<15}{'Result'}")
print(f"{'Video frame (face)':<25}{'Haar Cascade':<15}{len(faces)} face(s) detected")
print(f"{'Cube photo':<25}{'YOLO':<15}{len(cube_detections)} object(s) detected (expected: 0 or misclassified)")
print(f"{'Video frame':<25}{'YOLO':<15}{len(video_detections)} object(s) detected")

print("\nTask 3 completed. Check outputs/task3_detection folder.")