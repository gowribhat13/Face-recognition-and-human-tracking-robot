import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("✅ Camera works!")
else:
    print("❌ Camera not accessible.")

cap.release()
