import cv2
import numpy as np

def decide_direction(face_x: int, frame_center: int, deadzone: int = 50) -> str:
    if face_x < frame_center - deadzone:
        return "Turn Left"
    elif face_x > frame_center + deadzone:
        return "Turn Right"
    else:
        return "Move Forward"

def main():
    # --- Parameters ---
    deadzone = 60                 # tolerance around center
    cam_index = 0                 # change if you have multiple cameras
    draw_thickness = 2

    # Load Haar cascade for face detection
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade. Check your OpenCV installation.")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try a different cam index (0/1).")

    window_name = "Webcam Face Tracking (q=quit, s=save, +/- adjust deadzone)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        center_x = w // 2

        # Detect faces (convert to grayscale for Haar)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        face_center_x = None
        # Choose the largest detected face (if multiple)
        if len(faces) > 0:
            # faces: list of (x, y, w, h)
            areas = [fw * fh for (fx, fy, fw, fh) in faces]
            i_max = int(np.argmax(areas))
            (x, y, fw, fh) = faces[i_max]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), draw_thickness)

            # Face center
            face_center_x = x + fw // 2
            cv2.circle(frame, (face_center_x, y + fh // 2), 6, (0, 255, 0), -1)

        # Draw center line and deadzone
        cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 255), 1)
        cv2.rectangle(frame, (center_x - deadzone, 0), (center_x + deadzone, h), (50, 50, 50), 1)
        cv2.putText(frame, "Deadzone: {} px".format(deadzone), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Decide direction (only if a face is present)
        if face_center_x is not None:
            direction = decide_direction(face_center_x, center_x, deadzone)
            text = f"Direction: {direction} | face_x: {face_center_x} | center: {center_x}"
        else:
            direction = "No Face Detected"
            text = "Direction: N/A | No face detected"

        # Put direction text
        cv2.putText(frame, text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("tracking_webcam_demo.png", frame)
        elif key in (ord('+'), ord('=')):
            deadzone = min(deadzone + 5, w // 2 - 10)
        elif key == ord('-'):
            deadzone = max(deadzone - 5, 10)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
