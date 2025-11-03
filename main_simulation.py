import cv2
from face_recognition_module import detect_faces
from tracking_simulation import track_target

def main():
    print("=== Human Tracking Robot Integration (Face + Tracking) ===")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detect_faces(frame)

        # Track each detected face
        for (x, y, w, h) in faces:
            frame = track_target(frame, x, y, w, h)

        cv2.imshow("Face Detection + Tracking Integration", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
