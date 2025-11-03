import cv2

def track_target(frame, x, y, w, h):
    height, width, _ = frame.shape
    center_x = width // 2
    face_center_x = x + w // 2
    deadzone = 60  # still used for direction, but not drawn

    # ❌ Removed grey deadzone area
    # Draw only the center line (optional)
    cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 255), 1)

    # Draw the face rectangle and center dot
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame, (face_center_x, y + h // 2), 5, (0, 255, 0), -1)

    # Determine direction based on face position
    if face_center_x < center_x - deadzone:
        direction = "Turn Left"
    elif face_center_x > center_x + deadzone:
        direction = "Turn Right"
    else:
        direction = "Move Forward"

    # Display direction text
    cv2.putText(frame, f"Direction: {direction} | face_x: {face_center_x} | center: {center_x}",
                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame
