import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,           # We allow up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Fingertip landmark indices (Mediapipe numbering):
# Thumb tip = 4, Index tip = 8, Middle tip = 12, Ring tip = 16, Pinky tip = 20
FINGERTIPS = [4, 8, 12, 16, 20]

# Corresponding “lower joint” landmarks (to compare):
# Thumb lower joint = 2, Index PIP = 6, Middle PIP = 10, Ring PIP = 14, Pinky PIP = 18
FINGER_PIP = [2, 6, 10, 14, 18]

cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks, handedness_label):
    """
    Counts how many fingers are up for a single hand,
    including the thumb as a finger.
    
    hand_landmarks: the 21 Mediapipe landmarks for one hand
    handedness_label: "Left" or "Right" hand (affects thumb logic)
    Returns the number of extended fingers.
    """
    finger_count = 0

    # Convert landmarks to a list of (x, y) in image coordinates
    # Mediapipe gives normalized coords [0..1]; we’ll just compare them as is.
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

    # For each finger (thumb=0, index=1, etc.)
    for i in range(5):
        tip_id = FINGERTIPS[i]
        pip_id = FINGER_PIP[i]

        tip_x, tip_y = landmarks[tip_id]
        pip_x, pip_y = landmarks[pip_id]

        if i == 0:
            # Thumb
            #
            # Heuristic: for the RIGHT hand, the thumb is on the left side of the hand,
            # so if tip_x < pip_x => thumb is extended.
            # For the LEFT hand, the thumb is on the right side, so tip_x > pip_x => extended.
            #
            # This logic assumes your palm faces the camera.
            if handedness_label == "Right":
                if tip_x < pip_x:
                    finger_count += 1
            else:  # "Left" hand
                if tip_x > pip_x:
                    finger_count += 1
        else:
            # Other fingers: if tip_y < pip_y => finger is “up”
            if tip_y < pip_y:
                finger_count += 1

    return finger_count

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip frame horizontally for a mirror-like view (optional)
    frame = cv2.flip(frame, 1)

    # Convert to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # If hands are detected
    if results.multi_hand_landmarks and results.multi_handedness:
        # We'll store finger counts and bounding box centers for text placement
        finger_counts = []
        box_centers = []

        for hand_idx, hand_landmark in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # Determine left/right label from multi_handedness
            handedness_label = results.multi_handedness[hand_idx].classification[0].label
            # Count the fingers for this hand
            count = count_fingers(hand_landmark, handedness_label)
            finger_counts.append((count, handedness_label))

            # Optional: compute bounding box for the hand for text placement
            h, w, c = frame.shape
            xs = [lm.x for lm in hand_landmark.landmark]
            ys = [lm.y for lm in hand_landmark.landmark]
            # min/max in normalized coords
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            # Convert to pixel coords
            box_left, box_right = int(min_x * w), int(max_x * w)
            box_top, box_bottom = int(min_y * h), int(max_y * h)
            # We'll store the bounding box center
            cx = (box_left + box_right) // 2
            cy = (box_top + box_bottom) // 2
            box_centers.append((cx, cy))

        # Draw finger counts near the bounding box center
        for i, (count_label) in enumerate(finger_counts):
            count_val, hand_label = count_label
            cx, cy = box_centers[i]
            cv2.putText(frame, f"{hand_label}: {count_val}",
                        (cx - 30, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Finger Count (Both Hands)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
