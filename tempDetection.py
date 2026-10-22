import os
import time
import urllib.request

import cv2
import mediapipe as mp

CAM_W, CAM_H = 568, 320       # camera / processing resolution (16:9, 320p)
WIN_W, WIN_H = 1280, 720      # display window size

MODEL_PATH = "gesture_recognizer.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

# Landmark index pairs that make up the hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm edge
]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading gesture recognizer model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def draw_hand(frame, landmarks, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 4, color, -1)
    return pts


def _d(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def finger_states(pts):
    """Return [thumb, index, middle, ring, pinky] -> True if extended.
    Uses distances, so it works at any hand rotation."""
    wrist = pts[0]
    # (tip, pip) landmark indices for index, middle, ring, pinky
    pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    fingers = [_d(pts[t], wrist) > _d(pts[p], wrist) * 1.1 for t, p in pairs]
    # Thumb: tip farther from the pinky base than the thumb's IP joint
    thumb = _d(pts[4], pts[17]) > _d(pts[3], pts[17]) * 1.05
    return [thumb] + fingers


def classify_custom(pts):
    """Your own gestures. Return a name, or None to fall back to the built-in model."""
    thumb, index, middle, ring, pinky = finger_states(pts)
    hand_size = _d(pts[0], pts[9])  # wrist to middle-finger base
    pinching = _d(pts[4], pts[8]) < hand_size * 0.3

    if pinching and middle and ring and pinky:
        return "OK"
    if index and pinky and not middle and not ring:
        return "Rock"
    if index and middle and ring and not pinky and not thumb:
        return "Three"
    if index and middle and ring and pinky and not thumb:
        return "Four"
    # Add more rules here
    return None


def main():
    ensure_model()

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (try a different index than 0)")
    # Ask the camera itself for a low resolution (MJPG often unlocks more modes)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    print("Camera resolution:",
          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    start = time.monotonic()
    last_ts = -1

    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # If the camera ignored the requested size, downscale the frame itself
            if frame.shape[1] != CAM_W or frame.shape[0] != CAM_H:
                frame = cv2.resize(frame, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)

            # Mirror the image (selfie view) so Left/Right labels are correct
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp must be in milliseconds and strictly increasing
            ts = int((time.monotonic() - start) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            result = recognizer.recognize_for_video(mp_image, ts)

            # Upscale for display only; landmarks are normalized so they map directly
            frame = cv2.resize(frame, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)

            for i, landmarks in enumerate(result.hand_landmarks):
                pts = draw_hand(frame, landmarks)

                gesture, score = "None", 0.0
                if i < len(result.gestures) and result.gestures[i]:
                    gesture = result.gestures[i][0].category_name
                    score = result.gestures[i][0].score
                # Custom rules take priority over the built-in model
                custom = classify_custom(pts)
                if custom:
                    gesture, score = custom, 1.0
                side = result.handedness[i][0].category_name

                # Label placed just above the wrist of each hand
                x, y = pts[0]
                cv2.putText(
                    frame,
                    f"{side}: {gesture} ({score:.2f})",
                    (max(x - 60, 10), max(y + 40, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("MediaPipe Gestures", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
