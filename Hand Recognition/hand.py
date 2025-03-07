import cv2
import mediapipe as mp
import pyautogui
import threading
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,  
    min_detection_confidence=0.5,  
    min_tracking_confidence=0.5
)

frame = None
last_action_time = time.time()
gesture_text = ""
frame_skip = 2
frame_count = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def capture_frames():
    global frame
    while cap.isOpened():
        ret, new_frame = cap.read()
        if not ret:
            break
        frame = new_frame
        time.sleep(0.01)

cap_thread = threading.Thread(target=capture_frames, daemon=True)
cap_thread.start()

while cap.isOpened():
    if frame is None:
        continue

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    processed_frame = cv2.flip(frame.copy(), 1)
    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(processed_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            fingers = [8, 12, 16, 20]
            thumb_tip = landmarks[4]
            thumb_base = landmarks[2]

            h, w, _ = processed_frame.shape
            index_x, index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            middle_x, middle_y = int(landmarks[12].x * w), int(landmarks[12].y * h)

            fingers_up = all(landmarks[f].y < landmarks[f - 2].y for f in fingers)
            thumb_up = thumb_tip.y < thumb_base.y  

            if time.time() - last_action_time > 0.3:  
                if fingers_up and thumb_up:
                    pyautogui.press("up")
                    gesture_text = "TELAPAK TERBUKA - ATAS"
                elif index_y < middle_y:
                    pyautogui.press("up")
                    gesture_text = "ATAS"
                elif index_y > thumb_y:
                    pyautogui.press("down")
                    gesture_text = "BAWAH"
                elif index_x < thumb_x:
                    pyautogui.press("left")
                    gesture_text = "KIRI"
                elif index_x > thumb_x:
                    pyautogui.press("right")
                    gesture_text = "KANAN"

                last_action_time = time.time()

    if gesture_text:
        cv2.putText(processed_frame, f"{gesture_text}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Recognition", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
