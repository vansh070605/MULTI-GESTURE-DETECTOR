import cv2
import csv
import os
import numpy as np
import mediapipe as mp

gesture_name = input("Enter gesture name (e.g., peace, fist): ").strip().lower()
save_path = f"gestures/{gesture_name}.csv"

os.makedirs("gestures", exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print(f"[INFO] Collecting data for '{gesture_name}'. Press 's' to save frame, 'q' to quit.")

with open(save_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Save frame if 's' is pressed
                key = cv2.waitKey(1)
                if key == ord('s'):
                    row = []
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                    writer.writerow(row)
                    print("[INFO] Frame saved.")

        cv2.imshow("Collecting Gesture Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Data collection for '{gesture_name}' finished.")
