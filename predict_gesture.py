import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import os
import time

# Load gesture labels from file (recommended)
if os.path.exists("gestures.txt"):
    with open("gestures.txt", "r") as f:
        gestures = [line.strip() for line in f.readlines()]
else:
    gestures = os.listdir("gestures")
    gestures = [g.replace(".csv", "") for g in gestures]
    gestures.sort()

# Load the trained model
model = load_model('model.keras')
print("[INFO] Model loaded. Input shape:", model.input_shape)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
# Set the sequence length to match training (4 frames)
sequence_length = 4
sequence = deque(maxlen=sequence_length)

# Variables for current prediction update
current_prediction = ""
current_confidence = 0
last_prediction_time = 0
# Duration for which the current prediction is displayed
display_duration = 0.5  # seconds

# Confidence threshold for updating prediction
confidence_threshold = 0.7

print("[INFO] Starting real-time gesture prediction... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        # Process each detected hand (we use one hand in our setup)
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            sequence.append(landmarks)

        # When we have a full sequence of frames, get prediction
        if len(sequence) == sequence_length:
            input_data = np.expand_dims(np.array(sequence), axis=0).astype(np.float32)
            prediction = model.predict(input_data, verbose=0)[0]
            predicted_class = gestures[np.argmax(prediction)]
            confidence = np.max(prediction)
            
            print(f"[DEBUG] Prediction vector: {prediction}, Class: {predicted_class}, Confidence: {confidence:.2f}")
            
            # If confidence meets the threshold, update the current prediction
            if confidence > confidence_threshold:
                current_prediction = predicted_class
                current_confidence = confidence
                last_prediction_time = time.time()
    else:
        # No hand detected; clear the sequence to avoid stale data
        sequence.clear()

    # Display the current (or recently updated) prediction if it is recent enough
    if current_prediction and (time.time() - last_prediction_time < display_duration):
        cv2.putText(image, f"{current_prediction} ({current_confidence:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Gesture Prediction", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Gesture prediction stopped.")
