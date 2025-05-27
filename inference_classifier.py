
#المصفوفات
import cv2
import mediapipe as mp
import numpy as np
import joblib

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
model = joblib.load("models/sign_model.pkl")
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DEL"]

cap = cv2.VideoCapture(0)
sentence = ""

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        landmark_points = []
        for lm in landmarks.landmark:
            landmark_points.extend([lm.x, lm.y, lm.z])
        prediction = model.predict([landmark_points])
        predicted_label = labels[prediction[0]]
        if predicted_label == "SPACE":
            sentence += " "
        elif predicted_label == "DEL":
            sentence = sentence[:-1]
        else:
            sentence += predicted_label
        cv2.putText(frame, f'Letter: {predicted_label}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.putText(frame, sentence[-30:], (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("ASL Inference", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
