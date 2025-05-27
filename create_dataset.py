import cv2
import numpy as np
import os
import mediapipe as mp

# إعداد Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# مسار البيانات
DATA_PATH = 'Data_imag'
OUTPUT_FILE = 'Data_imag/dataset.npy'

# مصفوفات البيانات
X = []
y = []

# الأحرف المستخدمة
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DEL"]
label_map = {label: idx for idx, label in enumerate(labels)}

# عدد الصور المطلوبة لكل فئة (تم تغييره إلى 500)
max_images_per_label = 500

# معالجة الصور
for label in labels:
    path = os.path.join(DATA_PATH, label)
    if not os.path.exists(path):
        print(f"Directory not found for label {label}, skipping...")
        continue

    image_files = os.listdir(path)
    count = 0
    for img_file in image_files:
        if count >= max_images_per_label:
            break  # لا تتعدى 200 صورة لكل حرف
        img_path = os.path.join(path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            landmark_points = []
            for lm in landmarks.landmark:
                landmark_points.extend([lm.x, lm.y, lm.z])
            X.append(landmark_points)
            y.append(label_map[label])
            count += 1

# حفظ البيانات
X = np.array(X)
y = np.array(y)
np.save(OUTPUT_FILE, {'X': X, 'y': y})
print(f"تم إنشاء وحفظ البيانات ({len(X)} مثال) في {OUTPUT_FILE}")
