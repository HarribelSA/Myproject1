import os
import time
import cv2

def collect_data(language):
    if language == "ar":
        DATA_PATH = "Data_imag_ar"
        letters = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي") + ["SPACE", "DEL"]
    elif language == "en":
        DATA_PATH = "Data_imag"
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DEL"]
    else:
        print(" اللغة غير مدعومة.")
        return

    os.makedirs(DATA_PATH, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" لم يتم فتح الكاميرا.")
        return

    for letter in letters:
        letter_path = os.path.join(DATA_PATH, letter)
        os.makedirs(letter_path, exist_ok=True)

        print(f"\n التحضير لالتقاط صور الحرف: '{letter}'")
        print(" ضع يدك في المربع الأخضر...")
        time.sleep(10)

        count = 0
        while count < 500:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            x1, y1, x2, y2 = 100, 100, 400, 400
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f'{letter} - {count}/500', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            roi = frame[y1:y2, x1:x2]
            img_path = os.path.join(letter_path, f'{count}.jpg')
            cv2.imwrite(img_path, roi)
            count += 1

            cv2.imshow("جمع الصور", frame)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                print(" تم الإيقاف من قبل المستخدم.")
                cap.release()
                cv2.destroyAllWindows()
                return

            time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
    print(" تم الانتهاء من جمع البيانات.")

if __name__ == "__main__":
    lang = input("اختر اللغة (ar للعربية / en للإنجليزية): ").strip().lower()
    collect_data(lang)
