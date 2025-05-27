import sys  # استيراد مكتبة sys للتحكم في نظام التشغيل وعمليات الخروج من البرنامج
import cv2  # استيراد مكتبة OpenCV لمعالجة الفيديو والصور
import numpy as np  # استيراد مكتبة numpy للتعامل مع المصفوفات والعمليات العددية
import joblib  # استيراد joblib لتحميل نموذج التعلم الآلي المحفوظ
import time  # استيراد مكتبة time للتعامل مع الزمن والفواصل الزمنية

from PyQt5.QtWidgets import (  # استيراد عناصر واجهة المستخدم من PyQt5
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel,
    QTextEdit, QHBoxLayout, QStackedLayout, QMessageBox, QComboBox
)
from PyQt5.QtCore import QTimer, Qt, QUrl  # استيراد بعض الوظائف الأساسية من PyQt5
from PyQt5.QtGui import QImage, QPixmap, QFont  # استيراد عناصر الرسومات والخطوط
from PyQt5.QtWebEngineWidgets import QWebEngineView  # استيراد عنصر عرض صفحات الويب
import mediapipe as mp  # استيراد مكتبة Mediapipe لمعالجة تتبع اليد




class SignRecognizer(QWidget):  # تعريف فئة لواجهة التعرف على لغة الإشارة، ترث من QWidget
    def __init__(self, language='english', parent_stack=None):  # دالة المُهيئ مع اختيار اللغة والواجهة الأب
        super().__init__()  # استدعاء مُهيئ الفئة الأم
        self.delay_seconds = 3  # تأخير بالثواني قبل قبول حرف جديد لتجنب التكرار السريع
        self.last_time = time.time()  # توقيت آخر تحديث للحرف
        self.last_letter = None  # آخر حرف تم التعرف عليه
        self.current_text = ""  # النص الحالي المعروض
        self.mp_draw = mp.solutions.drawing_utils  # أداة رسم النقاط على اليد
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                         min_detection_confidence=0.7, min_tracking_confidence=0.7)  # إعدادات تتبع اليد
        self.mp_hands = mp.solutions.hands  # تهيئة مكتبة Mediapipe لتتبع اليد
        self.timer = QTimer()  # إنشاء مؤقت لتحديث الفيديو
        self.cap = cv2.VideoCapture(0)  # فتح الكاميرا (الجهاز رقم 0)
        self.letters = (list("ابتثجحخدذرزسشصطظعغفقكلمنهوي") + ["SPACE", "DEL"]
                        if self.language == 'arabic' else
                        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["SPACE", "DEL"])
        self.model = joblib.load("models/sign_model.pkl")  # محاولة تحميل النموذج المحفوظ
        self.clear_btn = QPushButton("🧹 مسح النص")  # زر لمسح النص المعروض
        self.text_output = QTextEdit(self)  # إنشاء مربع نص متعدد الأسطر لعرض النص الناتج
        self.video_label = QLabel(self)  # إنشاء مربع عرض الفيديو
        self.language = language  # حفظ اللغة المحددة (عربية أو إنجليزية)
        self.parent_stack = parent_stack  # حفظ المرجع للواجهة الأب (stack) للعودة إليها
        self.init_ui()  # استدعاء دالة تهيئة واجهة المستخدم

    def init_ui(self):  # دالة لإنشاء وتصميم الواجهة
        layout = QVBoxLayout()  # إنشاء تخطيط عمودي للعناصر

        back_btn = QPushButton("🔙 رجوع")  # زر رجوع
        back_btn.setFont(QFont("Arial", 12))  # تعيين نوع وحجم الخط للزر
        back_btn.setCursor(Qt.PointingHandCursor)  # تعيين شكل المؤشر عند المرور على الزر
        back_btn.setStyleSheet("""  # تعيين أنماط الزر (الألوان، الحدود، الحواف، المسافات)
            QPushButton {
                background-color: #81C784;
                color: white;
                border-radius: 10px;
                padding: 8px 16px;
                max-width: 100px;
            }
            QPushButton:hover {
                background-color: #66BB6A;
            }
        """)
        back_btn.clicked.connect(self.go_back)  # ربط حدث الضغط على الزر بدالة العودة
        layout.addWidget(back_btn, alignment=Qt.AlignLeft)  # إضافة الزر إلى التخطيط مع محاذاته لليسار

        title = QLabel("🧠 تحويل لغة الإشارة إلى نص")  # عنوان النافذة
        title.setFont(QFont("Arial", 18, QFont.Bold))  # تعيين الخط ليكون عريض وحجم 18
        title.setStyleSheet("color: #2E7D32;")  # تعيين لون النص
        title.setAlignment(Qt.AlignCenter)  # محاذاة النص للوسط
        layout.addWidget(title)  # إضافة العنوان إلى التخطيط

        self.video_label.setFixedSize(800, 450)  # تعيين أبعاد مربع الفيديو
        self.video_label.setStyleSheet("border: 2px solid #4CAF50; border-radius: 8px; background-color: #f0f0f0;")  # تنسيق مربع الفيديو (حدود، خلفية، حواف مدورة)
        layout.addWidget(self.video_label)  # إضافة مربع الفيديو إلى التخطيط

        self.text_output.setReadOnly(True)  # تعيين مربع النص ليكون للعرض فقط (غير قابل للتحرير)
        self.text_output.setFont(QFont("Arial", 14))  # تعيين الخط وحجمه لمربع النص
        self.text_output.setStyleSheet("""  # تعيين أنماط مربع النص (خلفية، حدود، ألوان، padding)
            background-color: #ffffff;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 8px;
            color: #333333;
        """)
        self.text_output.setFixedHeight(100)  # تعيين ارتفاع مربع النص
        layout.addWidget(self.text_output)  # إضافة مربع النص إلى التخطيط

        self.clear_btn.setFont(QFont("Arial", 12))  # تعيين الخط للزر
        self.clear_btn.setStyleSheet("""  # تعيين أنماط الزر (ألوان، حواف، padding)
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 10px 20px;
                max-width: 150px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.clear_btn.clicked.connect(self.clear_text)  # ربط حدث الضغط على الزر بدالة مسح النص

        btn_layout = QHBoxLayout()  # إنشاء تخطيط أفقي للأزرار
        btn_layout.addStretch()  # إضافة مساحة فارغة قبل الزر لدفعه للوسط
        btn_layout.addWidget(self.clear_btn)  # إضافة زر المسح إلى التخطيط الأفقي
        btn_layout.addStretch()  # إضافة مساحة فارغة بعد الزر لدفعه للوسط
        layout.addLayout(btn_layout)  # إضافة التخطيط الأفقي إلى التخطيط الرئيسي

        self.setLayout(layout)  # تعيين التخطيط الرئيسي للواجهة

        # تحميل نموذج التعرف من ملف pkl
        try:
            pass
        except Exception as e:  # في حال حدوث خطأ أثناء التحميل
            QMessageBox.critical(self, "خطأ", f"لم يتم العثور على النموذج:\n{e}")  # إظهار رسالة خطأ
            sys.exit(1)  # إنهاء البرنامج

        # تعيين قائمة الحروف بناء على اللغة (عربية أو إنجليزية)

        self.timer.timeout.connect(self.update_frame)  # ربط انتهاء المؤقت بدالة تحديث الإطار
        self.timer.start(30)  # بدء المؤقت بفاصل 30 ملي ثانية

    def clear_text(self):  # دالة لمسح النص
        self.current_text = ""  # تفريغ النص الحالي
        self.text_output.setText("")  # تحديث مربع النص ليصبح فارغاً

    def update_frame(self):  # دالة تحديث عرض الفيديو ومعالجة الإشارات
        ret, frame = self.cap.read()  # قراءة إطار جديد من الكاميرا
        if not ret:  # إذا لم يتم قراءة الإطار بنجاح
            return  # الخروج من الدالة

        frame = cv2.flip(frame, 1)  # قلب الصورة أفقياً (لتحريك المرآة)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # تحويل الصورة من BGR إلى RGB

        result = self.hands.process(img_rgb)  # معالجة الصورة لتتبع اليد

        if result.multi_hand_landmarks:  # إذا تم اكتشاف معالم اليد
            hand_landmarks = result.multi_hand_landmarks[0]  # أخذ أول يد فقط
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)  # رسم معالم اليد على الإطار

            landmarks = []  # قائمة لتخزين إحداثيات النقاط
            for lm in hand_landmarks.landmark:  # لكل نقطة في اليد
                landmarks.extend([lm.x, lm.y, lm.z])  # إضافة إحداثيات x,y,z للنقاط
            landmarks = np.array(landmarks).flatten()  # تحويل القائمة إلى مصفوفة numpy مفلطحة

            try:
                prediction = self.model.predict([landmarks])[0]  # توقع الحرف باستخدام النموذج
            except Exception:
                return  # إذا حدث خطأ في التوقع، تجاهل الإطار

            if prediction < 0 or prediction >= len(self.letters):  # إذا كانت النتيجة خارج النطاق
                return  # تجاهل الإطار

            letter = self.letters[prediction]  # الحصول على الحرف المتوقع من القائمة
            cv2.putText(frame, f'{letter}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (76, 175, 80), 3)  # عرض الحرف على الفيديو

            current_time = time.time()  # الوقت الحالي

            if letter != self.last_letter and (current_time - self.last_time) > self.delay_seconds:  # إذا كان الحرف مختلف ومر وقت كافٍ
                if letter == "SPACE":  # إذا كان الحرف space
                    self.current_text += " "  # إضافة فراغ للنص
                elif letter == "DEL":  # إذا كان الحرف DEL (حذف)
                    self.current_text = self.current_text[:-1]  # حذف آخر حرف من النص
                else:
                    self.current_text += letter  # إضافة الحرف إلى النص

                self.text_output.setText(self.current_text)  # تحديث مربع النص
                self.last_letter = letter  # تحديث آخر حرف تم إدخاله
                self.last_time = current_time  # تحديث وقت آخر إدخال

        h, w, ch = frame.shape  # أخذ أبعاد الإطار (ارتفاع، عرض، قنوات الألوان)
        bytes_per_line = ch * w  # حساب عدد البايت لكل سطر في الصورة
        qt_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)  # تحويل إطار OpenCV إلى QImage
        self.video_label.setPixmap(QPixmap.fromImage(qt_img))  # عرض الصورة في QLabel

    def go_back(self):  # دالة العودة للواجهة السابقة
        self.cap.release()  # تحرير الكاميرا
        self.timer.stop()  # إيقاف مؤقت تحديث الفيديو
        if self.parent_stack:  # إذا كان هناك مرجع للواجهة الأب
            self.parent_stack.setCurrentIndex(0)  # الانتقال إلى الصفحة الرئيسية
        self.deleteLater()  # حذف هذه الواجهة لتحرير الذاكرة

    def closeEvent(self, event):  # دالة استدعاء عند إغلاق النافذة
        self.cap.release()  # تحرير الكاميرا
        event.accept()  # قبول حدث الإغلاق

class MainWindow(QWidget):  # تعريف الواجهة الرئيسية للبرنامج
    def __init__(self):  # دالة المُهيئ
        super().__init__()  # استدعاء مُهيئ الفئة الأم
        self.lang_combo = QComboBox()
        self.setWindowTitle("برنامج لغة الإشارة")  # تعيين عنوان النافذة
        self.setGeometry(100, 100, 900, 700)  # تعيين أبعاد وموقع النافذة

        self.stacked_layout = QStackedLayout()  # إنشاء تخطيط مكدس متعدد الصفحات

        self.main_menu = self.create_main_menu()  # إنشاء صفحة القائمة الرئيسية
        self.language_selection = self.create_language_selection()  # إنشاء صفحة اختيار اللغة
        self.learning_options = self.create_learning_options_with_buttons()  # إنشاء صفحة خيارات التعلم

        self.web_page = QWidget()
        self.web_layout = QVBoxLayout()
        self.web_view = QWebEngineView()

        back_btn = QPushButton("🔙 رجوع")
        back_btn.setFont(QFont("Arial", 12))
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.setStyleSheet("""QPushButton {
            background-color: #81C784; color: white; border-radius: 10px;
            padding: 8px 16px; max-width: 100px;
        } QPushButton:hover { background-color: #66BB6A; }""")
        back_btn.clicked.connect(self.go_back_from_web)

        self.web_layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        self.web_layout.addWidget(self.web_view)
        self.web_page.setLayout(self.web_layout)

        self.stacked_layout.addWidget(self.main_menu)
        self.stacked_layout.addWidget(self.language_selection)
        self.stacked_layout.addWidget(self.learning_options)
        self.stacked_layout.addWidget(self.web_page)

        self.setLayout(self.stacked_layout)
        self.setStyleSheet("background-color: #f1f8e9;")

    def create_main_menu(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(25)

        # شعار
        logo = QLabel()
        pixmap = QPixmap("signs.png").scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)

        title = QLabel("مرحباً بك في برنامجي للغة الإشارة")
        title.setFont(QFont("Arial", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1B5E20;")
        layout.addWidget(title)

        btn_font = QFont("Arial", 14, QFont.Bold)

        translate_btn = QPushButton("💬 تحويل الإشارة إلى نص")
        translate_btn.setFont(btn_font)
        translate_btn.clicked.connect(lambda: self.stacked_layout.setCurrentWidget(self.language_selection))
        layout.addWidget(translate_btn)

        learn_btn = QPushButton("📘 تعلم لغة الإشارة")
        learn_btn.setFont(btn_font)
        learn_btn.clicked.connect(lambda: self.stacked_layout.setCurrentWidget(self.learning_options))
        layout.addWidget(learn_btn)

        exit_btn = QPushButton("🚪 خروج")
        exit_btn.setFont(btn_font)
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn)

        for btn in (translate_btn, learn_btn, exit_btn):
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""QPushButton {
                background-color: #4CAF50; color: white; border-radius: 12px;
                padding: 15px;
            } QPushButton:hover { background-color: #388E3C; }""")

        widget.setLayout(layout)
        return widget

    def create_language_selection(self):
        widget = QWidget()
        layout = QVBoxLayout()
        label = QLabel("اختر لغة التعرف:")
        label.setFont(QFont("Arial", 14))
        layout.addWidget(label)

        self.lang_combo.addItems(["عربية", "English"])
        layout.addWidget(self.lang_combo)

        btn = QPushButton("ابدأ")
        btn.setFont(QFont("Arial", 12))
        btn.clicked.connect(self.start_sign_recognition)
        layout.addWidget(btn)

        back_btn = QPushButton("🔙 رجوع")
        back_btn.clicked.connect(lambda: self.stacked_layout.setCurrentIndex(0))
        layout.addWidget(back_btn)

        widget.setLayout(layout)
        return widget

    def create_learning_options_with_buttons(self):
        # واجهة تعلم تحتوي على زرين لتعلم الحروف والكلمات مع نفس التنسيق
        widget = QWidget()
        layout = QVBoxLayout()

        label = QLabel("اختر نوع المحتوى التعليمي:")
        label.setFont(QFont("Arial", 14))
        layout.addWidget(label)

        # زر لتعلم الحروف
        letters_btn = QPushButton("📘 تعلم الحروف (يوتيوب)")
        letters_btn.setFont(QFont("Arial", 12))
        letters_btn.setCursor(Qt.PointingHandCursor)
        letters_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """)
        letters_btn.clicked.connect(lambda: self.open_learning_video_by_key("تعلم الحروف (يوتيوب)"))
        layout.addWidget(letters_btn)

        # زر لتعلم الكلمات
        words_btn = QPushButton("🗣️ تعلم الكلمات (يوتيوب)")
        words_btn.setFont(QFont("Arial", 12))
        words_btn.setCursor(Qt.PointingHandCursor)
        words_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """)
        words_btn.clicked.connect(lambda: self.open_learning_video_by_key("تعلم الكلمات (يوتيوب)"))
        layout.addWidget(words_btn)

        back_btn = QPushButton("🔙 رجوع")
        back_btn.clicked.connect(lambda: self.stacked_layout.setCurrentIndex(0))
        layout.addWidget(back_btn)

        widget.setLayout(layout)
        return widget

    def start_sign_recognition(self):
        language = 'arabic' if self.lang_combo.currentText() == "عربية" else 'english'
        self.sign_page = SignRecognizer(language=language, parent_stack=self.stacked_layout)
        self.stacked_layout.addWidget(self.sign_page)
        self.stacked_layout.setCurrentWidget(self.sign_page)

    def open_learning_video_by_key(self, key):
        urls = {
            "تعلم الحروف (يوتيوب)": "https://www.youtube.com/watch?v=L2TpDr0CAPM&ab_channel=Ana9outk",
            "تعلم الكلمات (يوتيوب)": "https://www.youtube.com/watch?v=example_words_video"
        }
        url = urls.get(key, "https://www.youtube.com")
        self.web_view.load(QUrl(url))
        self.stacked_layout.setCurrentWidget(self.web_page)

    def go_back_from_web(self):
        self.stacked_layout.setCurrentIndex(0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
