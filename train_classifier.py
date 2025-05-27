#الخوارزميات
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import os

# تحميل البايانات
data = np.load('Data_imag/dataset.npy', allow_pickle=True).item()
X = data['X']
y = data['y']

#تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# اعداد النموذج
n_trees = 200
model = RandomForestClassifier(n_estimators=n_trees)

#تسجيل وقت البدء
start_time = datetime.now()

# 5. تدريب النموذج
model.fit(X_train, y_train)

# 6. اختبار النموذج
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# 7. تسجيل وقت الانتهاء
end_time = datetime.now()

# 8. حفظ النموذج
model_path = 'models/sign_model.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)

# 9. طباعة الدقة
print(f"Model accuracy: {acc*100:.2f}%")

# 10. تسجيل تقرير نصي
log_content = f"""
===== Sign Language Model Training Report =====
Start Time       : {start_time.strftime('%Y-%m-%d %H:%M:%S')}
End Time         : {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration         : {str(end_time - start_time)}
Total Samples    : {len(X)}
Training Samples : {len(X_train)}
Testing Samples  : {len(X_test)}
Model Type       : Random Forest
Number of Trees  : {n_trees}
Accuracy         : {acc*100:.2f}%
Model Saved To   : {model_path}
================================================
"""

# حفظ التقرير
with open("models/training_log.txt", "w", encoding="utf-8") as log_file:
    log_file.write(log_content)

print("✅ Training log saved to models/training_log.txt")
