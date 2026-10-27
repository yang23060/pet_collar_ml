import pandas as pd
import joblib


# ==========================================
# 1. Load the trained model
# ==========================================

model = joblib.load("dog_activity_model.pkl")


# ==========================================
# 2. Enter new sensor features
# ==========================================

new_data = pd.DataFrame([{
    "movement_mean": 0.30,
    "movement_rms": 0.33,
    "movement_peak": 0.64,
    "movement_variance": 0.02,
    "rotation_mean": 106.43,
    "rotation_rms": 109.92,
    "rotation_peak": 177.11
}])


# ==========================================
# 3. Make prediction
# ==========================================

prediction = model.predict(new_data)


# ==========================================
# 4. Print result
# ==========================================

print("Predicted activity:", prediction[0])