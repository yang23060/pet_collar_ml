import serial
import pandas as pd
import joblib


# Load the trained model
model = joblib.load("dog_activity_model.pkl")

# Connect to the XIAO
ser = serial.Serial(
    "/dev/cu.usbmodem101",
    115200,
    timeout=1
)

print("Waiting for sensor data...")


while True:
    line = ser.readline().decode("utf-8").strip()

    if not line:
        continue

    # Ignore the CSV header
    if line.startswith("movement_mean"):
        continue

    try:
        values = [float(x) for x in line.split(",")]

        if len(values) != 7:
            continue

        # Put the sensor features into the same format
        # used when training the model
        new_data = pd.DataFrame([values], columns=[
            "movement_mean",
            "movement_rms",
            "movement_peak",
            "movement_variance",
            "rotation_mean",
            "rotation_rms",
            "rotation_peak"
        ])

        # Predict the activity
        prediction = model.predict(new_data)
        movement_rms = values[1]

        activity_level = min(movement_rms * 100, 100)

        print(
            "Activity:",
            prediction,
            "| Level:",
            round(activity_level)
        )
    except ValueError:
        continue