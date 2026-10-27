import serial
import time
import joblib
from collections import deque
from datetime import datetime


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_FILE = "dog_activity_model.pkl"

SERIAL_PORT = "/dev/cu.usbmodem101"
BAUD_RATE = 115200

ACTIVITY_FILE = "activity_history.csv"
CONTINUOUS_FILE = "activity_continuous.csv"
INTENSIVE_FILE = "intensive_activity.csv"

# Number of predictions used for smoothing
SMOOTHING_WINDOW = 3

# High-intensity detection
HIGH_INTENSITY_WINDOW_SECONDS = 30
SHAKING_THRESHOLD = 0.50

# Require at least this many predictions before
# making a high-intensity decision
MIN_PREDICTIONS = 5


# --------------------------------------------------
# Load ML model
# --------------------------------------------------

model = joblib.load(MODEL_FILE)


# --------------------------------------------------
# Connect to XIAO
# --------------------------------------------------

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

print("Connected to XIAO")
print("Starting activity tracker...")
print()


# --------------------------------------------------
# Prediction smoothing
# --------------------------------------------------

prediction_history = deque(
    maxlen=SMOOTHING_WINDOW
)


# --------------------------------------------------
# High-intensity rolling window
# --------------------------------------------------

high_intensity_history = deque()

high_intensity_active = False
high_intensity_start = None


# --------------------------------------------------
# Activity tracking
# --------------------------------------------------

current_activity = None
activity_start_time = None


# --------------------------------------------------
# Main loop
# --------------------------------------------------

try:

    while True:

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()

        if not line:
            continue

        values = line.split(",")

        # We expect 7 sensor features
        if len(values) != 7:
            continue

        try:
            features = [
                float(value)
                for value in values
            ]

        except ValueError:
            continue


        # --------------------------------------------------
        # ML prediction
        # --------------------------------------------------

        raw_prediction = model.predict(
            [features]
        )[0]


        # --------------------------------------------------
        # 3-prediction smoothing
        # --------------------------------------------------

        prediction_history.append(
            raw_prediction
        )

        prediction = max(
            set(prediction_history),
            key=prediction_history.count
        )


        # --------------------------------------------------
        # Current timestamp
        # --------------------------------------------------

        now = time.time()


        # --------------------------------------------------
        # Activity session tracking
        # --------------------------------------------------

        if current_activity is None:

            current_activity = prediction
            activity_start_time = now

        elif prediction != current_activity:

            duration = (
                now - activity_start_time
            )

            timestamp = datetime.fromtimestamp(
                activity_start_time
            ).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )

            # Save completed activity session
            with open(
                ACTIVITY_FILE,
                "a"
            ) as file:

                file.write(
                    f"{timestamp},"
                    f"{current_activity},"
                    f"{duration:.2f}\n"
                )


            # Start new activity session
            current_activity = prediction
            activity_start_time = now


        # --------------------------------------------------
        # Continuous activity logging
        # --------------------------------------------------

        # movement_rms is the second feature
        movement_rms = features[1]

        # Internal activity level
        # Used later by sleep detection.
        activity_level = min(
            movement_rms * 100,
            100
        )

        timestamp = datetime.fromtimestamp(
            now
        ).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )

        with open(
            CONTINUOUS_FILE,
            "a"
        ) as file:

            file.write(
                f"{timestamp},"
                f"{prediction},"
                f"{activity_level:.2f}\n"
            )


        # --------------------------------------------------
        # 30-second rolling high-intensity detection
        # --------------------------------------------------

        high_intensity_history.append(
            (now, prediction)
        )


        # Remove predictions older than 30 seconds
        while (
            high_intensity_history
            and (
                now
                - high_intensity_history[0][0]
                > HIGH_INTENSITY_WINDOW_SECONDS
            )
        ):

            high_intensity_history.popleft()


        # Count shaking predictions
        shaking_count = sum(
            1
            for _, activity
            in high_intensity_history
            if activity == "shaking"
        )


        total_predictions = len(
            high_intensity_history
        )


        if total_predictions >= MIN_PREDICTIONS:

            shaking_ratio = (
                shaking_count
                / total_predictions
            )

        else:

            shaking_ratio = 0


        # --------------------------------------------------
        # Start high-intensity session
        # --------------------------------------------------

        if (
            not high_intensity_active
            and shaking_ratio >= SHAKING_THRESHOLD
        ):

            high_intensity_active = True

            # Use the beginning of the
            # 30-second window as the
            # approximate start time.
            high_intensity_start = (
                high_intensity_history[0][0]
            )

            print()
            print(
                "HIGH-INTENSITY START"
            )
            print(
                f"Shaking ratio: "
                f"{shaking_ratio:.0%}"
            )
            print()


        # --------------------------------------------------
        # End high-intensity session
        # --------------------------------------------------

        elif (
            high_intensity_active
            and shaking_ratio < SHAKING_THRESHOLD
        ):

            high_intensity_active = False

            high_intensity_end = now

            duration = (
                high_intensity_end
                - high_intensity_start
            )


            start_timestamp = (
                datetime.fromtimestamp(
                    high_intensity_start
                ).strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )
            )

            end_timestamp = (
                datetime.fromtimestamp(
                    high_intensity_end
                ).strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )
            )


            # Save high-intensity session
            with open(
                INTENSIVE_FILE,
                "a"
            ) as file:

                file.write(
                    f"{start_timestamp},"
                    f"{end_timestamp},"
                    f"{duration:.2f}\n"
                )


            print()
            print(
                "HIGH-INTENSITY END"
            )

            print(
                f"Duration: "
                f"{duration:.1f} seconds"
            )

            print()


            high_intensity_start = None


        # --------------------------------------------------
        # Display prediction
        # --------------------------------------------------

        print(
            f"Activity: {prediction}"
        )


except KeyboardInterrupt:

    print()
    print("Stopping activity tracker...")


finally:

    # Save the current activity session
    if (
        current_activity is not None
        and activity_start_time is not None
    ):

        now = time.time()

        duration = (
            now - activity_start_time
        )

        timestamp = datetime.fromtimestamp(
            activity_start_time
        ).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )

        with open(
            ACTIVITY_FILE,
            "a"
        ) as file:

            file.write(
                f"{timestamp},"
                f"{current_activity},"
                f"{duration:.2f}\n"
            )


    # Save an unfinished high-intensity session
    if (
        high_intensity_active
        and high_intensity_start is not None
    ):

        now = time.time()

        duration = (
            now - high_intensity_start
        )

        start_timestamp = (
            datetime.fromtimestamp(
                high_intensity_start
            ).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )
        )

        end_timestamp = (
            datetime.fromtimestamp(
                now
            ).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )
        )

        with open(
            INTENSIVE_FILE,
            "a"
        ) as file:

            file.write(
                f"{start_timestamp},"
                f"{end_timestamp},"
                f"{duration:.2f}\n"
            )


    ser.close()

    print("Serial connection closed.")