import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import serial

from heart_rate_analysis import analyze_heart_rate


# ======================================================
# FILES
# ======================================================

MODEL_FILE = "dog_activity_model.pkl"

PPG_FILE = "ppg_data.csv"

HEART_RATE_FILE = "heart_rate_history.csv"


# ======================================================
# SERIAL
# ======================================================

SERIAL_PORT = "/dev/cu.usbmodem101"
BAUD_RATE = 115200


# ======================================================
# RESTING-CANDIDATE SETTINGS
# ======================================================
#
# IMPORTANT:
#
# We are NOT using this to say the dog is asleep.
#
# We only want the dog/collar to be sufficiently still
# before attempting an optical heart-rate measurement.
#
# 15 seconds is for development testing.
#
# Later we can determine a better value from real data.
#
# ======================================================

REST_CONFIRM_SECONDS = 15

# If substantial movement continues this long,
# the resting candidate ends.

MOVEMENT_CONFIRM_SECONDS = 5


# ======================================================
# PPG SETTINGS
# ======================================================

PPG_STABILIZE_SECONDS = 5

PPG_RECORD_SECONDS = 45

# Testing only.
#
# If the dog remains resting, another measurement
# can be attempted after this amount of time.

PPG_REPEAT_SECONDS = 60


# ======================================================
# ACTIVITY MODEL
# ======================================================

SMOOTHING_WINDOW = 3

FEATURE_NAMES = [
    "movement_mean",
    "movement_rms",
    "movement_peak",
    "movement_variance",
    "rotation_mean",
    "rotation_rms",
    "rotation_peak",
]


# ======================================================
# LOAD MODEL
# ======================================================

model = joblib.load(
    MODEL_FILE
)


# ======================================================
# OPEN SERIAL
# ======================================================

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

print()
print("Connected to XIAO")
print("Starting pet collar...")
print()


# ======================================================
# ACTIVITY PREDICTION
# ======================================================

prediction_history = deque(
    maxlen=SMOOTHING_WINDOW
)


# ======================================================
# RESTING STATE
# ======================================================

resting_candidate = False

rest_candidate_start = None

movement_candidate_start = None


# ======================================================
# PPG STATE
# ======================================================

ppg_active = False

ppg_start_time = None

ppg_recording_start = None

ppg_samples = []

last_ppg_measurement_time = None


# ======================================================
# MOTION CONTEXT DURING PPG
# ======================================================
#
# These let us record what the IMU was seeing
# during the physiological measurement.
#
# ======================================================

ppg_activity_predictions = []

ppg_movement_rms_values = []

ppg_rotation_rms_values = []


# ======================================================
# SEND COMMAND
# ======================================================

def send_command(command):

    ser.write(
        (command + "\n").encode(
            "utf-8"
        )
    )

    ser.flush()


# ======================================================
# ACTIVITY PREDICTION
# ======================================================

def predict_activity(features):

    feature_frame = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES
    )

    raw_prediction = (
        model.predict(
            feature_frame
        )[0]
    )

    prediction_history.append(
        raw_prediction
    )

    return max(
        set(prediction_history),
        key=prediction_history.count
    )


# ======================================================
# START PPG
# ======================================================

def start_ppg_measurement():

    global ppg_active
    global ppg_start_time
    global ppg_recording_start
    global ppg_samples

    global ppg_activity_predictions
    global ppg_movement_rms_values
    global ppg_rotation_rms_values


    if ppg_active:
        return


    print()
    print(
        "Starting low-activity heart-rate measurement..."
    )


    ppg_samples = []

    ppg_activity_predictions = []

    ppg_movement_rms_values = []

    ppg_rotation_rms_values = []


    ppg_active = True

    ppg_start_time = time.time()

    ppg_recording_start = (
        ppg_start_time
        +
        PPG_STABILIZE_SECONDS
    )


    send_command(
        "PPG_ON"
    )


# ======================================================
# STOP PPG
# ======================================================

def stop_ppg_sensor():

    global ppg_active

    send_command(
        "PPG_OFF"
    )

    ppg_active = False


# ======================================================
# SAVE RAW PPG
# ======================================================

def save_ppg():

    if len(ppg_samples) == 0:

        print(
            "No PPG samples collected."
        )

        return False


    with open(
        PPG_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "time_ms",
            "ir"
        ])

        writer.writerows(
            ppg_samples
        )


    print(
        "Saved raw PPG:",
        PPG_FILE
    )

    return True


# ======================================================
# GET MOTION CONTEXT
# ======================================================

def get_motion_context():

    if len(
        ppg_activity_predictions
    ) == 0:

        return {
            "dominant_activity": "unknown",
            "still_percentage": 0,
            "movement_rms": None,
            "rotation_rms": None,
        }


    dominant_activity = max(
        set(
            ppg_activity_predictions
        ),
        key=
            ppg_activity_predictions.count
    )


    still_count = (
        ppg_activity_predictions.count(
            "still"
        )
    )


    still_percentage = (
        still_count
        /
        len(
            ppg_activity_predictions
        )
        *
        100
    )


    average_movement_rms = (
        sum(
            ppg_movement_rms_values
        )
        /
        len(
            ppg_movement_rms_values
        )
    )


    average_rotation_rms = (
        sum(
            ppg_rotation_rms_values
        )
        /
        len(
            ppg_rotation_rms_values
        )
    )


    return {

        "dominant_activity":
            dominant_activity,

        "still_percentage":
            still_percentage,

        "movement_rms":
            average_movement_rms,

        "rotation_rms":
            average_rotation_rms,
    }


# ======================================================
# SAVE HR + MOTION CONTEXT
# ======================================================

def save_measurement(
    result,
    motion_context
):

    timestamp = (
        datetime.now()
        .isoformat(
            timespec="seconds"
        )
    )


    row = {

        "timestamp":
            timestamp,

        "bpm":
            result["bpm"],

        "median_ibi_ms":
            result["median_ibi_ms"],

        "clean_ibi_count":
            result["clean_ibi_count"],

        "raw_ibi_count":
            result["raw_ibi_count"],

        "measurement_duration_seconds":
            result["measurement_duration"],

        "signal_quality":
            result["signal_quality"],

        "dominant_activity":
            motion_context[
                "dominant_activity"
            ],

        "still_percentage":
            round(
                motion_context[
                    "still_percentage"
                ],
                2
            ),

        "movement_rms":
            motion_context[
                "movement_rms"
            ],

        "rotation_rms":
            motion_context[
                "rotation_rms"
            ],

        # We intentionally do NOT automatically
        # label this measurement as sleep.

        "sleep_label":
            "unknown",
    }


    file_exists = (
        Path(
            HEART_RATE_FILE
        ).exists()
    )


    with open(
        HEART_RATE_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=row.keys()
        )

        if not file_exists:

            writer.writeheader()

        writer.writerow(
            row
        )


    print()
    print(
        "HEART-RATE MEASUREMENT SAVED"
    )

    print(
        "----------------------------"
    )

    print(
        "Heart rate:",
        round(
            result["bpm"],
            1
        ),
        "BPM"
    )

    print(
        "Median IBI:",
        round(
            result["median_ibi_ms"],
            1
        ),
        "ms"
    )

    print(
        "Signal quality:",
        result["signal_quality"]
    )

    print(
        "Dominant activity:",
        motion_context[
            "dominant_activity"
        ]
    )

    print(
        "Still during measurement:",
        round(
            motion_context[
                "still_percentage"
            ],
            1
        ),
        "%"
    )

    print(
        "Sleep label: unknown"
    )

    print(
        "Saved to:",
        HEART_RATE_FILE
    )

    print()


# ======================================================
# ANALYZE MEASUREMENT
# ======================================================

def analyze_measurement():

    print()
    print(
        "Analyzing heart rate..."
    )


    result = analyze_heart_rate(
        PPG_FILE
    )


    if result is None:

        print(
            "Measurement rejected:"
        )

        print(
            "not enough reliable heart-rate data."
        )

        print()

        return


    if (
        result[
            "signal_quality"
        ]
        ==
        "poor"
    ):

        print(
            "Measurement rejected:"
        )

        print(
            "poor PPG signal quality."
        )

        print()

        return


    motion_context = (
        get_motion_context()
    )


    save_measurement(
        result,
        motion_context
    )


# ======================================================
# FINISH PPG
# ======================================================

def finish_ppg_measurement():

    global last_ppg_measurement_time


    stop_ppg_sensor()


    print()
    print(
        "PPG measurement finished."
    )

    print(
        "Samples collected:",
        len(
            ppg_samples
        )
    )


    last_ppg_measurement_time = (
        time.time()
    )


    if save_ppg():

        analyze_measurement()


# ======================================================
# CANCEL PPG
# ======================================================

def cancel_ppg_measurement():

    global ppg_samples


    if not ppg_active:
        return


    stop_ppg_sensor()

    ppg_samples = []


    print()
    print(
        "Heart-rate measurement cancelled."
    )

    print(
        "Reason: sustained movement."
    )

    print()


# ======================================================
# PROCESS PPG PACKET
# ======================================================

def process_ppg_packet(
    parts
):

    if not ppg_active:
        return


    if len(parts) != 3:
        return


    try:

        sensor_time = int(
            parts[1]
        )

        ir_value = int(
            parts[2]
        )

    except ValueError:

        return


    # Ignore sensor stabilization period.

    if (
        time.time()
        <
        ppg_recording_start
    ):

        return


    ppg_samples.append(
        (
            sensor_time,
            ir_value
        )
    )


# ======================================================
# CHECK PPG TIMER
# ======================================================

def check_ppg_timer():

    if not ppg_active:
        return


    end_time = (
        ppg_recording_start
        +
        PPG_RECORD_SECONDS
    )


    if (
        time.time()
        >=
        end_time
    ):

        finish_ppg_measurement()


# ======================================================
# PROCESS IMU PACKET
# ======================================================

def process_imu_packet(
    parts
):

    global resting_candidate
    global rest_candidate_start
    global movement_candidate_start
    global last_ppg_measurement_time


    if len(parts) != 8:
        return


    try:

        features = [
            float(value)
            for value
            in parts[1:]
        ]

    except ValueError:

        return


    prediction = (
        predict_activity(
            features
        )
    )


    # Feature positions:
    #
    # 0 movement_mean
    # 1 movement_rms
    # 2 movement_peak
    # 3 movement_variance
    # 4 rotation_mean
    # 5 rotation_rms
    # 6 rotation_peak

    movement_rms = features[1]

    rotation_rms = features[5]


    now = time.time()


    # ==================================================
    # SAVE MOTION CONTEXT WHILE PPG IS RUNNING
    # ==================================================

    if ppg_active:

        ppg_activity_predictions.append(
            prediction
        )

        ppg_movement_rms_values.append(
            movement_rms
        )

        ppg_rotation_rms_values.append(
            rotation_rms
        )


    print(
        "Activity:",
        prediction,
        "| Resting candidate:",
        "YES"
        if resting_candidate
        else "NO",
        "| PPG:",
        "ON"
        if ppg_active
        else "OFF"
    )


    # ==================================================
    # NOT CURRENTLY A RESTING CANDIDATE
    # ==================================================

    if not resting_candidate:

        # For now we use the activity model's
        # "still" state as a trigger to ATTEMPT
        # physiological measurement.
        #
        # It does NOT mean sleeping.

        if prediction == "still":

            if (
                rest_candidate_start
                is None
            ):

                rest_candidate_start = (
                    now
                )

                print(
                    "Low-activity period started..."
                )


            rest_duration = (
                now
                -
                rest_candidate_start
            )


            print(
                "Low activity for:",
                round(
                    rest_duration,
                    1
                ),
                "seconds"
            )


            if (
                rest_duration
                >=
                REST_CONFIRM_SECONDS
            ):

                resting_candidate = True

                rest_candidate_start = None

                movement_candidate_start = None


                print()
                print(
                    "RESTING CANDIDATE"
                )

                print(
                    "This does NOT mean sleep."
                )

                print()


                start_ppg_measurement()


        else:

            rest_candidate_start = None


    # ==================================================
    # RESTING CANDIDATE
    # ==================================================

    else:

        if prediction == "still":

            movement_candidate_start = None


        else:

            if (
                movement_candidate_start
                is None
            ):

                movement_candidate_start = (
                    now
                )

                print(
                    "Possible end of resting period..."
                )


            movement_duration = (
                now
                -
                movement_candidate_start
            )


            print(
                "Movement for:",
                round(
                    movement_duration,
                    1
                ),
                "seconds"
            )


            if (
                movement_duration
                >=
                MOVEMENT_CONFIRM_SECONDS
            ):

                resting_candidate = False

                movement_candidate_start = None

                rest_candidate_start = None


                if ppg_active:

                    cancel_ppg_measurement()


                last_ppg_measurement_time = None


                print()
                print(
                    "RESTING PERIOD ENDED"
                )

                print()


# ======================================================
# REPEAT HR MEASUREMENT
# ======================================================

def check_repeat_measurement():

    if not resting_candidate:
        return


    if ppg_active:
        return


    if (
        last_ppg_measurement_time
        is None
    ):

        return


    elapsed = (
        time.time()
        -
        last_ppg_measurement_time
    )


    if (
        elapsed
        >=
        PPG_REPEAT_SECONDS
    ):

        print()
        print(
            "Low activity continues."
        )

        print(
            "Starting another heart-rate measurement."
        )

        start_ppg_measurement()


# ======================================================
# MAIN LOOP
# ======================================================

try:

    while True:

        line = (
            ser.readline()
            .decode(
                "utf-8",
                errors="ignore"
            )
            .strip()
        )


        check_ppg_timer()

        check_repeat_measurement()


        if not line:
            continue


        parts = line.split(",")

        packet_type = (
            parts[0]
        )


        # ------------------------------------------
        # STATUS
        # ------------------------------------------

        if (
            packet_type
            ==
            "STATUS"
        ):

            print(
                line
            )

            continue


        # ------------------------------------------
        # PPG
        # ------------------------------------------

        if (
            packet_type
            ==
            "PPG"
        ):

            process_ppg_packet(
                parts
            )

            continue


        # ------------------------------------------
        # IMU
        # ------------------------------------------

        if (
            packet_type
            ==
            "IMU"
        ):

            process_imu_packet(
                parts
            )

            continue


# ======================================================
# CTRL+C
# ======================================================

except KeyboardInterrupt:

    print()
    print(
        "Stopping pet collar..."
    )


# ======================================================
# CLEANUP
# ======================================================

finally:

    try:

        send_command(
            "PPG_OFF"
        )

    except Exception:

        pass


    ser.close()


    print(
        "Serial connection closed."
    )