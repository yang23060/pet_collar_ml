import pandas as pd
from pathlib import Path
from datetime import datetime

from heart_rate_analysis import analyze_heart_rate


# --------------------------------
# Files
# --------------------------------

PPG_FILE = "ppg_data.csv"

HEART_RATE_FILE = (
    "heart_rate_history.csv"
)


# --------------------------------
# Determine sleep type
# --------------------------------

def get_sleep_type(timestamp):

    hour = timestamp.hour

    # For now:
    #
    # 9 PM - 7 AM = night sleep
    # 7 AM - 9 PM = nap
    #
    # Later, the actual sleep detector
    # will provide this information.

    if hour >= 21 or hour < 7:

        return "night"

    return "nap"


# --------------------------------
# Save heart-rate measurement
# --------------------------------

def save_heart_rate(
    bpm,
    median_ibi,
    clean_ibi_count,
    raw_ibi_count,
    measurement_duration,
    signal_quality
):

    now = datetime.now()

    sleep_type = get_sleep_type(
        now
    )


    # Create one heart-rate record

    record = {
        "timestamp": now,

        "bpm": round(
            bpm,
            1
        ),

        "median_ibi_ms": round(
            median_ibi,
            1
        ),

        "clean_ibi_count":
            clean_ibi_count,

        "raw_ibi_count":
            raw_ibi_count,

        "measurement_duration_seconds":
            measurement_duration,

        "signal_quality":
            signal_quality,

        "sleep_type":
            sleep_type
    }


    new_record = pd.DataFrame(
        [record]
    )


    # Check whether history file
    # already exists.

    file_exists = Path(
        HEART_RATE_FILE
    ).exists()


    # Append measurement to history.

    new_record.to_csv(
        HEART_RATE_FILE,
        mode="a",
        header=not file_exists,
        index=False
    )


    # --------------------------------
    # Display result
    # --------------------------------

    print()

    print(
        "Heart-rate measurement saved"
    )

    print(
        "-----------------------------"
    )

    print(
        "Time:",
        now.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print(
        "Sleep type:",
        sleep_type
    )

    print(
        "Resting BPM:",
        round(
            bpm,
            1
        )
    )

    print(
        "Median IBI:",
        round(
            median_ibi,
            1
        ),
        "ms"
    )

    print(
        "Clean IBI:",
        clean_ibi_count
    )

    print(
        "Raw IBI:",
        raw_ibi_count
    )

    print(
        "Measurement duration:",
        measurement_duration,
        "seconds"
    )

    print(
        "Signal quality:",
        signal_quality
    )


# --------------------------------
# Analyze latest PPG recording
# --------------------------------

print()

print(
    "Analyzing sleep heart rate..."
)

print()


# Make sure PPG file exists.

if not Path(PPG_FILE).exists():

    print(
        "PPG file not found:"
    )

    print(
        PPG_FILE
    )

    exit()


# Run the heart-rate analyzer.

result = analyze_heart_rate(
    PPG_FILE
)


# --------------------------------
# Check result
# --------------------------------

if result is None:

    print(
        "Heart-rate measurement rejected."
    )

    print(
        "Not enough reliable PPG data."
    )

else:

    # --------------------------------
    # Save result
    # --------------------------------

    save_heart_rate(

        bpm=
            result["bpm"],

        median_ibi=
            result["median_ibi_ms"],

        clean_ibi_count=
            result["clean_ibi_count"],

        raw_ibi_count=
            result["raw_ibi_count"],

        measurement_duration=
            result["measurement_duration"],

        signal_quality=
            result["signal_quality"]
    )