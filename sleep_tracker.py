import pandas as pd
from datetime import datetime, timedelta

# =========================
# Settings
# =========================

FILE = "activity_continuous.csv"
OUTPUT_FILE = "sleep_sessions.csv"

SLEEP_THRESHOLD = 15

# Low activity must continue this long before sleep begins
MIN_SLEEP_MINUTES = 5

# Movement must continue this long before we declare the dog awake
WAKE_CONFIRMATION_MINUTES = 5

# Night is 9 PM -> 7 AM
NIGHT_START_HOUR = 21
NIGHT_END_HOUR = 7


# =========================
# Load activity data
# =========================

data = pd.read_csv(
    FILE,
    names=["timestamp", "activity", "activity_level"]
)

data["timestamp"] = pd.to_datetime(data["timestamp"])

data["activity_level"] = pd.to_numeric(
    data["activity_level"],
    errors="coerce"
)

data = data.dropna(subset=["timestamp", "activity_level"])

data = data.sort_values("timestamp").reset_index(drop=True)


# =========================
# Helper functions
# =========================

def is_low_activity(activity_level):
    return activity_level <= SLEEP_THRESHOLD


def is_night(timestamp):
    hour = timestamp.hour

    # Night: 9 PM -> midnight
    if hour >= NIGHT_START_HOUR:
        return True

    # Night: midnight -> 7 AM
    if hour < NIGHT_END_HOUR:
        return True

    return False


# =========================
# Detect sleep sessions
# =========================

sleep_sessions = []

state = "AWAKE"

potential_sleep_start = None
sleep_start = None

wake_start = None


for i, row in data.iterrows():

    timestamp = row["timestamp"]
    activity_level = row["activity_level"]

    low_activity = is_low_activity(activity_level)

    # ---------------------------------
    # STATE: AWAKE
    # ---------------------------------

    if state == "AWAKE":

        if low_activity:

            # First low-activity point
            if potential_sleep_start is None:
                potential_sleep_start = timestamp

            elapsed = timestamp - potential_sleep_start

            # Low activity has lasted long enough
            if elapsed >= timedelta(minutes=MIN_SLEEP_MINUTES):

                sleep_start = potential_sleep_start
                state = "SLEEPING"

                wake_start = None

        else:

            # Activity interrupted the potential sleep
            potential_sleep_start = None


    # ---------------------------------
    # STATE: SLEEPING
    # ---------------------------------

    elif state == "SLEEPING":

        if low_activity:

            # Still sleeping
            wake_start = None

        else:

            # Movement detected
            if wake_start is None:
                wake_start = timestamp

            elapsed = timestamp - wake_start

            # Movement has continued long enough
            if elapsed >= timedelta(
                minutes=WAKE_CONFIRMATION_MINUTES
            ):

                sleep_end = wake_start

                duration = (
                    sleep_end - sleep_start
                ).total_seconds()

                sleep_sessions.append({
                    "start_time": sleep_start,
                    "end_time": sleep_end,
                    "duration_seconds": duration
                })

                # Reset
                state = "AWAKE"
                sleep_start = None
                potential_sleep_start = None
                wake_start = None


# =========================
# Handle sleep continuing
# until the end of the data
# =========================

if state == "SLEEPING" and sleep_start is not None:

    sleep_end = data.iloc[-1]["timestamp"]

    duration = (
        sleep_end - sleep_start
    ).total_seconds()

    sleep_sessions.append({
        "start_time": sleep_start,
        "end_time": sleep_end,
        "duration_seconds": duration
    })


# =========================
# Save sleep sessions
# =========================

if sleep_sessions:

    sleep_df = pd.DataFrame(sleep_sessions)

    sleep_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

else:

    sleep_df = pd.DataFrame(
        columns=[
            "start_time",
            "end_time",
            "duration_seconds"
        ]
    )

    sleep_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


# =========================
# Display detected sessions
# =========================

print()
print("Detected Sleep Sessions")
print("-----------------------")

if len(sleep_df) == 0:

    print("No sleep sessions detected.")

else:

    for _, session in sleep_df.iterrows():

        start = pd.to_datetime(
            session["start_time"]
        )

        end = pd.to_datetime(
            session["end_time"]
        )

        duration_minutes = (
            session["duration_seconds"] / 60
        )

        print(
            f"Start: {start}"
        )

        print(
            f"End: {end}"
        )

        print(
            f"Duration: {duration_minutes:.2f} minutes"
        )

        print()


# =========================
# Classify night sleep vs naps
# =========================

if len(sleep_df) > 0:

    sleep_df["start_time"] = pd.to_datetime(
        sleep_df["start_time"]
    )

    sleep_df["end_time"] = pd.to_datetime(
        sleep_df["end_time"]
    )

    sleep_df["is_night"] = sleep_df[
        "start_time"
    ].apply(is_night)


# =========================
# Daily sleep summary
# =========================

print()
print("Daily Sleep")
print("-----------------------")

if len(sleep_df) == 0:

    print("No sleep data available.")

else:

    # Use the date of the morning wake-up
    # for overnight sleep.
    #
    # Example:
    #
    # Oct 5 11 PM -> Oct 6 7 AM
    #
    # belongs to Oct 6.

    daily_summaries = {}

    for _, session in sleep_df.iterrows():

        start = session["start_time"]
        end = session["end_time"]

        duration = session["duration_seconds"]

        # If sleep starts before midnight
        # and continues into the next day,
        # assign it to the date when it ends.
        if start.date() != end.date():

            sleep_day = end.date()

        else:

            sleep_day = start.date()

        if sleep_day not in daily_summaries:

            daily_summaries[sleep_day] = {
                "total": 0,
                "night": 0,
                "nap": 0
            }

        daily_summaries[sleep_day]["total"] += duration

        if is_night(start):

            daily_summaries[sleep_day]["night"] += duration

        else:

            daily_summaries[sleep_day]["nap"] += duration


    for date, summary in sorted(
        daily_summaries.items()
    ):

        total_minutes = summary["total"] / 60
        night_minutes = summary["night"] / 60
        nap_minutes = summary["nap"] / 60

        total_hours = int(total_minutes // 60)
        total_remaining = int(total_minutes % 60)

        night_hours = int(night_minutes // 60)
        night_remaining = int(night_minutes % 60)

        nap_hours = int(nap_minutes // 60)
        nap_remaining = int(nap_minutes % 60)

        print(date)

        print(
            f"  Total Sleep: "
            f"{total_hours}h {total_remaining}m"
        )

        print(
            f"  Night Sleep: "
            f"{night_hours}h {night_remaining}m"
        )

        print(
            f"  Daytime Sleep: "
            f"{nap_hours}h {nap_remaining}m"
        )