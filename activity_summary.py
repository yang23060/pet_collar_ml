import pandas as pd
from pathlib import Path

ACTIVITY_FILE = "activity_history.csv"
INTENSIVE_FILE = "intensive_activity.csv"


def format_duration(seconds):
    """Convert seconds into hours and minutes."""
    total_minutes = int(seconds // 60)

    hours = total_minutes // 60
    minutes = total_minutes % 60

    if hours > 0:
        return f"{hours}h {minutes}m"

    return f"{minutes}m"


def read_activity_history():
    """Read activity history.

    Only the first three columns are needed:
    timestamp, activity, duration_seconds
    """

    if not Path(ACTIVITY_FILE).exists():
        return pd.DataFrame(
            columns=[
                "timestamp",
                "activity",
                "duration_seconds"
            ]
        )

    data = pd.read_csv(
        ACTIVITY_FILE,
        header=None,
        usecols=[0, 1, 2],
        names=[
            "timestamp",
            "activity",
            "duration_seconds"
        ]
    )

    # Remove a possible header row
    data = data[data["timestamp"] != "timestamp"]

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    data["duration_seconds"] = pd.to_numeric(
        data["duration_seconds"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "timestamp",
            "activity",
            "duration_seconds"
        ]
    )

    return data


def read_intensive_activity():
    """Read high-intensity activity sessions."""

    if not Path(INTENSIVE_FILE).exists():
        return pd.DataFrame(
            columns=[
                "start_time",
                "end_time",
                "duration_seconds"
            ]
        )

    data = pd.read_csv(
        INTENSIVE_FILE,
        header=None,
        names=[
            "start_time",
            "end_time",
            "duration_seconds"
        ]
    )

    # Remove a possible header row
    data = data[data["start_time"] != "start_time"]

    data["start_time"] = pd.to_datetime(
        data["start_time"],
        errors="coerce"
    )

    data["end_time"] = pd.to_datetime(
        data["end_time"],
        errors="coerce"
    )

    data["duration_seconds"] = pd.to_numeric(
        data["duration_seconds"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "start_time",
            "end_time",
            "duration_seconds"
        ]
    )

    return data


# --------------------------------------------------
# Read data
# --------------------------------------------------

activity_data = read_activity_history()
intensive_data = read_intensive_activity()


print()
print("Activity Summary")
print("-----------------------")


if activity_data.empty:
    print("No activity data available.")
    exit()


# --------------------------------------------------
# Add date
# --------------------------------------------------

activity_data["date"] = activity_data["timestamp"].dt.date

if not intensive_data.empty:
    intensive_data["date"] = (
        intensive_data["start_time"].dt.date
    )


# --------------------------------------------------
# Calculate activity
# --------------------------------------------------

daily_summaries = {}


for date, group in activity_data.groupby("date"):

    active_time = group.loc[
        group["activity"].isin([
            "moving",
            "shaking"
        ]),
        "duration_seconds"
    ].sum()

    resting_time = group.loc[
        group["activity"].isin([
            "still",
            "rotating"
        ]),
        "duration_seconds"
    ].sum()

    daily_summaries[date] = {
        "active": active_time,
        "resting": resting_time,
        "high_intensity": 0.0
    }


# --------------------------------------------------
# Add high-intensity activity
# --------------------------------------------------

if not intensive_data.empty:

    for date, group in intensive_data.groupby("date"):

        high_intensity_time = (
            group["duration_seconds"].sum()
        )

        if date not in daily_summaries:
            daily_summaries[date] = {
                "active": 0.0,
                "resting": 0.0,
                "high_intensity": 0.0
            }

        daily_summaries[date]["high_intensity"] = (
            high_intensity_time
        )


# --------------------------------------------------
# Print daily summaries
# --------------------------------------------------

for date in sorted(daily_summaries):

    summary = daily_summaries[date]

    active_time = summary["active"]
    resting_time = summary["resting"]
    high_intensity_time = summary["high_intensity"]

    if active_time > 0:
        high_intensity_percentage = (
            high_intensity_time
            / active_time
            * 100
        )
    else:
        high_intensity_percentage = 0

    print()
    print(date)
    print()

    print("Active Time")
    print(f"  {format_duration(active_time)}")
    print()

    print("High-intensity")
    print(f"  {format_duration(high_intensity_time)}")
    print()

    print("High-intensity of Active Time")
    print(f"  {high_intensity_percentage:.0f}%")
    print()

    print("Resting")
    print(f"  {format_duration(resting_time)}")