import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Load activity history
# -----------------------------

data = pd.read_csv(
    "activity_history.csv",
    header=None,
    names=[
        "timestamp",
        "activity",
        "activity_level",
        "duration_seconds"
    ]
)


# -----------------------------
# Calculate total time
# -----------------------------

total_time = data["duration_seconds"].sum()


# -----------------------------
# Calculate activity score
# -----------------------------

weighted_activity = (
    data["activity_level"]
    * data["duration_seconds"]
).sum()


if total_time == 0:
    activity_score = 0
else:
    activity_score = (
        weighted_activity / total_time
    )


# Keep score between 0 and 100
activity_score = min(
    max(activity_score, 0),
    100
)


# -----------------------------
# Display score
# -----------------------------

print(
    f"Activity Level: "
    f"{activity_score:.1f} / 100"
)


# -----------------------------
# Create bar graph
# -----------------------------

plt.figure(figsize=(6, 5))

plt.bar(
    ["Activity"],
    [activity_score]
)

plt.ylim(0, 100)

plt.ylabel("Score")
plt.title("Pet Activity Level")

plt.text(
    0,
    activity_score + 3,
    f"{activity_score:.1f} / 100",
    ha="center",
    fontsize=14
)

plt.tight_layout()

plt.show()