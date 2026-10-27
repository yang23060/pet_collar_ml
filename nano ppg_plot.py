import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import (
    butter,
    filtfilt,
    find_peaks
)

FILE = "heart_rate_history.csv"


# --------------------------------
# 1. Load data
# --------------------------------

data = pd.read_csv(FILE)

data["time_seconds"] = (
    data["time_ms"]
    - data["time_ms"].iloc[0]
) / 1000.0

time = data["time_seconds"].to_numpy()
ir = data["ir"].to_numpy()


# --------------------------------
# 2. Calculate sampling rate
# --------------------------------

sample_intervals = (
    data["time_ms"]
    .diff()
    .dropna()
)

average_interval_ms = (
    sample_intervals.median()
)

sampling_rate = (
    1000.0 / average_interval_ms
)

print("Samples:", len(data))
print("Duration:", round(time[-1], 2), "seconds")
print(
    "Sampling rate:",
    round(sampling_rate, 2),
    "Hz"
)


# --------------------------------
# 3. Band-pass filter
# --------------------------------

low_cutoff = 0.7
high_cutoff = 3.5

nyquist = sampling_rate / 2

low = low_cutoff / nyquist
high = high_cutoff / nyquist

b, a = butter(
    3,
    [low, high],
    btype="bandpass"
)

filtered_ir = filtfilt(
    b,
    a,
    ir
)


# --------------------------------
# 4. Adaptive peak detection
# --------------------------------

# We analyze the signal in smaller windows.
# This allows the detector to adapt when
# PPG amplitude changes.

WINDOW_SECONDS = 5

window_samples = int(
    WINDOW_SECONDS * sampling_rate
)

minimum_peak_distance = int(
    0.35 * sampling_rate
)

all_peaks = []


for start in range(
    0,
    len(filtered_ir),
    window_samples
):

    end = min(
        start + window_samples,
        len(filtered_ir)
    )

    segment = filtered_ir[
        start:end
    ]

    # Skip very short sections
    if len(segment) < 10:
        continue

    # Measure local signal amplitude
    local_std = np.std(segment)

    # Adaptive prominence
    prominence = max(
        local_std * 0.30,
        20
    )

    peaks, properties = find_peaks(
        segment,
        distance=minimum_peak_distance,
        prominence=prominence
    )

    # Convert local peak indexes
    # into indexes for the full recording
    peaks = peaks + start

    all_peaks.extend(peaks)


all_peaks = np.array(
    all_peaks,
    dtype=int
)


# --------------------------------
# 5. Ignore startup artifact
# --------------------------------

# Keep the data, but don't use peaks
# during the first two seconds.

valid_peaks = all_peaks[
    time[all_peaks] >= 2
]


# --------------------------------
# 6. Calculate IBI
# --------------------------------

peak_times = time[
    valid_peaks
]

ibi_seconds = np.diff(
    peak_times
)

ibi_ms = (
    ibi_seconds * 1000
)


# --------------------------------
# 7. Basic IBI validation
# --------------------------------

valid_ibi_mask = (
    (ibi_ms >= 350)
    &
    (ibi_ms <= 1500)
)

valid_ibi = ibi_ms[
    valid_ibi_mask
]


# --------------------------------
# 8. Calculate BPM
# --------------------------------

print()

print(
    "Detected heartbeat candidates:",
    len(valid_peaks)
)

if len(valid_ibi) > 0:

    # Median is more resistant to
    # occasional bad intervals than mean.

    median_ibi = np.median(
        valid_ibi
    )

    bpm = (
        60000.0
        / median_ibi
    )

    print(
        "Valid intervals:",
        len(valid_ibi)
    )

    print(
        "Median IBI:",
        round(median_ibi, 1),
        "ms"
    )

    print(
        "Estimated BPM:",
        round(bpm, 1)
    )

    print()

    print("IBI measurements:")

    for ibi in valid_ibi:

        print(
            round(ibi, 1),
            "ms"
        )

else:

    print(
        "No valid heartbeat intervals."
    )


# --------------------------------
# 9. Graph display range
# --------------------------------

lower_limit = np.percentile(
    filtered_ir,
    2
)

upper_limit = np.percentile(
    filtered_ir,
    98
)


# --------------------------------
# 10. Plot
# --------------------------------

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    time,
    filtered_ir,
    label="Filtered PPG"
)

plt.scatter(
    time[valid_peaks],
    filtered_ir[valid_peaks],
    marker="o",
    label="Heartbeat candidates"
)

plt.ylim(
    lower_limit,
    upper_limit
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Filtered PPG"
)

plt.title(
    "MAX30102 Adaptive Heartbeat Detection"
)

plt.legend()

plt.tight_layout()

plt.show()