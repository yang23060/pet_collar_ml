import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks


# ======================================================
# SETTINGS
# ======================================================

FILE = "ppg_data.csv"

LOW_CUTOFF = 0.7
HIGH_CUTOFF = 3.5

FILTER_ORDER = 3

WINDOW_SECONDS = 5

IGNORE_FIRST_SECONDS = 2


# ======================================================
# LOAD DATA
# ======================================================

data = pd.read_csv(FILE)

time_ms = data["time_ms"].to_numpy()
ir = data["ir"].to_numpy()


# Convert milliseconds to seconds.

time_seconds = (
    time_ms - time_ms[0]
) / 1000.0


# ======================================================
# DETERMINE ACTUAL SAMPLING RATE
# ======================================================

sample_intervals = np.diff(
    time_ms
)

median_interval_ms = np.median(
    sample_intervals
)

sampling_rate = (
    1000.0
    /
    median_interval_ms
)


print()
print("PPG Diagnostic")
print("------------------------")

print(
    "Samples:",
    len(ir)
)

print(
    "Median sample interval:",
    round(
        median_interval_ms,
        2
    ),
    "ms"
)

print(
    "Estimated sampling rate:",
    round(
        sampling_rate,
        2
    ),
    "Hz"
)


# ======================================================
# BANDPASS FILTER
# ======================================================

nyquist = (
    0.5
    *
    sampling_rate
)

low = (
    LOW_CUTOFF
    /
    nyquist
)

high = (
    HIGH_CUTOFF
    /
    nyquist
)


b, a = butter(
    FILTER_ORDER,
    [low, high],
    btype="band"
)


filtered_ir = filtfilt(
    b,
    a,
    ir
)


# ======================================================
# FIND PEAKS
# ======================================================

window_samples = int(
    WINDOW_SECONDS
    *
    sampling_rate
)


minimum_peak_distance = int(
    0.35
    *
    sampling_rate
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


    if len(segment) < 10:
        continue


    local_std = np.std(
        segment
    )


    prominence = max(
        local_std * 0.30,
        20
    )


    peaks, _ = find_peaks(
        segment,
        distance=minimum_peak_distance,
        prominence=prominence
    )


    peaks = (
        peaks + start
    )


    all_peaks.extend(
        peaks.tolist()
    )


all_peaks = np.array(
    all_peaks,
    dtype=int
)


# ======================================================
# IGNORE STARTUP
# ======================================================

valid_start = (
    time_seconds
    >=
    IGNORE_FIRST_SECONDS
)


first_valid_index = np.argmax(
    valid_start
)


all_peaks = all_peaks[
    all_peaks
    >=
    first_valid_index
]


# ======================================================
# CALCULATE IBI
# ======================================================

peak_times_ms = (
    time_ms[
        all_peaks
    ]
)


ibi = np.diff(
    peak_times_ms
)


# Basic physiological range used by current prototype.

valid_ibi = ibi[
    (ibi >= 350)
    &
    (ibi <= 1500)
]


print(
    "Detected peaks:",
    len(all_peaks)
)

print(
    "Raw intervals:",
    len(ibi)
)

print(
    "Intervals in basic range:",
    len(valid_ibi)
)


if len(valid_ibi) > 0:

    median_ibi = np.median(
        valid_ibi
    )

    bpm = (
        60000
        /
        median_ibi
    )


    print(
        "Basic median IBI:",
        round(
            median_ibi,
            1
        ),
        "ms"
    )

    print(
        "Basic BPM:",
        round(
            bpm,
            1
        )
    )


print()


# ======================================================
# PRINT INTERVALS
# ======================================================

print("Detected intervals")
print("------------------------")


for interval in ibi:

    print(
        round(
            interval,
            1
        ),
        "ms"
    )


# ======================================================
# RAW PPG PLOT
# ======================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time_seconds,
    ir
)

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "IR value"
)

plt.title(
    "Raw MAX30102 PPG Signal"
)

plt.tight_layout()

plt.show()


# ======================================================
# FILTERED SIGNAL + DETECTED PEAKS
# ======================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    time_seconds,
    filtered_ir,
    label="Filtered PPG"
)


plt.scatter(
    time_seconds[
        all_peaks
    ],
    filtered_ir[
        all_peaks
    ],
    label="Detected peaks"
)


plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Filtered amplitude"
)

plt.title(
    "Filtered PPG with Detected Heartbeat Candidates"
)

plt.legend()

plt.tight_layout()

plt.show()


# ======================================================
# IBI PLOT
# ======================================================

if len(ibi) > 0:

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        peak_times_ms[1:]
        / 1000.0,
        ibi,
        marker="o"
    )


    plt.axhline(
        350,
        linestyle="--",
        label="350 ms lower limit"
    )


    plt.axhline(
        1500,
        linestyle="--",
        label="1500 ms upper limit"
    )


    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "IBI (ms)"
    )

    plt.title(
        "Detected Inter-Beat Intervals"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()