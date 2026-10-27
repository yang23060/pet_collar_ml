import pandas as pd
import numpy as np

from scipy.signal import butter, filtfilt, find_peaks


# ======================================================
# SETTINGS
# ======================================================

DEFAULT_FILE = "ppg_data.csv"

LOW_CUTOFF = 0.7
HIGH_CUTOFF = 3.5

FILTER_ORDER = 3

WINDOW_SECONDS = 5

IGNORE_FIRST_SECONDS = 2


# ======================================================
# HEART-RATE ANALYSIS
# ======================================================

def analyze_heart_rate(file=DEFAULT_FILE):

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    try:
        data = pd.read_csv(file)

    except Exception as error:

        print(
            "Could not read PPG file:",
            error
        )

        return None


    if (
        "time_ms" not in data.columns
        or
        "ir" not in data.columns
    ):

        print(
            "PPG file is missing required columns."
        )

        return None


    time_ms = data[
        "time_ms"
    ].to_numpy()

    ir = data[
        "ir"
    ].to_numpy()


    if len(time_ms) < 20:

        print(
            "Not enough PPG samples."
        )

        return None


    # --------------------------------------------------
    # SAMPLING RATE
    # --------------------------------------------------

    sample_intervals = np.diff(
        time_ms
    )


    valid_sample_intervals = (
        sample_intervals[
            sample_intervals > 0
        ]
    )


    if (
        len(valid_sample_intervals)
        == 0
    ):

        print(
            "Invalid PPG timestamps."
        )

        return None


    median_sample_interval = (
        np.median(
            valid_sample_intervals
        )
    )


    sampling_rate = (
        1000.0
        /
        median_sample_interval
    )


    # --------------------------------------------------
    # BANDPASS FILTER
    # --------------------------------------------------

    nyquist = (
        sampling_rate
        /
        2.0
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


    if high >= 1:

        print(
            "Sampling rate too low "
            "for configured filter."
        )

        return None


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


    # --------------------------------------------------
    # ADAPTIVE PEAK DETECTION
    # --------------------------------------------------

    window_samples = int(
        WINDOW_SECONDS
        *
        sampling_rate
    )


    minimum_peak_distance = max(
        1,
        int(
            0.35
            *
            sampling_rate
        )
    )


    detected_peaks = []


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


        detected_peaks.extend(
            (
                peaks + start
            ).tolist()
        )


    detected_peaks = np.array(
        detected_peaks,
        dtype=int
    )


    # --------------------------------------------------
    # IGNORE FIRST TWO SECONDS
    # --------------------------------------------------

    minimum_time = (
        time_ms[0]
        +
        IGNORE_FIRST_SECONDS
        *
        1000
    )


    detected_peaks = (
        detected_peaks[
            time_ms[
                detected_peaks
            ]
            >=
            minimum_time
        ]
    )


    if len(detected_peaks) < 3:

        return None


    # --------------------------------------------------
    # RAW IBI
    # --------------------------------------------------

    peak_times = (
        time_ms[
            detected_peaks
        ]
    )


    raw_ibi = np.diff(
        peak_times
    )


    raw_ibi_count = len(
        raw_ibi
    )


    if raw_ibi_count == 0:

        return None


    # --------------------------------------------------
    # BASIC PHYSIOLOGICAL FILTER
    # --------------------------------------------------
    #
    # Keep a broad range here.
    #
    # We do NOT want to hard-code one expected
    # heart rate.
    #
    # --------------------------------------------------

    basic_ibi = raw_ibi[
        (raw_ibi >= 350)
        &
        (raw_ibi <= 1500)
    ]


    if len(basic_ibi) < 5:

        return None


    # --------------------------------------------------
    # FIND DOMINANT IBI CLUSTER
    # --------------------------------------------------
    #
    # OLD METHOD:
    #
    # first few intervals
    #       ↓
    # create baseline
    #       ↓
    # accept/reject later intervals
    #
    # Problem:
    # if the beginning contains false peaks,
    # the cleaner can lock onto the wrong rhythm.
    #
    #
    # NEW METHOD:
    #
    # Every candidate interval gets a score.
    #
    # Its score = how many other intervals are
    # reasonably close to it.
    #
    # The strongest cluster becomes our
    # provisional rhythm.
    #
    # --------------------------------------------------

    best_center = None

    best_score = -1


    for candidate in basic_ibi:

        tolerance = (
            candidate
            *
            0.20
        )


        score = np.sum(
            np.abs(
                basic_ibi
                -
                candidate
            )
            <=
            tolerance
        )


        if score > best_score:

            best_score = score

            best_center = candidate


    if best_center is None:

        return None


    # --------------------------------------------------
    # INITIAL CLUSTER
    # --------------------------------------------------

    initial_tolerance = (
        best_center
        *
        0.20
    )


    initial_cluster = (
        basic_ibi[
            np.abs(
                basic_ibi
                -
                best_center
            )
            <=
            initial_tolerance
        ]
    )


    if len(initial_cluster) < 5:

        return None


    # --------------------------------------------------
    # ROBUST CENTER
    # --------------------------------------------------

    cluster_center = (
        np.median(
            initial_cluster
        )
    )


    # --------------------------------------------------
    # FINAL CLEANING
    # --------------------------------------------------
    #
    # Recalculate around the median of the
    # dominant cluster.
    #
    # --------------------------------------------------

    final_tolerance = (
        cluster_center
        *
        0.20
    )


    clean_ibi = (
        basic_ibi[
            np.abs(
                basic_ibi
                -
                cluster_center
            )
            <=
            final_tolerance
        ]
    )


    clean_ibi_count = len(
        clean_ibi
    )


    if clean_ibi_count < 10:

        return None


    # --------------------------------------------------
    # FINAL BPM
    # --------------------------------------------------

    median_ibi = float(
        np.median(
            clean_ibi
        )
    )


    bpm = (
        60000.0
        /
        median_ibi
    )


    # --------------------------------------------------
    # SIGNAL QUALITY
    # --------------------------------------------------
    #
    # For now this still represents beat consistency,
    # not a complete optical signal-quality index.
    #
    # --------------------------------------------------

    consistency_ratio = (
        clean_ibi_count
        /
        raw_ibi_count
    )


    if consistency_ratio >= 0.80:

        signal_quality = "good"

    elif consistency_ratio >= 0.60:

        signal_quality = "fair"

    else:

        signal_quality = "poor"


    # --------------------------------------------------
    # DURATION
    # --------------------------------------------------

    measurement_duration = (
        time_ms[-1]
        -
        time_ms[0]
    ) / 1000.0


    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    return {

        "bpm":
            bpm,

        "median_ibi_ms":
            median_ibi,

        "clean_ibi_count":
            clean_ibi_count,

        "raw_ibi_count":
            raw_ibi_count,

        "measurement_duration":
            measurement_duration,

        "signal_quality":
            signal_quality,

        # Diagnostic information

        "sampling_rate":
            sampling_rate,

        "dominant_ibi_ms":
            float(
                cluster_center
            ),

        "consistency_ratio":
            consistency_ratio,
    }


# ======================================================
# DIRECT RUN
# ======================================================

if __name__ == "__main__":

    result = analyze_heart_rate(
        DEFAULT_FILE
    )


    if result is None:

        print()
        print(
            "Heart Rate Analysis Failed"
        )

    else:

        print()
        print(
            "Heart Rate Analysis"
        )

        print(
            "------------------------"
        )

        print(
            "Sampling rate:",
            round(
                result[
                    "sampling_rate"
                ],
                2
            ),
            "Hz"
        )

        print(
            "Resting BPM:",
            round(
                result[
                    "bpm"
                ],
                1
            )
        )

        print(
            "Median IBI:",
            round(
                result[
                    "median_ibi_ms"
                ],
                1
            ),
            "ms"
        )

        print(
            "Dominant IBI:",
            round(
                result[
                    "dominant_ibi_ms"
                ],
                1
            ),
            "ms"
        )

        print(
            "Clean IBI:",
            result[
                "clean_ibi_count"
            ]
        )

        print(
            "Raw IBI:",
            result[
                "raw_ibi_count"
            ]
        )

        print(
            "Accepted:",
            round(
                result[
                    "consistency_ratio"
                ]
                *
                100,
                1
            ),
            "%"
        )

        print(
            "Signal quality:",
            result[
                "signal_quality"
            ]
        )