import serial
import csv
import time

SERIAL_PORT = "/dev/cu.usbmodem101"
BAUD_RATE = 115200

OUTPUT_FILE = "ppg_data.csv"

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

print("Connected to XIAO.")
print("Recording PPG...")
print("Press Ctrl+C to stop.")
print()

with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "time_ms",
        "ir"
    ])

    try:
        while True:

            line = ser.readline().decode(
                "utf-8",
                errors="ignore"
            ).strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) != 2:
                continue

            try:
                sensor_time = int(parts[0])
                ir = int(parts[1])

            except ValueError:
                continue

            writer.writerow([
                sensor_time,
                ir
            ])

            file.flush()

            print(
                f"Time: {sensor_time} ms | "
                f"IR: {ir}"
            )

    except KeyboardInterrupt:

        print()
        print("Recording stopped.")

ser.close()

print("Saved to:", OUTPUT_FILE)