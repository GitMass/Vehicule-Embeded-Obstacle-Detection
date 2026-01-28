import serial
import pynmea2
import time
import csv
import os

PORT = "/dev/ttyAMA0"
BAUD = 9600
CSV_FILE = "gps_positions.csv"

def append_to_csv(timestamp, lat, lon, num_sats, fix_type):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "latitude", "longitude", "num_sats", "fix_type", "google_maps"])

        maps_url = f"https://www.google.com/maps?q={lat},{lon}"
        writer.writerow([timestamp, lat, lon, num_sats, fix_type, maps_url])

def run_gps_analysis():
    lat, lon = None, None
    num_sats = 0      # Satellites used for Fix
    sats_visible = 0  # Satellites seen in sky
    fix_type = 1
    last_lat, last_lon = None, None

    try:
        ser = serial.Serial(PORT, baudrate=BAUD, timeout=1)
        print(f"--- Diagnostic GPS démarré sur {PORT} ---")

        while True:
            start_time = time.time()

            while time.time() < start_time + 2:
                try:
                    line = ser.readline().decode("ascii", errors="replace").strip()
                    if not line or not line.startswith('$'):
                        continue

                    # GSV: Shows how many satellites are visible in the sky
                    if "GSV" in line:
                        msg = pynmea2.parse(line)
                        sats_visible = int(msg.num_svs) if msg.num_svs else 0

                    # GGA: Shows how many satellites are actually being USED for coordinates
                    if "GGA" in line:
                        msg = pynmea2.parse(line)
                        num_sats = int(msg.num_sats) if msg.num_sats else 0
                        if msg.gps_qual > 0:
                            lat, lon = msg.latitude, msg.longitude

                    # GSA: Shows the Fix type (1=No Fix, 2=2D, 3=3D)
                    if "GSA" in line:
                        msg = pynmea2.parse(line)
                        fix_type = int(msg.mode_fix_type) if msg.mode_fix_type else 1

                except Exception:
                    continue

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print("\n" + "-"*30)
            print(f"Temps: {timestamp}")
            print(f"Ciel: {sats_visible} satellites visibles")
            print(f"Utilisés: {num_sats} satellites | Fix: {fix_type}D")

            if lat is not None and lon is not None:
                lat_r, lon_r = round(lat, 6), round(lon, 6)
                print(f"Position: {lat_r}, {lon_r}")

                if (lat_r != last_lat) or (lon_r != last_lon):
                    append_to_csv(timestamp, lat_r, lon_r, num_sats, fix_type)
                    print(f"Enregistré dans {CSV_FILE}")
                    last_lat, last_lon = lat_r, lon_r
                else:
                    print("Position stable (pas d'écriture)")
            else:
                if sats_visible > 0:
                    print(f"Position: Signal trop faible ({sats_visible} vus). Patientez...")
                else:
                    print("Position: Recherche de signal... (Vérifiez l'antenne)")

    except Exception as e:
        print(f"Erreur critique: {e}")
    finally:
        if 'ser' in locals():
            ser.close()

if __name__ == "__main__":
    run_gps_analysis()
