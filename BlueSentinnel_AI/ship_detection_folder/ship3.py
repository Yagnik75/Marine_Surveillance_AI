import cv2
import rasterio
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import os
from tkinter import Tk, filedialog
from pushbullet import Pushbullet

# 🔐 Pushbullet API Key
PUSHBULLET_API_KEY = "api_key"
pb = Pushbullet(PUSHBULLET_API_KEY)

# 🚫 Define restricted zones
ZONES = {
    "Great Barrier Reef Marine Park": [
        (-18.2861, 147.7000),
        (-18.2861, 148.0000),
        (-18.5861, 148.0000),
        (-18.5861, 147.7000),
    ],
    "Diego Garcia Military Exclusion Zone": [
        (-7.3133, 72.4118),
        (-7.3133, 72.6118),
        (-7.5133, 72.6118),
        (-7.5133, 72.4118),
    ],
    "Philippines Exclusive Economic Zone": [
        (12.0, 122.0),
        (12.0, 122.5),
        (12.5, 122.5),
        (12.5, 122.0),
    ],
}

ZONE_POLYGONS = {name: Polygon(coords) for name, coords in ZONES.items()}

# 🧠 Load YOLOv8 model
model = YOLO("best.pt")


def check_zone(lat, lon):
    point = Point(lat, lon)
    for zone_name, polygon in ZONE_POLYGONS.items():
        if polygon.contains(point):
            return zone_name
    return None


def read_tif_image(tif_path):
    with rasterio.open(tif_path) as dataset:
        img_array = dataset.read([1, 2, 3])  # RGB bands
        img_array = np.transpose(img_array, (1, 2, 0))  # Convert to HWC
        transform = dataset.transform
    bgr_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return bgr_image, transform


def send_alert(zone_name, lat, lon):
    message = f" Illegal ship detected in {zone_name} at ({lat:.4f}, {lon:.4f})"
    print(f"📩 Sending alert: {message}")
    pb.push_note("Marine Surveillance Alert", message)


def detect_illegal_ships(image, transform):
    results = model(image)[0]
    illegal_count = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        lon, lat = rasterio.transform.xy(transform, cy, cx)
        zone_name = check_zone(lat, lon)

        if zone_name:
            illegal_count += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = f"Illegal ship: {zone_name}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            text_y = 30
            cv2.putText(
                image,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            print(f" Ship in restricted zone: {zone_name} at ({lat:.4f}, {lon:.4f})")
            send_alert(zone_name, lat, lon)

    print(f"\n Total illegal ships detected: {illegal_count}")

    # Display the image without saving
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Illegal Ship Detection Result")
    plt.show()


def select_and_run():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        filetypes=[("GeoTIFF files", "*.tif *.tiff")]
    )
    if not file_path:
        print("No file selected.")
        return

    image_array, transform = read_tif_image(file_path)
    detect_illegal_ships(image_array, transform)


if __name__ == "__main__":
    select_and_run()
