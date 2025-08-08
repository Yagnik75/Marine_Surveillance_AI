import os
import requests
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pushbullet import Pushbullet
from geopy.geocoders import Nominatim
from rasterio.warp import transform_bounds
import pyproj

# ----------------------------------
# Pushbullet Setup
# ----------------------------------
PUSHBULLET_API_KEY = (
    "api_key"  # Replace with your Pushbullet API key
)
pb = Pushbullet(PUSHBULLET_API_KEY)


# ----------------------------------
# Step 1: Download sample Durban image (if not already present)
# ----------------------------------
def download_sample_image():
    url = "https://www.dropbox.com/s/vbhbykl86x0evkh/durban_20190424.tif?dl=1"
    filename = "durban_20190424.tif"

    if not os.path.exists(filename):
        print("[INFO] Downloading sample image...")
        response = requests.get(url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print("[INFO] Download complete.")
    else:
        print("[INFO] Sample image already exists.")
    return filename


# ----------------------------------
# Step 2: Read Sentinel-2 bands and apply detection
# ----------------------------------
def detect_marine_debris(image_path, threshold=1.0):
    with rasterio.open(image_path) as src:
        profile = src.profile
        bounds = src.bounds
        crs = src.crs
        bands = src.read()

    # Sentinel-2 band mapping (assuming image contains B2 to B8A)
    blue = bands[0].astype(float)  # B2
    green = bands[1].astype(float)  # B3
    red = bands[2].astype(float)  # B4
    nir = bands[6].astype(float)  # B8

    # Apply threshold to detect likely debris
    index = (red - blue) / (red + blue + 1e-6)
    debris_mask = (index > threshold).astype(np.uint8)

    print(f"[INFO] Debris detected in {np.sum(debris_mask)} pixels")

    # Reconstruct RGB image for display
    rgb_image = np.stack([red, green, blue], axis=-1)
    rgb_image = np.clip(rgb_image / 3000.0, 0, 1)  # Normalize for display

    latlon_bounds = None
    location_name = "Unknown"

    try:
        # Convert bounds to lat/lon
        latlon_bounds = transform_bounds(crs, "EPSG:4326", *bounds)
        lat_center = (latlon_bounds[1] + latlon_bounds[3]) / 2
        lon_center = (latlon_bounds[0] + latlon_bounds[2]) / 2
        print(f"[INFO] Image location:")
        print(f"       Latitude:  {latlon_bounds[1]:.6f} to {latlon_bounds[3]:.6f}")
        print(f"       Longitude: {latlon_bounds[0]:.6f} to {latlon_bounds[2]:.6f}")

        # Reverse geocode using Geopy
        geolocator = Nominatim(user_agent="marine_debris_detector")
        location = geolocator.reverse((lat_center, lon_center), timeout=10)
        if location:
            location_name = location.address
            print(f"[INFO] Approximate Location: {location_name}")
        else:
            print("[INFO] Location name not found.")
    except Exception as e:
        print("[WARNING] Coordinate processing error:", e)

    # Send Pushbullet Alert
    message = (
        f"🛑 Marine Debris Detected!\n"
        f"Pixels Detected: {np.sum(debris_mask)}\n"
        f"Lat: {latlon_bounds[1]:.6f} to {latlon_bounds[3]:.6f}\n"
        f"Lon: {latlon_bounds[0]:.6f} to {latlon_bounds[2]:.6f}\n"
        f"Location: {location_name}"
    )
    pb.push_note("Marine Debris Alert", message)

    return debris_mask, rgb_image, profile


# ----------------------------------
# Step 3: Visualize detection
# ----------------------------------
def visualize_detection(rgb_image, debris_mask):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].imshow(rgb_image)
    axs[0].set_title("Original RGB Image")
    axs[0].axis("off")

    axs[1].imshow(debris_mask, cmap="gray")
    axs[1].set_title("Detected Marine Debris (Thresholded)")
    axs[1].axis("off")
    axs[1].text(
        10,
        20,
        "White: Debris detected\nBlack: No debris",
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.5),
    )

    plt.tight_layout()
    plt.show()


# ----------------------------------
# Step 4: Save debris mask as GeoTIFF
# ----------------------------------
def save_mask(mask, profile, out_path="debris_mask.tif"):
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)
    print(f"[INFO] Debris mask saved to {out_path}")


# ----------------------------------
# Main
# ----------------------------------
def main():
    image_path = download_sample_image()
    debris_mask, rgb_image, profile = detect_marine_debris(image_path, threshold=0.1)
    visualize_detection(rgb_image, debris_mask)
    save_mask(debris_mask, profile)


if __name__ == "__main__":
    main()
