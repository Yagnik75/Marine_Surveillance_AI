import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from pushbullet import Pushbullet
from geopy.geocoders import Nominatim
import os

# → Pushbullet setup
PUSHBULLET_API_KEY = "api_key"  # replace with your token
pb = Pushbullet(PUSHBULLET_API_KEY)

# → Reverse geocoding setup
geolocator = Nominatim(user_agent="marine_oil_spill_app")

# → Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# → Load ResNet18 model
model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())

model.load_state_dict(torch.load("bestoilspilldetectmodel.pt", map_location=device))
model = model.to(device)
model.eval()
print("✅ Model loaded.")

# → Image preprocess transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# → GUI: choose TIFF image
Tk().withdraw()
image_path = filedialog.askopenfilename(
    filetypes=[("TIFF files", "*.tif *.tiff")], title="Select a .tif satellite image"
)

if image_path:
    print(f"\nSelected image: {image_path}")

    # Load the TIFF using rasterio
    with rasterio.open(image_path) as src:
        image_array = src.read()
        image = reshape_as_image(image_array)  # HxWxC
        center_row, center_col = image.shape[0] // 2, image.shape[1] // 2
        lon, lat = src.xy(center_row, center_col)
        lat, lon = round(lat, 5), round(lon, 5)

    # → Reverse geocode for location
    try:
        location = geolocator.reverse(f"{lat}, {lon}", language="en", timeout=10)
        address = location.raw.get("address", {})
        city = address.get("city") or address.get("state") or ""
        country = address.get("country") or ""
        location_name = ", ".join(filter(None, [city, country]))
    except Exception as e:
        location_name = ""
        print(f"[WARNING] Reverse geocode failed: {e}")

    # Predict oil spill
    image_pil = Image.fromarray(image).convert("RGB")
    input_tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(input_tensor).item()
        is_oil_spill = prob > 0.5

    label = "Detected OIL SPILL" if is_oil_spill else "No Oil Spill Detected"
    print(f"\n {label} at Latitude {lat}, Longitude {lon}")
    if location_name:
        print(f"Location: {location_name}")

    # → Pushbullet alert
    if is_oil_spill:
        msg = f"Oil spill detected at ({lat}, {lon})\n{location_name}"
        print(f"Sending alert: {msg}")
        pb.push_note("Marine Surveillance Alert", msg)

    # Display the image
    plt.imshow(image_pil)
    plt.title(f"{label}\n{location_name}")
    plt.axis("off")
    plt.show()

else:
    print("No image selected.")
