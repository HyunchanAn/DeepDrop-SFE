
import requests
import os

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Contact_angle.svg/800px-Contact_angle.svg.png"
save_path = "demo/assets/contact_angle_ref.png"

# Ensure directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

try:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"Successfully downloaded to {save_path}")
except Exception as e:
    print(f"Failed to download: {e}")
