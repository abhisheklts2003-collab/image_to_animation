import imageio
import requests
import numpy as np
from io import BytesIO
from PIL import Image

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(img)

def load_video_from_url(url):
    response = requests.get(url, stream=True)
    video_path = "driving.mp4"
    with open(video_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    return imageio.mimread(video_path)
