from fastapi import FastAPI
from pydantic import BaseModel
from utils import load_image_from_url, load_video_from_url
import imageio
import os

from demo import load_checkpoints, make_animation  # from FOMM repo

app = FastAPI()

# Load pretrained FOMM model (Colab me checkpoint pehle download hoga)
generator, kp_detector = load_checkpoints(
    config_path="config/vox-256.yaml",
    checkpoint_path="checkpoints/vox-cpk.pth.tar"
)

class MotionRequest(BaseModel):
    image_url: str
    driving_url: str

@app.post("/process")
def process_motion(data: MotionRequest):
    # Load inputs
    source_image = load_image_from_url(data.image_url)
    driving_video = load_video_from_url(data.driving_url)

    # Apply motion transfer
    predictions = make_animation(source_image, driving_video, generator, kp_detector)

    # Save output video
    output_path = "output.mp4"
    imageio.mimsave(output_path, [frame for frame in predictions], fps=30)

    return {"status": "success", "video_path": output_path}
