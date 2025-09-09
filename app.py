from fastapi import FastAPI
from pydantic import BaseModel
from utils import load_image_from_url, load_video_from_url
import imageio
import os
import gdown  # new import

from demo import load_checkpoints, make_animation  # from FOMM repo

app = FastAPI()

# --------- Auto-Download Checkpoint ---------
os.makedirs("checkpoints", exist_ok=True)
checkpoint_path = "checkpoints/vox-cpk.pth.tar"
config_path = "config/vox-256.yaml"

if not os.path.exists(checkpoint_path):
    print("Downloading pretrained model from Google Drive...")
    # 👇 यहाँ अपना Google Drive file ID डालो
    url = "https://drive.google.com/uc?id=1JzEtdlW8T5qvs7X4jkYDcQJxQeUu7l3y"  
    gdown.download(url, checkpoint_path, quiet=False)
else:
    print("Checkpoint already exists.")


# --------- Load pretrained FOMM model ---------
generator, kp_detector = load_checkpoints(
    config_path=config_path,
    checkpoint_path=checkpoint_path
)

# --------- API Schema ---------
class MotionRequest(BaseModel):
    image_url: str
    driving_url: str

# --------- API Route ---------
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
