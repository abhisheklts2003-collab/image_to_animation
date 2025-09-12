import os
import uvicorn
import aiohttp
import asyncio
import shutil
import imageio
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from animate import animate
from logger import Logger, Visualizer
from frames_dataset import PairedDataset
from sync_batchnorm import DataParallelWithCallback

app = FastAPI()

# Paths
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ✅ helper function (file download)
async def fetch_file(url: str, save_path: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                with open(save_path, 'wb') as f:
                    f.write(await resp.read())
            else:
                raise Exception(f"Failed to download {url}, status {resp.status}")


@app.post("/animate/")
async def generate_video(
    source_url: str = Form(...),
    driver_url: str = Form(...)
):
    # download files
    source_path = os.path.join(UPLOAD_DIR, "source.png")
    driver_path = os.path.join(UPLOAD_DIR, "driver.mp4")

    await fetch_file(source_url, source_path)
    await fetch_file(driver_url, driver_path)

    # load pretrained models (adjust paths as per your setup)
    from demo import load_checkpoints
    generator, kp_detector = load_checkpoints(
        config_path="config/vox-256.yaml",
        checkpoint_path="checkpoints/vox-cpk.pth.tar"
    )

    # call animate function
    dataset = {"source": source_path, "driving": driver_path}
    result_video_path = os.path.join(RESULT_DIR, "result.mp4")

    animate(
        config={"animate_params": {"num_pairs": 1, "normalization_params": {
            "use_relative_movement": True,
            "adapt_movement_scale": True,
            "use_relative_jacobian": False
        }},
        "visualizer_params": {"draw_border": True, "colormap": "gist_rainbow"}},
        generator=generator,
        kp_detector=kp_detector,
        checkpoint="checkpoints/vox-cpk.pth.tar",
        log_dir=RESULT_DIR,
        dataset=dataset
    )

    return {"result_url": f"/download/{os.path.basename(result_video_path)}"}


# ✅ endpoint का नाम safe रखा
@app.get("/download/{filename}")
async def download_result(filename: str):
    file_path = os.path.join(RESULT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type="video/mp4")
    return {"error": "File not found"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
