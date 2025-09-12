import os
import uvicorn
import aiohttp
import imageio
import torch
import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from demo import load_checkpoints, make_animation
from skimage import img_as_ubyte
from skimage.io import imread
import cv2
import traceback

app = FastAPI()

# Paths
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Ngrok public URL (set manually in Colab)
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8000")

# ✅ Root endpoint
@app.get("/")
async def root():
    return {"message": "🚀 Animation API is running! Use POST /animate/ endpoint."}

# ✅ helper function (file download)
async def fetch_file(url: str, save_path: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                with open(save_path, 'wb') as f:
                    f.write(await resp.read())
            else:
                raise Exception(f"❌ Failed to download {url}, status {resp.status}")

# ✅ Animate endpoint
@app.post("/animate/")
async def generate_video(
    request: Request,
    source_url: str = Form(...),
    driver_url: str = Form(...)
):
    try:
        # Debug: incoming form data
        body = await request.form()
        print("📩 Incoming form data:", dict(body))

        # download files
        source_path = os.path.join(UPLOAD_DIR, "source.png")
        driver_path = os.path.join(UPLOAD_DIR, "driver.mp4")

        await fetch_file(source_url, source_path)
        await fetch_file(driver_url, driver_path)

        # load pretrained models
        generator, kp_detector = load_checkpoints(
            config_path="config/vox-256.yaml",
            checkpoint_path="checkpoints/vox-cpk.pth.tar"
        )

        # process input
        source_image = imread(source_path)
        driving_video = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                         for frame in imageio.get_reader(driver_path)]

        # generate animation
        predictions = make_animation(
            source_image, driving_video, generator, kp_detector,
            relative=True, adapt_movement_scale=True
        )

        # save video as output.mp4
        result_video_path = os.path.join(RESULT_DIR, "output.mp4")
        imageio.mimsave(result_video_path, [img_as_ubyte(frame) for frame in predictions], fps=30)

        # ✅ Return ngrok download URL
        download_url = f"{PUBLIC_URL}/download/{os.path.basename(result_video_path)}"
        return {"result_url": download_url}

    except Exception as e:
        # 🔴 Show error both in response & logs
        error_msg = str(e)
        tb = traceback.format_exc()
        print("❌ ERROR:", error_msg)
        print(tb)
        return JSONResponse(
            status_code=500,
            content={
                "error": error_msg,
                "traceback": tb
            }
        )

# ✅ Endpoint to download result video
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(RESULT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename, media_type="video/mp4")
    return JSONResponse(status_code=404, content={"error": "File not found"})
