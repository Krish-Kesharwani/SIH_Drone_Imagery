from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from services.frame_extractor import extract_frames
from services.feature_matcher import match_features
from services.pose_estimator import estimate_pose
from services.triangulation import triangulate_points
from services.feature_matcher import get_matched_points
from services.pointcloud_writer import save_ply
import numpy as np
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend running"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "success",
        "filename": file.filename
    }

@app.post("/extract-frames")
async def extract_video_frames(filename: str):

    video_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    frame_folder = os.path.join(
        "frames",
        filename.split(".")[0]
    )

    count = extract_frames(
        video_path,
        frame_folder
    )

    return {
        "status": "success",
        "frames": count
    }

@app.post("/match-frames")
async def match_frames():

    frame_dir = "frames/earth"

    frames = sorted(os.listdir(frame_dir))

    img1 = os.path.join(frame_dir, frames[0])
    img2 = os.path.join(frame_dir, frames[30])

    result = match_features(
    img1,
    img2
    )

    return result

@app.post("/estimate-pose")
async def estimate_camera_pose():

    img1 = "frames/earth/frame_00000.jpg"
    img2 = "frames/earth/frame_00010.jpg"

    R, t = estimate_pose(
    img1,
    img2
    )

    return {
        "rotation": R.tolist(),
        "translation": t.tolist()
    }


@app.post("/triangulate")
async def triangulate():

    img1 = "frames/earth/frame_00000.jpg"
    img2 = "frames/earth/frame_00010.jpg"

    pts1, pts2 = get_matched_points(
        img1,
        img2
    )

    R, t = estimate_pose(
        img1,
        img2
    )

    points_3d = triangulate_points(
        pts1,
        pts2,
        R,
        t
    )

    return {
        "points_3d": len(points_3d)
    }

@app.post("/generate-pointcloud")
async def generate_pointcloud():

    img1 = "frames/earth/frame_00000.jpg"
    img2 = "frames/earth/frame_00010.jpg"

    pts1, pts2 = get_matched_points(
        img1,
        img2
    )

    R, t = estimate_pose(
        img1,
        img2
    )

    points_3d = triangulate_points(
        pts1,
        pts2,
        R,
        t
    )

    ply_file = save_ply(
        points_3d,
        "earth_cloud.ply"
    )

    return {
        "status": "success",
        "points": len(points_3d),
        "file": ply_file
    }