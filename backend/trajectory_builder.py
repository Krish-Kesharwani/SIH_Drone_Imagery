import os
from services.pose_estimator import estimate_pose


def build_trajectory(frame_folder):

    frames = sorted([
        f for f in os.listdir(frame_folder)
        if f.endswith(".jpg")
    ])

    poses = []

    for i in range(len(frames) - 1):

        img1 = os.path.join(
            frame_folder,
            frames[i]
        )

        img2 = os.path.join(
            frame_folder,
            frames[i + 1]
        )

        try:

            R, t = estimate_pose(
                img1,
                img2
            )

            poses.append({
                "frame1": frames[i],
                "frame2": frames[i + 1]
            })

        except Exception as e:

            print(
                f"Skipped pair {i}: {e}"
            )

    return poses