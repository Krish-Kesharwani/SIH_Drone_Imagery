import cv2
import numpy as np


def estimate_pose(image1_path, image2_path):

    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    )

    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    )

    h, w = gray1.shape

    focal = w

    K = np.array([
        [focal, 0, w/2],
        [0, focal, h/2],
        [0, 0, 1]
    ])

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    pts1 = pts1[
        mask.ravel() == 1
    ]

    pts2 = pts2[
        mask.ravel() == 1
    ]

    _, R, t, mask = cv2.recoverPose(
        E,
        pts1,
        pts2,
        K
    )

    np.save(
        "rotation.npy",
        R
    )

    np.save(
        "translation.npy",
        t
    )

    return R, t