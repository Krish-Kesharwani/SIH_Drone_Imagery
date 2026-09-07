import cv2
import os
import numpy as np

def match_features(image1_path, image2_path):

    # Create outputs folder if not exists
    os.makedirs("outputs", exist_ok=True)

    cv2.imwrite(
        "outputs/matches.jpg",
        match_img
    )
    # Read images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Safety check
    if img1 is None:
        raise Exception(f"Could not read image: {image1_path}")

    if img2 is None:
        raise Exception(f"Could not read image: {image2_path}")

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Brute Force matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw best 100 matches
    matched_image = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:100],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Save visualization
    output_path = "outputs/matches.jpg"

    cv2.imwrite(
        output_path,
        matched_image
    )

    return {
        "match_count": len(matches),
        "output_image": output_path
    }


def get_matched_points(
    img1_path,
    img2_path
):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2GRAY
    )

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = sift.detectAndCompute(
        gray2,
        None
    )

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    match_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good[:100],
        None,
        flags=2
    )

    cv2.imwrite(
        "outputs/matches.jpg",
        match_img
    )
    pts1 = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in good
        ]
    )

    pts2 = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in good
        ]
    )

    return pts1, pts2