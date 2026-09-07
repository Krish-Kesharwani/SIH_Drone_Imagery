import numpy as np
import cv2


def triangulate_points(
    pts1,
    pts2,
    R,
    t
):

    if len(t.shape) == 1:
        t = t.reshape(3, 1)

    P1 = np.hstack(
        (
            np.eye(3),
            np.zeros((3, 1))
        )
    )

    P2 = np.hstack(
        (
            R,
            t
        )
    )

    points_4d = cv2.triangulatePoints(
        P1,
        P2,
        pts1.T,
        pts2.T
    )

    points_3d = (
        points_4d[:3]
        / points_4d[3]
    )

    return points_3d.T