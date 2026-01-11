from py_compile import main
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python import vision
from utils.convert import convert_landmarks_mediapipe_to_dlib
from utils.landmark_detection import detect_landmarks
from utils.morph import morph_triangle, create_face_mask
from utils.delaunay_tri import get_triangles

def run_morph_loop(W, H, image1, image2, m1, m2):
    # show a few intermediate alphas; press ESC to exit early
    for alpha in np.linspace(0.0, 1.0, 100):
        m_avg = (1 - alpha) * m1 + alpha * m2
        triangles = get_triangles(m_avg, W, H)

        morphed_face = np.zeros_like(image1, dtype=np.float32)
        for tri in triangles:
            t1 = [(m1[i][0] * W, m1[i][1] * H) for i in tri]
            t2 = [(m2[i][0] * W, m2[i][1] * H) for i in tri]
            t = [(m_avg[i][0] * W, m_avg[i][1] * H) for i in tri]
            morph_triangle(image1, image2, morphed_face, t1, t2, t, alpha)

        # smooth and prepare mask/background
        mask = create_face_mask(m_avg, H, W)
        mask = cv.GaussianBlur(mask, (21, 21), 0)

        # blend backgrounds so the whole image transitions
        bg = ((1 - alpha) * image1 + alpha * image2).astype(np.uint8)

        hull_pts = np.array([(int(x*W), int(y*H)) for x,y,_ in m_avg])
        hull = cv.convexHull(hull_pts)
        center = tuple(np.mean(hull.reshape(-1, 2), axis=0).astype(int))

        result = cv.seamlessClone(
            morphed_face.astype(np.float32),
            bg,
            mask,
            center,
            cv.NORMAL_CLONE
        )

        cv.imshow("Face Morph", cv.cvtColor(result, cv.COLOR_RGB2BGR))
        k = cv.waitKey(300) & 0xFF
        if k == 27:  # ESC to exit early
            break

if __name__ == "__main__":
    # ---------------- LOAD IMAGES ----------------
    image1 = cv.imread("images/tsuyu.png")
    image2 = cv.imread("images/karina.png")

    image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
    image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)

    image1 = cv.resize(image1, (500, 500))
    image2 = cv.resize(image2, (500, 500))

    l1 = detect_landmarks(image1)
    l2 = detect_landmarks(image2)

    l1 = convert_landmarks_mediapipe_to_dlib(
        np.array([[p.x, p.y, p.z] for p in l1])
    )
    l2 = convert_landmarks_mediapipe_to_dlib(
        np.array([[p.x, p.y, p.z] for p in l2])
    )

    m1 = np.array(l1)
    m2 = np.array(l2)

    H, W = 500, 500

    run_morph_loop(W, H, image1, image2, m1, m2)
    
