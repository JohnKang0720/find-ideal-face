import cv2 as cv
import numpy as np

def morph_triangle(img1, img2, out, t1, t2, t, alpha):
    r = cv.boundingRect(np.float32([t]))
    r1 = cv.boundingRect(np.float32([t1]))
    r2 = cv.boundingRect(np.float32([t2]))

    tRect = [(t[i][0] - r[0], t[i][1] - r[1]) for i in range(3)]
    t1Rect = [(t1[i][0] - r1[0], t1[i][1] - r1[1]) for i in range(3)]
    t2Rect = [(t2[i][0] - r2[0], t2[i][1] - r2[1]) for i in range(3)]

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)
    mask = cv.GaussianBlur(mask, (9,9), 0)

    src1_roi = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    src2_roi = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

    M1 = cv.getAffineTransform(np.float32(t1Rect), np.float32(tRect))
    M2 = cv.getAffineTransform(np.float32(t2Rect), np.float32(tRect))

    warp1 = cv.warpAffine(src1_roi, M1, (r[2], r[3]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    warp2 = cv.warpAffine(src2_roi, M2, (r[2], r[3]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)

    blended = (1 - alpha) * warp1.astype(np.float32) + alpha * warp2.astype(np.float32)

    out[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] *= (1 - mask)
    out[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] += blended * mask

def create_face_mask(landmarks, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([(int(x*w), int(y*h)) for x, y, _ in landmarks], np.int32)
    hull = cv.convexHull(pts)
    cv.fillConvexPoly(mask, hull, 255)
    return mask