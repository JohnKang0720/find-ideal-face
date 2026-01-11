import cv2 as cv

def rect_contains(rect, point):
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

def get_triangles(landmarks, w, h):
    rect = (0, 0, w, h)
    subdiv = cv.Subdiv2D(rect)

    points = [(int(x*w), int(y*h)) for x, y, _ in landmarks]
    for p in points:
        subdiv.insert(p)

    triangle_list = subdiv.getTriangleList()
    pt_to_idx = {p: i for i, p in enumerate(points)}
    triangles = []

    for t in triangle_list:
        pts = [(int(t[i]), int(t[i+1])) for i in range(0, 6, 2)]
        if all(rect_contains(rect, p) for p in pts):
            try:
                triangles.append(tuple(pt_to_idx[p] for p in pts))
            except KeyError:
                pass
    return triangles