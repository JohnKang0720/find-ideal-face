
import mediapipe as mp
from mediapipe.tasks.python import vision

model_path = "./model/face_landmarker.task"

def detect_landmarks(img):
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE
    )
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        result = landmarker.detect(mp_image)
        return result.face_landmarks[0]

