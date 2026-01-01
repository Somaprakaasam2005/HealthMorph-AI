import cv2
import numpy as np


def detect_and_prepare(bgr_image: np.ndarray):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        h, w = gray.shape
        x, y, bw, bh = 0, 0, w, h
    else:
        areas = [w * h for (_, _, w, h) in faces]
        idx = int(np.argmax(areas))
        x, y, bw, bh = faces[idx]

    x2, y2 = x + bw, y + bh
    x, y = max(0, x), max(0, y)
    x2, y2 = min(bgr_image.shape[1], x2), min(bgr_image.shape[0], y2)

    face_crop = bgr_image[y:y2, x:x2]
    face_resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
    bbox = (x, y, bw, bh)
    return face_resized, bbox
