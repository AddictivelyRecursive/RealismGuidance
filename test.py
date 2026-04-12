from pathlib import Path
import cv2
from modules.masking.generator import LandmarkMaskGenerator

src = Path("data/source/27003.jpg")
print("exists:", src.exists())

gen = LandmarkMaskGenerator(
    predictor_path="checkpoints/shape_predictor_68_face_landmarks.dat"
)

print("STEP 1: load image")
img = gen.load_image_bgr(src)
print("loaded:", type(img), img.shape, img.dtype)

print("STEP 2: convert gray")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("gray:", gray.shape, gray.dtype)

print("STEP 3: detector")
faces = gen.detector(gray, 0)
print("faces@0:", len(faces))

print("STEP 4: detector level 1")
faces = gen.detector(gray, 1)
print("faces@1:", len(faces))

print("STEP 5: detector level 2")
faces = gen.detector(gray, 2)
print("faces@2:", len(faces))

if len(faces) == 0:
    raise RuntimeError("No faces detected")

face = max(faces, key=lambda r: r.width() * r.height())
print("face rect:", face.left(), face.top(), face.right(), face.bottom())

print("STEP 6: predictor on gray")
shape = gen.landmark_predictor(gray, face)
print("predictor ok")

print("STEP 7: crop")
cropped = gen.crop_around_face(img, face)
print("cropped:", cropped.shape)

print("STEP 8: second detect")
gray2 = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
faces2 = gen.detector(gray2, 0)
print("faces2:", len(faces2))