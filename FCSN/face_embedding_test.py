from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# import tflite_runtime.interpreter as tflite

IMG_SIZE = 112

interpreter = tf.lite.Interpreter(model_path="./FCSN/MobileFaceNet_9925_9680.tflite")
# interpreter = tf.lite.Interpreter(model_path="./FCSN/siamese_embedding_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_face(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img = img_resized.astype("float32")
    # mimic MobileNetV2 preprocess_input: scale to [-1, 1]
    img = (img - 127.5) / 128.0
    img = np.expand_dims(img, axis=0)
    return img


def get_embedding(img_bgr):
    img = preprocess_face(img_bgr)
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()
    emb = interpreter.get_tensor(output_details[0]["index"])[0]
    # Should already be L2 normalized by the model
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def l2_distance(a, b):
    return np.linalg.norm(a - b)


def preprocess_colors(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)

    img = tf.image.resize(img, (100, 100))
    img = img / 255.0

    return img


current_dir = Path().cwd()
output_dir = current_dir / "faceDetection" / "outputs"
image_dir = current_dir / "faceDetection" / "images"

# latest = max(image_dir.glob("*.jpg"))
latest = max(output_dir.glob("*.png"))
anchorPath = Path("faceDetection/anchorImages/20251203_225434_749628_face000.png")
# anchorPath = Path("faceDetection/images/20251204_100222_479738.jpg")
# anchorPath = Path("faceDetection/anchorImages/20251204_100222_479738_face000.png")
# test_path = Path("faceDetection/images/20251204_100643_735812.jpg")
test_path = Path("faceDetection/anchorImages/Aaron_Peirsol_0001_face000.png")
# test_path = Path("faceDetection/anchorImages/20251204_100222_479738_face000.png")

anchor_img = cv2.imread(anchorPath)
test_img = cv2.imread(latest)

anchor_emb = get_embedding(anchor_img)
test_emb = get_embedding(test_img)

dist = l2_distance(anchor_emb, test_emb)
print("distance:", dist)

matching = False

if dist < 1.0:  # threshold to tune
    print("Same person")
    matching = True
else:
    print("Different person")


fig, axs = plt.subplots(1, 2)

axs[0].imshow(cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB))
axs[0].set_title("Anchor")
axs[0].axis("off")

axs[1].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
axs[1].set_title("Detected person")
axs[1].axis("off")

fig.suptitle(f"Matching: {matching}, distance: {dist}")

plt.savefig("output.png")
