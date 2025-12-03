import argparse
import os
import time

import cv2

from TFLiteFaceDetector import UltraLightFaceDetecion

parser = argparse.ArgumentParser(description="TFLite Face Detector")

parser.add_argument(
    "--net_type",
    default="RFB",
    type=str,
    help="The network architecture ,optional: RFB (higher precision) or slim (faster)",
)
parser.add_argument("--img_path", type=str, help="Image path for inference")
parser.add_argument("--video_path", type=str, help="Video path for inference")
parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs",
    help="Directory where cropped face images will be saved",
)

args = parser.parse_args()


def _save_crops(img, boxes, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)

    h, w = img.shape[:2]
    saved = 0

    for idx, result in enumerate(boxes.astype(int)):
        x1, y1, x2, y2 = result

        # clamp to image bounds
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]
        resized = cv2.resize(crop, (100, 100))

        filename = f"{prefix}_face{idx:03d}.png"
        cv2.imwrite(os.path.join(output_dir, filename), resized)
        saved += 1

    return saved


def image_inference(image_path, model_path, output_dir):
    fd = UltraLightFaceDetecion(model_path, conf_threshold=0.6, nms_max_output_size=100)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")

    boxes, scores = fd.inference(img)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    target_dir = os.path.join(output_dir, base_name)
    saved = _save_crops(img, boxes, target_dir, prefix=base_name)
    print(f"Saved {saved} face crops to {target_dir}")


def video_inference(video, model_path, output_dir):
    fd = UltraLightFaceDetecion(model_path, conf_threshold=0.88)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video}")

    base_name = os.path.splitext(os.path.basename(video))[0]
    target_dir = os.path.join(output_dir, base_name)

    frame_idx = 0
    total_saved = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        start_time = time.perf_counter()
        boxes, scores = fd.inference(frame)
        print(time.perf_counter() - start_time)

        prefix = f"{base_name}_frame{frame_idx:05d}"
        total_saved += _save_crops(frame, boxes, target_dir, prefix)
        frame_idx += 1

    cap.release()
    print(f"Saved {total_saved} face crops to {target_dir}")


if __name__ == "__main__":
    filepath = f"./models/version-slim-320_without_postprocessing.tflite"

    if args.img_path:
        image_inference(args.img_path, filepath, args.output_dir)
    elif args.video_path:
        video_inference(args.video_path, filepath, args.output_dir)
    else:
        print("--img_path or --video_path must be filled")
