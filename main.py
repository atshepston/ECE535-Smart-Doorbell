import argparse
from pathlib import Path

from faceDetection.inference_test import image_inference
from faceDetection.takePicture import takePicture
from FCSN.run_FNSN import run_FNSN

current_dir = Path.cwd()
image_dir = current_dir / "faceDetection" / "images"
modelFilepath = "./faceDetection/models/version-slim-320_without_postprocessing.tflite"
output_dir = current_dir / "faceDetection" / "outputs"


def main(anchor_path: Path):
    image_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = takePicture()
    cropped_images = image_inference(image_path, modelFilepath, output_dir)
    results = []
    for cropped_path in cropped_images:
        results.append(run_FNSN(str(anchor_path), str(cropped_path)))
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture a snapshot and compare it to an anchor face image."
    )
    parser.add_argument(
        "--anchor",
        help="Path to the anchor image used for comparison.",
    )
    args = parser.parse_args()
    anchor_path = Path(args.anchor)
    main(anchor_path)
