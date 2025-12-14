from pathlib import Path

from faceDetection.inference_test import image_inference
from faceDetection.takePicture import takePicture
from FCSN.run_FNSN import run_FNSN

current_dir = Path.cwd()
image_dir = current_dir / "faceDetection" / "images"
modelFilepath = "./faceDetection/models/version-slim-320_without_postprocessing.tflite"
output_dir = current_dir / "faceDetection" / "outputs"


def main():
    image_path = takePicture()
    cropped_images = image_inference(image_path, modelFilepath, output_dir)
    anchor_path = "faceDetection/outputs/20251214_110402_450141_face000.png"
    anchor_path = "faceDetection/anchorImages/anchorAidan.png"
    results = []
    for cropped_path in cropped_images:
        results.append(run_FNSN(anchor_path, cropped_path))
    print(results)


if __name__ == "__main__":
    main()
