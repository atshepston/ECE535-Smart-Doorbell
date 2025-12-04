import time
from datetime import datetime
from pathlib import Path

from libcamera import Transform
from picamera2 import Picamera2, Preview


def takePicture():
    current_dir = Path.cwd()
    image_dir = current_dir / "faceDetection" / "images"
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"

    picam = Picamera2()

    config = picam.create_preview_configuration(
        transform=Transform(vflip=True, hflip=True)
    )
    config = picam.create_preview_configuration(
        transform=Transform(vflip=True, hflip=True)
    )
    picam.configure(config)

    # picam.start_preview(Preview.DRM)

    picam.start()
    picam.capture_file(image_dir / filename)

    picam.close()


if __name__ == "__main__":
    takePicture()
