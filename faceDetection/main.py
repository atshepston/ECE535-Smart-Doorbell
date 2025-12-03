import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from TFLiteFaceDetector import UltraLightFaceDetecion

# from sklearn.datasets import fetch_lfw_people

# TODO: Load dataset to train model


# Can use a KNN model to do comparisons

# Need detector
# then classifier
# Use openCV for taking images and stuff like that
# Don't need to train a model myself
# Convert Facenet to tflite and use that


faceDetectorModelLink = "https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master/tflite/pretrained/version-slim-320_without_postprocessing.tflite"


MODEL_PATH = "./models/version-slim-320_without_postprocessing.tflite"


def main():
    # test = fetch_lfw_people(min_faces_per_person=60)
    # ds, info = tfds.load("lfw", split="train", with_info=True)
    # ds = ds.take(1)
    # # for image, label in ds:
    # #     print(image.shape, label)
    # print(info.features)
    # for i in tfds.as_numpy(ds):
    #     print(i["label"])
    # # for image, label in ds:
    # #     print(label["label"])
    # fig = tfds.show_examples(ds, info)
    # plt.show()
    # fig.show()
    fd = UltraLightFaceDetecion(
        MODEL_PATH,
        input_size=(320, 240),
        conf_threshold=0.6,
        center_variance=0.1,
        size_variance=0.2,
        nms_max_output_size=200,
        nms_iou_threshold=0.3,
    )
    # interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    # interpreter.allocate_tensors()
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    # input_shape = input_details[0]["shape"]
    # input_dtype = input_details[0]["dtype"]
    #
    # print(f"Model successfully loaded from: {MODEL_PATH}")
    # print(f"Input Shape: {input_shape}, Data Type: {input_dtype}")
    # print(f"Number of Output Tensors: {len(output_details)}")


if __name__ == "__main__":
    main()
