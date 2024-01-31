# coding: utf-8
"""
Load a model and predict output.
"""
import os

import cv2
import torch
import numpy as np

# model_path = './output/resnet1/epoch_16.pt'
# model_path = './output/1/epoch_99.pt'
model_path = 'best.pt'
video_path = 'data/VID_20200526_170846.mp4'


def load_model(path):
    return torch.load(path)


def output(model, image):
    image = np.reshape(image, (1, 28, 28))
    image = torch.from_numpy(image).float().cuda()
    return int(model(image).softmax(1).argmax(1))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)

    model = load_model(model_path)
    model.eval()

    capture = cv2.VideoCapture(video_path)
    # capture = cv2.VideoCapture(0)

    cv2.namedWindow("gray", 0)
    cv2.resizeWindow("gray", 280, 280)
    cv2.namedWindow("frame", 0)
    cv2.resizeWindow("frame", 800, 600)
    while capture.isOpened():
        ret, frame = capture.read()
        if frame is None:
            break
        # Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize
        gray = gray[gray.shape[0] // 6:gray.shape[0] // 6 * 5, gray.shape[1] // 6:gray.shape[1] // 6 * 5]
        gray = cv2.resize(gray, (28, 28))

        # Binarize
        gray = 255 - gray
        cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.threshold(gray, 128, 255, 0, gray)

        # Predict
        label = output(model, gray)

        # Draw
        cv2.putText(frame, str(label), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 10)
        cv2.imshow("frame", frame)
        cv2.imshow("gray", gray)
        cv2.waitKey(1)
