# coding: utf-8
"""
Load a model and predict output.
"""
import cv2
import torch
import numpy as np

model_path = './output/1000_0.01_dropout0.7/epoch_17.pd'
video_path = 'data/VID_20200526_170846.mp4'


def load_model(path):
    return torch.load(model_path)


def c(image):
    return 255 * (image - image.min()) / (image.max() - image.min())


def output(model, image):
    image = np.reshape(image, (1, 28, 28))
    image = torch.from_numpy(image).float().cuda()
    return int(model(image).softmax(1).argmax(1))


# def RGB2gray(R, G, B):
#     return (R * 299 + G * 587 + B * 114 + 500) / 1000


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model = load_model(model_path)

    capture = cv2.VideoCapture(video_path)

    while capture.isOpened():
        ret, frame = capture.read()
        """调整角度"""
        # frame = cv2.rotate(frame, 0)
        """将RGB转换成灰度"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        """调整分辨率至28 * 28"""
        gray = cv2.resize(gray, (28, 28))
        """将灰度倒置"""
        gray = 255 - gray
        """调整对比度"""
        # gray = c(gray)
        # cv2.normalize(gray, np.zeros_like(gray), 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        """预测标签"""
        label = output(model, gray)
        """将标签放在图片里"""
        cv2.putText(frame, str(label), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 10)

        """将原始图片显示在窗口中"""
        cv2.namedWindow("frame", 0)
        cv2.resizeWindow("frame", 1024, 1024)
        # cv2.imshow("frame", gray)
        cv2.imshow("frame", frame)
        """将灰度图片显示在另一个窗口中"""
        cv2.namedWindow("gray", 0)
        cv2.resizeWindow("gray", 280, 280)
        cv2.imshow("gray", gray)
        """设置每帧时间"""
        cv2.waitKey(50)
