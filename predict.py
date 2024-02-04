"""
Predict using trained model
"""
import os

import click
import cv2
import torch
import numpy as np

from utils import image_iterator, imwrite


def predict_images(source, model):
    images = list(map(preprocess, source))
    images = to_batch(images)

    labels = model(images)

    print()


def predict_video(source, model):
    capture = cv2.VideoCapture(source)

    # cv2.namedWindow("gray", 0)
    # cv2.resizeWindow("gray", 280, 280)
    # cv2.namedWindow("frame", 0)
    # cv2.resizeWindow("frame", 800, 800)

    while capture.isOpened():
        ret, frame = capture.read()
        if frame is None:
            break

        img_gray = preprocess(frame)

        # Predict
        label = int(inference(model, np.expand_dims(img_gray, 0)))

        # Draw label
        cv2.putText(frame, str(label), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 10)

        # Show results
        # cv2.imshow("frame", frame)
        # cv2.imshow("gray", img_gray)
        # cv2.waitKey(1)


def preprocess(img, invert=True, norm=True, binarize=True):
    """
    Preprocess on a single image, and a grayscale image of size 28 x 28.
    """
    img_resized = cv2.resize(img, (28, 28))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    if invert:
        img_gray = 255 - img_gray
    if norm:
        cv2.normalize(img_gray, img_gray, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    if binarize:
        cv2.threshold(img_gray, 128, 255, 0, img_gray)
    return img_gray


def to_batch(images):
    return np.expand_dims(np.concatenate(list(map(lambda img: np.expand_dims(img, 0), images))), 1) if len(images) > 1 \
        else np.expand_dims(images[0], 0)


def inference(model, batch):
    batch = torch.tensor(batch, dtype=torch.float32)
    return model(batch).argmax(1).cpu().numpy()


@click.command()
@click.option('-s', '--source', type=str, required=True,
              help='Source of prediction, which can be a path to an image, a video or a directory.')
@click.option('-m', '--model-path', type=str, required=True,
              help='Path to the trained model.')
@click.option('-o', '--out_dir', type=str, default=None,
              help='Predicting output directory.')
def predict(source, model_path, out_dir):
    # Prepare output directory
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_root = './output/'
        out_dir = os.path.join(out_root, 'predict')
        if os.path.isdir(out_dir):
            i = 1
            while True:
                out_dir = os.path.join(out_root, f'predict_{i}')
                if not os.path.isdir(out_dir):
                    break
                i += 1
        os.makedirs(out_dir)
    print(f'Results will be saved in \"{out_dir}\"')

    # Set default device to CUDA if available
    if torch.cuda.is_available():
        torch.set_default_device('cuda')

    # Load the model
    model = torch.load(model_path)
    model.eval()

    # Predict
    if source.isdigit():
        source = int(source)
    last_video_path = ''
    video_writer = None
    for img_info in image_iterator(source):
        # Image
        if len(img_info) == 2:
            if video_writer is not None:
                video_writer.release()
                video_writer = None

            img_path, img = img_info
            img_out_path = os.path.join(out_dir, os.path.split(img_path)[1])

            # Predict, draw and save
            label = int(inference(model, to_batch([preprocess(img)])))
            text, font_face, font_scale, thickness = str(label), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2
            (text_width, text_height), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
            cv2.putText(img, text, (0, text_height), font_face, font_scale, (255, 255, 0), thickness)
            imwrite(img_out_path, img)

        # Video
        else:
            video_path, frame, i, fps = img_info
            if video_path != last_video_path:
                if video_writer is not None:
                    video_writer.release()
                video_out_path = os.path.join(out_dir, os.path.splitext(os.path.split(video_path)[1])[0] + '.avi')
                video_writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'XVID'),
                                               fps, (frame.shape[1], frame.shape[0]), True)
                last_video_path = video_path

            # Predict, draw and save
            label = int(inference(model, to_batch([preprocess(frame)])))
            text, font_face, font_scale, thickness = str(label), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2
            (text_width, text_height), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
            cv2.putText(frame, text, (0, text_height), font_face, font_scale, (255, 255, 0), thickness)
            video_writer.write(frame)

    if video_writer is not None:
        video_writer.release()


if __name__ == '__main__':
    predict()
