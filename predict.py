"""
Predict using trained model
"""
import os

import click
from tqdm import tqdm
import cv2
import torch
import numpy as np

from utils import image_iterator, imwrite, get_image_quantity, get_image_source_type


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

    with tqdm(total=get_image_quantity(source),
              desc=f'Predicting {get_image_source_type(source)} \"{source}\"') as _tqdm:

        for img_info in image_iterator(source):
            # Image
            if len(img_info) == 2:
                img_path, img = img_info
                _tqdm.set_postfix_str(f'image: \"{img_path}\".')
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None

                # Predict
                label = int(inference(model, to_batch([preprocess(img)])))

                # Draw and save
                text, font_face, font_scale, thickness = str(label), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2
                (text_width, text_height), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
                cv2.putText(img, text, (0, text_height), font_face, font_scale, (255, 255, 0), thickness)
                img_out_path = os.path.join(out_dir, os.path.split(img_path)[1])
                imwrite(img_out_path, img)

            # Video
            else:
                video_path, frame, i, fps = img_info
                _tqdm.set_postfix_str(f'video: \"{video_path}\", frame index: {i}.')
                if video_path != last_video_path:
                    if video_writer is not None:
                        video_writer.release()
                    video_out_path = os.path.join(out_dir, os.path.splitext(os.path.split(video_path)[1])[0] + '.avi')
                    video_writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter.fourcc(*'XVID'),
                                                   fps, (frame.shape[1], frame.shape[0]), True)
                    last_video_path = video_path

                # Predict
                label = int(inference(model, to_batch([preprocess(frame)])))

                # Draw and write
                text, font_face, font_scale, thickness = str(label), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2
                (text_width, text_height), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
                cv2.putText(frame, text, (0, text_height), font_face, font_scale, (255, 255, 0), thickness)
                video_writer.write(frame)

            _tqdm.update()

        _tqdm.set_postfix_str('done!')

    if video_writer is not None:
        video_writer.release()


if __name__ == '__main__':
    predict()
