"""
Predict using trained model
"""
import os

import click
from tqdm import tqdm
import cv2
import torch
import numpy as np

from utils import image_generator, imwrite, get_image_quantity, get_image_source_type, draw_label


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
@click.option('--save/--no-save', default=True,
              help='Whether to save predicted images or videos.')
@click.option('--save-labels/--no-save-labels', default=False,
              help='Whether to save the label file.')
def predict(source, model_path, out_dir, save, save_labels):
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
    label_file = open(os.path.join(out_dir, 'labels.txt'), 'w', encoding='utf-8') if save_labels else None

    with tqdm(total=get_image_quantity(source),
              desc=f'Predicting {get_image_source_type(source)} \"{source}\"') as _tqdm:

        for img_info in image_generator(source):
            # Image
            if len(img_info) == 2:
                img_path, img = img_info
                _tqdm.set_postfix_str(f'image: \"{img_path}\".')

                if save and video_writer is not None:
                    video_writer.release()
                    video_writer = None

                # Predict
                label = int(inference(model, to_batch([preprocess(img)]))[0])

                # Draw and save
                if save:
                    draw_label(img, label)

                    img_out_path = os.path.join(out_dir, os.path.split(img_path)[1])
                    imwrite(img_out_path, img)

                # Save label
                if save_labels:
                    label_file.write(f'\"{img_path}\" {label}\n')

            # Video
            else:
                video_path, frame, i, fps = img_info
                _tqdm.set_postfix_str(f'video: \"{video_path}\", frame index: {i}.')

                if save and video_path != last_video_path:
                    if video_writer is not None:
                        video_writer.release()
                    video_out_path = str(os.path.join(
                        out_dir,
                        os.path.splitext(os.path.split(video_path)[1])[0] + '.avi'
                    ))
                    video_writer = cv2.VideoWriter(
                        video_out_path, cv2.VideoWriter.fourcc('X', 'V', 'I', 'D'),
                        fps, (frame.shape[1], frame.shape[0]), True
                    )
                    last_video_path = video_path

                # Predict
                label = int(inference(model, to_batch([preprocess(frame)]))[0])

                # Draw and write
                if save:
                    draw_label(frame, label)

                    video_writer.write(frame)

                # Save label
                if save_labels:
                    label_file.write(f'\"{video_path}\" {i} {label}\n')

            _tqdm.update()

        _tqdm.set_postfix_str('done!')

    if save and video_writer is not None:
        video_writer.release()

    if save_labels:
        label_file.close()


if __name__ == '__main__':
    predict()
