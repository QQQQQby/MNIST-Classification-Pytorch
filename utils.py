"""
Utils
"""

import os
from typing import Union

import numpy as np
import cv2


def draw_label(img: np.ndarray, label: int,
               font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.5, color=(255, 255, 0), thickness=2):
    label = str(label)
    (text_width, text_height), _ = cv2.getTextSize(label, font_face, font_scale, thickness)
    cv2.putText(img, label, (0, text_height), font_face, font_scale, color, thickness)


def shuffle_arrays_in_unison(*arrays):
    """Shuffle multiple arrays in unison"""
    if not arrays:
        return
    rng_state = np.random.get_state()
    np.random.shuffle(arrays[0])
    for i in range(1, len(arrays)):
        np.random.set_state(rng_state)
        np.random.shuffle(arrays[i])


IMAGE_FORMATS = {'.png', '.jpeg', '.jpg', '.bmp', '.tif', '.tiff'}
VIDEO_FORMATS = {'.mp4', '.mkv', '.avi', '.flv', '.rmvb'}


def imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img):
    return cv2.imencode(os.path.splitext(path)[1], img)[1].tofile(path)


def video_frame_info_generator(source: Union[str, int]):
    capture = cv2.VideoCapture(source)
    fps = capture.get(cv2.CAP_PROP_FPS)
    i = 0
    while capture.isOpened():
        _, frame = capture.read()
        if frame is None:
            break
        yield source, frame, i, fps
        i += 1


def image_generator(source: Union[str, int]):
    # Camera
    if isinstance(source, int):
        for res in video_frame_info_generator(source):
            yield res

    # File
    elif os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        abs_path = os.path.abspath(source)
        if ext in IMAGE_FORMATS:
            # Image
            yield abs_path, imread(abs_path)

        elif ext in VIDEO_FORMATS:
            # Video
            for res in video_frame_info_generator(abs_path):
                yield res

        else:
            raise NotImplementedError('Unsupported file format!')

    # Directory
    elif os.path.isdir(source):
        for filename in os.listdir(source):
            abs_path = os.path.abspath(os.path.join(source, filename))

            if not os.path.isfile(abs_path):
                continue

            ext = os.path.splitext(abs_path)[1].lower()
            if ext in IMAGE_FORMATS:
                # Image
                yield abs_path, imread(abs_path)

            elif ext in VIDEO_FORMATS:
                # Video
                for res in video_frame_info_generator(abs_path):
                    yield res

    else:
        raise FileNotFoundError


def get_image_source_type(source: Union[str, int]):
    # Camera
    if isinstance(source, int):
        return 'camera'

    # File
    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in IMAGE_FORMATS:
            # Image
            return 'image'

        if ext in VIDEO_FORMATS:
            # Video
            return 'video'

        raise NotImplementedError('Unsupported file format!')

    # Directory
    if os.path.isdir(source):
        return 'directory'

    raise FileNotFoundError


def get_image_quantity(source: Union[str, int]):
    # Camera
    if isinstance(source, int):
        return None

    # File
    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in IMAGE_FORMATS:
            # Image
            return 1

        if ext in VIDEO_FORMATS:
            # Video
            capture = cv2.VideoCapture(source)
            return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        raise NotImplementedError('Unsupported file format!')

    # Directory
    if os.path.isdir(source):
        total = 0
        for filename in os.listdir(source):
            abs_path = os.path.abspath(os.path.join(source, filename))

            if not os.path.isfile(abs_path):
                continue

            ext = os.path.splitext(abs_path)[1].lower()
            if ext in IMAGE_FORMATS:
                # Image
                total += 1

            elif ext in VIDEO_FORMATS:
                # Video
                capture = cv2.VideoCapture(abs_path)
                total += int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return total

    else:
        raise FileNotFoundError
