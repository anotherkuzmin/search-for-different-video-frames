from typing import Iterable, List

import cv2

from image_processing.algorithms import ssim, hamming_distance, ahash, dhash
from video_processing.video_reader import FrameHolder


def filter_by_ssim(images: Iterable[FrameHolder], threshold=0.7) -> List[FrameHolder]:
    current = None
    filtered_list = []
    for item in images:
        if current is None:
            current = item
            filtered_list.append(item)
            continue

        if ssim(cv2.resize(cv2.cvtColor(current.frame, cv2.COLOR_BGR2GRAY), (32, 32)),
                cv2.resize(cv2.cvtColor(item.frame, cv2.COLOR_BGR2GRAY), (32, 32))) < threshold:
            current = item
            filtered_list.append(item)

    return filtered_list


def filter_by_ahash(images: Iterable[FrameHolder], threshold=10) -> List[FrameHolder]:
    current = None
    filtered_list = []
    for item in images:
        if current is None:
            current = item
            filtered_list.append(item)
            continue

        if hamming_distance(ahash(current.frame), ahash(item.frame)) > threshold:
            current = item
            filtered_list.append(item)

    return filtered_list


def filter_by_dhash(images: Iterable[FrameHolder], threshold=10) -> List[FrameHolder]:
    current = None
    filtered_list = []
    for item in images:
        if current is None:
            current = item
            filtered_list.append(item)
            continue

        if hamming_distance(dhash(current.frame), dhash(item.frame)) > threshold:
            current = item
            filtered_list.append(item)

    return filtered_list
