from typing import List, Iterable

import cv2
import numpy as np


class FrameHolder:

    def __init__(self, frame: np.ndarray, index_number: int):
        self.frame = frame
        self.index_number = index_number


class VideoReader:

    def __init__(self, path_to_video_file: str):
        self._path_to_video = path_to_video_file

    def read_all_frames(self) -> List[FrameHolder]:
        return list(self.get_frames_one_by_one_generator())

    def get_frames_one_by_one_generator(self) -> Iterable[FrameHolder]:
        video = cv2.VideoCapture(self._path_to_video)
        index = 0
        while video.grab():
            ret, frame = video.retrieve()

            indexed_frame = FrameHolder(frame, index)
            index += 1
            yield indexed_frame

        video.release()
