import os
from shutil import rmtree

from image_processing.filters import *
from video_processing.video_reader import VideoReader

rmtree("out", ignore_errors=True)

vr = VideoReader("resources/shortFlight.mp4")
frames = vr.read_all_frames()

# SSIM sequence
os.makedirs("out/ssim")
for image in filter_by_ssim(frames, 0.6):
    file_path = os.path.join("out", "ssim", str(image.index_number) + '.jpg')
    cv2.imwrite(file_path, image.frame)

# ahash sequence
os.makedirs("out/ahash")
for image in filter_by_ahash(frames, 15):
    file_path = os.path.join("out", "ahash", str(image.index_number) + '.jpg')
    cv2.imwrite(file_path, image.frame)

# dhash sequence
os.makedirs("out/dhash")
for image in filter_by_dhash(frames, 15):
    file_path = os.path.join("out", "dhash", str(image.index_number) + '.jpg')
    cv2.imwrite(file_path, image.frame)
