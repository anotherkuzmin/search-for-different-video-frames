import math

import cv2
import numpy as np


def covariation(matrix_1: np.ndarray, matrix_2: np.ndarray):
    width = matrix_1.shape[1]
    height = matrix_1.shape[0]

    mean_m1 = np.mean(matrix_1)
    mean_m2 = np.mean(matrix_2)

    sum = 0
    for y in range(height):
        for x in range(width):
            sum += (matrix_1[y][x] - mean_m1) * (matrix_2[y][x] - mean_m2)

    return sum / (width * height)


def ssim(matrix_1: np.ndarray, matrix_2: np.ndarray):
    """
    SSIM is used for measuring the similarity between two images.
    The resultant SSIM index is a decimal value between -1 and 1,
    and value 1 is only reachable in the case of two identical sets of data.

    :param matrix_1:
    :param matrix_2:
    :return:
    """
    mean_m1 = np.mean(matrix_1)
    mean_m2 = np.mean(matrix_2)
    var_m1 = np.var(matrix_1)
    var_m2 = np.var(matrix_2)
    return (
            (covariation(matrix_1, matrix_2) / (math.sqrt(var_m1) * math.sqrt(var_m2))) *

            ((2 * mean_m1 * mean_m2) / (mean_m1 ** 2 + mean_m2 ** 2)) *

            ((2 * math.sqrt(var_m1) * math.sqrt(var_m2)) / (var_m1 + var_m2))
    )


def ahash(image: np.ndarray, hash_size=8) -> int:
    """
    Average hashing,
    for each of the pixels output 1 if the pixel is bigger or equal to the average and 0 otherwise.

    :param image:
    :param hash_size:
    :return:
    """
    resized = cv2.resize(image, (hash_size, hash_size))
    mean = np.mean(resized)
    return sum([2 ** i for (i, v) in enumerate(resized.flatten()) if v >= mean])


def dhash(image: np.ndarray, hash_size=8) -> int:
    """
    Difference hashing - gradient hash,
    calculate the difference for each of the pixel and compares the difference with the average differences.

    :param image:
    :param hash_size:
    :return:
    """
    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(image, (hash_size + 1, hash_size))

    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def hamming_distance(a: int, b: int) -> int:
    return "{0:b}".format(a ^ b).count("1")
