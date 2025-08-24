import cv2
import numpy as np

def compute_histogram_intersection(image_path1: str, image_path2: str) -> float:

    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both image paths are invalid.")
    hist1 = np.histogram(img1, bins=256, range=(0, 256))[0]
    hist2 = np.histogram(img2, bins=256, range=(0, 256))[0]
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    intersection = np.sum(np.minimum(hist1, hist2))
    return float(intersection)
