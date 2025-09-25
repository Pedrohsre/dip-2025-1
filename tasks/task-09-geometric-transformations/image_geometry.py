# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    height, width = img.shape
    
    #1. Translated image (shift right and down)
    translated = np.zeros_like(img)
    shift_x, shift_y = 20, 15  # pixels to shift
    if shift_y < height and shift_x < width:
        translated[shift_y:, shift_x:] = img[:height-shift_y, :width-shift_x]
    
    #2. Rotated image (90 degrees clockwise)
    rotated = np.rot90(img, k=-1)  # k=-1 for clockwise
    
    #3. Horizontally stretched image (scale width by 1.5)
    new_width = int(width * 1.5)
    stretched = np.zeros((height, new_width))
    for i in range(height):
        for j in range(new_width):
            # Map back to original coordinates
            orig_j = int(j / 1.5)
            if orig_j < width:
                stretched[i, j] = img[i, orig_j]
    
    #4. Horizontally mirrored image (flip along vertical axis)
    mirrored = np.fliplr(img)
    
    #5. Barrel distorted image (simple distortion using a radial function)
    distorted = np.zeros_like(img)
    center_x, center_y = width // 2, height // 2
    
    for y in range(height):
        for x in range(width):
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                max_distance = np.sqrt(center_x*center_x + center_y*center_y)
                normalized_distance = distance / max_distance
                distortion_factor = 1 - 0.3 * normalized_distance * normalized_distance
                
                orig_x = int(center_x + dx * distortion_factor)
                orig_y = int(center_y + dy * distortion_factor)
                
                if 0 <= orig_x < width and 0 <= orig_y < height:
                    distorted[y, x] = img[orig_y, orig_x]
    
    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched.astype(img.dtype),
        "mirrored": mirrored,
        "distorted": distorted
    }