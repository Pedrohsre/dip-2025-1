"""
task-11-blur-estimation-with-fourier-transform.py

>>> IMPORTANT <<<
Implement the function `frequency_blur_score` below.

Rules:
- Keep the function name and signature EXACTLY the same.
- Do NOT use any external network calls.
- You may ONLY use standard Python, NumPy, and OpenCV (cv2).
- Return a single float (higher = sharper OR lower = blurrier, but be consistent).

Tip (from the FFT blur-detection tutorial):
- Convert to grayscale
- 2D FFT -> shift DC to center
- Zero-out a centered square (low frequencies)
- Magnitude spectrum (e.g., log1p(abs(...)))
- Use the mean magnitude of the remaining spectrum as the score
"""

from typing import Union
import numpy as np
import cv2


def frequency_blur_score(
    image: Union[np.ndarray, "cv2.Mat"],
    center_size: int = 60
) -> float:
    """
    Compute a blur/sharpness score in the frequency domain.

    Parameters
    ----------
    image : np.ndarray
        Input image, grayscale or BGR. Any dtype accepted; will be converted to float32.
    center_size : int, default=60
        Side length of the central square (low-frequency) region to suppress.

    Returns
    -------
    float
        A scalar score. You should make it so that SHARPER images get a HIGHER score.
        (This will align with the grader's expectation.)
    """
    # ====== YOUR CODE STARTS HERE ======

    # Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Convert to float32
    gray = gray.astype(np.float32)
    
    # 2D FFT -> shift DC to center
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    
    # Zero-out a centered square (low frequencies)
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    
    # copy to modify
    fft_filtered = fft_shifted.copy()
    
    half_size = center_size // 2
    y1 = max(0, center_y - half_size)
    y2 = min(h, center_y + half_size)
    x1 = max(0, center_x - half_size)
    x2 = min(w, center_x + half_size)
    
    # Remove low frequencies
    fft_filtered[y1:y2, x1:x2] = 0
    
    # magnitude
    magnitude = np.abs(fft_filtered)
    magnitude_log = np.log1p(magnitude)
    
    # SHARPER images get a HIGHER score
    score = np.mean(magnitude_log)
    
    # ====== YOUR CODE ENDS HERE ======
    return float(score)
    return score
