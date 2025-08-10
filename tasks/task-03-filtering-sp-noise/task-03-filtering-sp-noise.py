import cv2
import numpy as np

def remove_salt_and_pepper_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes salt and pepper noise from a grayscale image.

    Parameters:
        image (np.ndarray): Noisy input image (grayscale).

    Returns:
        np.ndarray: Denoised image.
    """
    denoised_image = cv2.medianBlur(image, 5)
    return denoised_image

if __name__ == "__main__":
    noisy_image = cv2.imread("C:\\Users\\pedro\\Documents\\pdi\\dip-2025-1\\img\\head.png", cv2.IMREAD_GRAYSCALE)

    if noisy_image is None:
        print("Error")
    else:
        denoised_image = remove_salt_and_pepper_noise(noisy_image)
        
        cv2.imwrite("denoised_image.png", denoised_image)
        print("Denoised image saved as 'denoised_image.png'")
        
        try:
            cv2.imshow("Original Noisy Image", noisy_image)
            cv2.imshow("Denoised Image", denoised_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("Display not available, but image was saved successfully.")
