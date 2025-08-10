import numpy as np

def compute_histogram_intersection(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute the histogram intersection similarity score between two grayscale images.

    This function calculates the similarity between the grayscale intensity 
    distributions of two images by computing the intersection of their 
    normalized 256-bin histograms.

    The histogram intersection is defined as the sum of the minimum values 
    in each corresponding bin of the two normalized histograms. The result 
    ranges from 0.0 (no overlap) to 1.0 (identical histograms).

    Parameters:
        img1 (np.ndarray): First input image as a 2D NumPy array (grayscale).
        img2 (np.ndarray): Second input image as a 2D NumPy array (grayscale).

    Returns:
        float: Histogram intersection score in the range [0.0, 1.0].

    Raises:
        ValueError: If either input is not a 2D array (i.e., not grayscale).
    """    
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Both input images must be 2D grayscale arrays.")

    ### START CODE HERE ###
    # Compute histograms for both images
    hist1 = np.histogram(img1, bins=256, range=(0, 256))[0]
    hist2 = np.histogram(img2, bins=256, range=(0, 256))[0]
    
    # Normalize histograms
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Compute intersection
    intersection = np.sum(np.minimum(hist1, hist2))
    ### END CODE HERE ###


    return float(intersection)


if __name__ == "__main__":
    # Teste simples com imagens sintéticas
    print("Testando a função de interseção de histogramas...")
    
    # Teste 1: Imagens idênticas (deve dar 1.0)
    img1 = np.full((100, 100), 128, dtype=np.uint8)
    img2 = np.full((100, 100), 128, dtype=np.uint8)
    resultado1 = compute_histogram_intersection(img1, img2)
    print(f"Teste 1 - Imagens idênticas: {resultado1:.4f} (esperado: 1.0)")
    
    # Teste 2: Imagens diferentes (deve dar 0.0)
    img3 = np.full((100, 100), 64, dtype=np.uint8)
    img4 = np.full((100, 100), 192, dtype=np.uint8)
    resultado2 = compute_histogram_intersection(img3, img4)
    print(f"Teste 2 - Imagens diferentes: {resultado2:.4f} (esperado: 0.0)")
    
    # Teste 3: Imagens com alguma sobreposição
    img5 = np.random.randint(0, 128, (100, 100), dtype=np.uint8)
    img6 = np.random.randint(64, 256, (100, 100), dtype=np.uint8)
    resultado3 = compute_histogram_intersection(img5, img6)
    print(f"Teste 3 - Imagens com sobreposição parcial: {resultado3:.4f}")
    
    print("Testes concluídos!")
