import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def sample_function(func, K):
    # Generowanie K punktów równomiernie rozłożonych w zakresie od 0 do 1
    points = np.linspace(0, 1, K)

    # Próbkowanie funkcji w wygenerowanych punktach
    samples = np.vectorize(func)(points)

    return points, samples


# Wczytanie obrazu
image_path = 'saguaro.jpg'
image = imageio.imread(image_path)

# Funkcje interpolujące
interpolations = ['nearest', 'bilinear', 'bicubic']

# Liczby próbek
K_values = [3 * 2 ** k for k in range(1, 4)]

# Rozmiary obrazów po pomniejszeniu
p_values = [2 ** p for p in [7, 8]]

for interp in interpolations:
    for K in K_values:
        # Wygładzanie obrazu za pomocą splotu
        filter = np.ones((K, K)) / (K ** 2)
        smoothed_image = np.zeros_like(image)
        for i in range(3):  # Dla każdego kanału koloru
            smoothed_image[..., i] = ndimage.convolve(image[..., i], filter, mode='constant', cval=0.0)

        for p in p_values:
            # Zmniejszenie obrazu
            resized_image = ndimage.zoom(smoothed_image, (p / image.shape[0], p / image.shape[1], 1), order=0)

            # Przywrócenie oryginalnego rozmiaru obrazu
            restored_image = ndimage.zoom(resized_image, (image.shape[0] / p, image.shape[1] / p, 1), order=0)

            # Zapisanie obrazu
            output_path = f'output_{interp}_{K}_{p}.jpg'
            imageio.imsave(output_path, restored_image)