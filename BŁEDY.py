from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from PIL import Image

def calculate_image_quality(original_image_path, output_image_path):
    # Wczytaj obrazy i przekształć je na tablice numpy
    original_image = np.array(Image.open(original_image_path).convert('L'))
    output_image = np.array(Image.open(output_image_path).convert('L'))

    # Oblicz MSE
    mse = mean_squared_error(original_image, output_image)

    # Oblicz MAE
    mae = mean_absolute_error(original_image, output_image)

    return mse, mae

# Ścieżki do obrazów
original_image_path = 'saguaro.jpg'
output_image_path = 'output_bicubic_6_256.jpg'

# Oblicz jakość obrazu
mse, mae = calculate_image_quality(original_image_path, output_image_path)

print(f'MSE: {mse}, MAE: {mae}')
