import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_float
from core.extractor import calculate_all_haralick_descriptors
from sklearn.preprocessing import StandardScaler

# Escalador treinado (fixo, ideal seria persistido junto ao modelo)
scaler = StandardScaler()

def process_uploaded_image(uploaded_file):
    image = imread(uploaded_file)
    if image.ndim == 3:
        image = rgb2gray(image)
    image = img_as_float(image)

    features = calculate_all_haralick_descriptors(image)
    features = np.array(features).reshape(1, -1)
    return scaler.fit_transform(features) 