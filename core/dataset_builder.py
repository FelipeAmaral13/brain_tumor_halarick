# core/dataset_builder.py

import os
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_float
from core.extractor import calculate_all_haralick_descriptors

def generate_dataset_from_directory(input_dir: str, output_csv: str) -> None:
    """
    Percorre diretórios de imagens classificados por rótulo e salva os descritores de Haralick em CSV.
    Estrutura esperada: data/class_name/*.jpg

    Args:
        input_dir (str): Caminho para a pasta base com subpastas por classe.
        output_csv (str): Caminho do arquivo CSV de saída.
    """
    data = []

    for label in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, label)
        if not os.path.isdir(class_dir):
            continue

        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            path = os.path.join(class_dir, img_file)
            try:
                image = imread(path)
                if image.ndim == 3:
                    image = rgb2gray(image)
                image = img_as_float(image)

                features = calculate_all_haralick_descriptors(image)
                data.append(features + [label])

            except Exception as e:
                print(f"[ERRO] {path}: {e}")

    columns = [f"f{i+1}" for i in range(14)] + ["label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"✅ Dataset salvo em {output_csv}")
