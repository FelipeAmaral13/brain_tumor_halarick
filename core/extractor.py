# extractor.py
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from skimage.io import imread
import math

def calculate_all_haralick_descriptors(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    """
    Calcula os 14 descritores de Haralick para uma imagem em tons de cinza.
    Retorna a média dos descritores para as direções e distâncias informadas.
    """
    # Quantiza imagem para os níveis desejados
    image = (image / image.max() * (levels - 1)).astype(np.uint8)

    # Matriz de coocorrência
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    # Descritores diretos do skimage
    asm = graycoprops(glcm, 'ASM').mean()
    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    energy = graycoprops(glcm, 'energy').mean()

    # Entrópia
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))

    # Distribuições marginais
    px = np.sum(glcm, axis=(1, 3)).flatten()
    py = np.sum(glcm, axis=(0, 3)).flatten()


    # Validação de tamanho
    assert px.shape[0] == levels, f"px shape mismatch: {px.shape[0]} != {levels}"
    assert py.shape[0] == levels, f"py shape mismatch: {py.shape[0]} != {levels}"

    # Momentos
    ux = np.dot(np.arange(levels), px)
    uy = np.dot(np.arange(levels), py)
    sigx = np.sqrt(np.dot((np.arange(levels) - ux) ** 2, px))
    sigy = np.sqrt(np.dot((np.arange(levels) - uy) ** 2, py))

    # f4: Variance
    variance = np.sum([(i - ux) ** 2 * px[i] for i in range(levels)])

    # f6-8: Sum features
    p_sum = np.array([np.sum([glcm[i, j, 0, 0]
                              for i in range(levels)
                              for j in range(levels)
                              if i + j == k]) for k in range(2 * levels)])
    sum_avg = np.sum([i * p_sum[i] for i in range(len(p_sum))])
    sum_var = np.sum([(i - sum_avg) ** 2 * p_sum[i] for i in range(len(p_sum))])
    sum_entropy = -np.sum([p * np.log2(p + 1e-10) for p in p_sum])

    # f10-11: Difference features
    p_diff = np.array([np.sum([glcm[i, j, 0, 0]
                               for i in range(levels)
                               for j in range(levels)
                               if abs(i - j) == k]) for k in range(levels)])
    diff_var = np.var(p_diff)
    diff_entropy = -np.sum([p * np.log2(p + 1e-10) for p in p_diff])

    # f12: IMC1
    HX = -np.sum([p * np.log2(p + 1e-10) for p in px])
    HY = -np.sum([p * np.log2(p + 1e-10) for p in py])
    HXY1 = -np.sum([glcm[i, j, 0, 0] * np.log2(px[i] * py[j] + 1e-10)
                    for i in range(levels) for j in range(levels)])
    if max(HX, HY) == 0:
        imc1 = 0
    else:
        imc1 = (entropy - HXY1) / max(HX, HY)

    # f13: IMC2
    HXY2 = -np.sum([px[i] * py[j] * np.log2(px[i] * py[j] + 1e-10)
                    for i in range(levels) for j in range(levels)])
    imc2 = math.sqrt(1 - math.exp(-2 * (HXY2 - entropy))) if HXY2 > entropy else 0

    # f14: MCC (aproximação via maior valor da GLCM)
    mcc = np.max(glcm)

    return [
        asm, contrast, correlation, variance, homogeneity,
        sum_avg, sum_var, sum_entropy, entropy, diff_var,
        diff_entropy, imc1, imc2, mcc
    ]
