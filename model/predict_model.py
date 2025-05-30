import sys
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_float
from core.extractor import calculate_all_haralick_descriptors


MODEL_PATH = "trained/haralick_model.h5"
SCALER_PATH = "trained/scaler.joblib"
ENCODER_PATH = "trained/label_encoder.joblib"
CLASSES = ['brain_glioma', 'brain_menin', 'brain_tumor']  # pode ser carregado dinamicamente também


def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, scaler, label_encoder

def preprocess_image(image_path):
    image = imread(image_path)
    if image.ndim == 3:
        image = rgb2gray(image)
    image = img_as_float(image)
    return image

def predict(image_path):
    model, scaler, label_encoder = load_artifacts()
    image = preprocess_image(image_path)
    features = calculate_all_haralick_descriptors(image)
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    confidence = float(np.max(prediction))

    print(f"🧠 Previsão: {predicted_class}")
    print(f"📈 Confiança: {confidence:.2%}")
    print("📊 Probabilidades por classe:")
    for i, prob in enumerate(prediction[0]):
        print(f"  - {label_encoder.inverse_transform([i])[0]}: {prob:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python predict_model.py <caminho_da_imagem>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict(image_path)
