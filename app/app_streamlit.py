# app/app.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_float

from core.extractor import calculate_all_haralick_descriptors
from config import MODEL_PATH, SCALER_PATH, ENCODER_PATH

st.set_page_config(page_title="Classificador de Tumores Cerebrais", page_icon="🧠", layout="centered")


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, scaler, label_encoder


def preprocess_image(uploaded_file):
    image = imread(uploaded_file)
    if image.ndim == 3:
        image = rgb2gray(image)
    image = img_as_float(image)
    return image


def predict_from_image(image, model, scaler, label_encoder):
    features = calculate_all_haralick_descriptors(image)
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)
    predicted_index = np.argmax(prediction)
    predicted_class = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(np.max(prediction))

    return predicted_class, confidence, prediction[0], label_encoder.classes_


st.title("🧠 Classificador de Tumores Cerebrais com Descritores de Haralick")
st.write("Envie uma imagem para classificação (RM/TC) como glioma, meningioma ou tumor cerebral.")

uploaded_file = st.file_uploader("📤 Faça upload da imagem", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagem carregada", use_container_width=True)

    if st.button("🔍 Classificar"):
        try:
            image = preprocess_image(uploaded_file)
            model, scaler, label_encoder = load_artifacts()
            predicted_class, confidence, probs, class_names = predict_from_image(
                image, model, scaler, label_encoder
            )

            st.success(f"Classe prevista: **{predicted_class}**")
            st.metric(label="Confiança", value=f"{confidence:.2%}")

            st.write("Distribuição de Probabilidades:")
            st.bar_chart({label: float(prob) for label, prob in zip(class_names, probs)})

        except Exception as e:
            st.error(f"Erro ao processar a imagem: {e}")
