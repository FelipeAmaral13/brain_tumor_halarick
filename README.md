# 🧠 HARALRICK_DESCRITORES

**Sistema de classificação automática de imagens cerebrais (RM/TC)** utilizando **descritores de textura de Haralick** combinados com uma **Rede Neural Artificial (RNA)** desenvolvida em Keras. O projeto realiza extração, treinamento, validação e deploy com interface via **Streamlit**.

---

## 📁 Estrutura do Projeto

```bash
HALARICK_DESCRITORES/
├── app/                     # Interface do usuário via Streamlit
│   ├── app_streamlit.py     # App principal para predição via upload de imagem
│   └── config.py            # Caminhos e parâmetros globais
│
├── core/                    # Módulos de processamento e extração
│   ├── dataset_builder.py   # Geração de CSV a partir de imagens
│   ├── extractor.py         # Cálculo dos 14 descritores de Haralick
│   └── utils.py             # Funções auxiliares (scaling, predição)
│
├── model/                   # Pipeline de modelagem e inferência
│   ├── model.py             # Treinamento completo com Keras
│   ├── predict_model.py     # Predição via linha de comando
│   └── trained/             # Modelos e scalers persistidos (.h5, .joblib)
│
├── images/                  # Diretório com imagens organizadas por classe
│   ├── brain_glioma/
│   ├── brain_menin/
│   └── brain_tumor/
│
├── notebook/                # Análises exploratórias (EDA)
│   └── ead.py
│
├── trained/                 # Artefatos gerados no treinamento
│   └── haralick_model.h5
│
├── haralick_dataset.csv     # Dataset final com vetores de descritores
├── main.py                  # Geração inicial do CSV com dataset_builder
├── README.md
└── requirements.txt
```

---

## 🚀 Como executar o projeto

### 🔧 Instalação das dependências

```bash
uv pip install -r requirements.txt
```

> Ou, se estiver usando `pip` padrão:
```bash
pip install -r requirements.txt
```

---

## 🧪 1. Treinamento do Modelo

```bash
python model/model.py
```

---

## 📊 2. Predição via terminal

```bash
python model/predict_model.py images/brain_menin/001.jpg
```

---

## 🌐 3. Interface via Streamlit

```bash
streamlit run app/app_streamlit.py
```

---

## 🔍 Funcionalidades

- 📥 Upload de imagens no navegador
- 🧠 Extração de 14 descritores de Haralick por GLCM
- 🤖 RNA multicamada com dropout e redução dinâmica do learning rate
- 📊 Métricas: Accuracy, F1-macro, Curva de treino/validação
- 🔬 Ideal para aplicações clínicas exploratórias (ex: triagem de imagens)

---

## 📌 Requisitos

- Python 3.9+
- TensorFlow 2.x
- scikit-image
- scikit-learn
- Streamlit

---

## 📁 Artefatos Gerados

| Arquivo                          | Descrição                                  |
|----------------------------------|--------------------------------------------|
| `haralick_model.h5`              | Modelo Keras treinado                      |
| `scaler.joblib`                  | Escalador padrão usado no treino           |
| `label_encoder.joblib`           | Codificador das classes                    |
| `haralick_dataset.csv`           | Dataset de entrada (descritores + label)  |
| `history.csv`                    | Histórico de treino (loss e accuracy)      |

---

## 🤝 Contribuição

- Estrutura modular e extensível para outros classificadores (SVM, RF, etc.)
- Ideal para uso acadêmico, ensino de visão computacional e P&D em saúde

---