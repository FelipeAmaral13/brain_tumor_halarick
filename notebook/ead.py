import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

# Carrega o dataset
df = pd.read_csv("model/trained/haralick_dataset.csv")

# Estrutura básica do dataset
basic_info = {
    "shape": df.shape,
    "columns": df.columns.tolist(),
    "dtypes": df.dtypes,
    "missing_values": df.isnull().sum(),
    "duplicated_rows": df.duplicated().sum(),
    "class_distribution": df["label"].value_counts()
}

# Estatísticas descritivas

# Correlação entre os descritores
correlation_matrix = df.drop(columns=['filename', 'label']).corr()

# Normalização
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}

# Aplicando as normalizações
scaled_data = {
    name: pd.DataFrame(scaler.fit_transform(df.drop(columns=['filename', 'label'])), 
                       columns=[f'f{i+1}' for i in range(14)])
    for name, scaler in scalers.items()
}


