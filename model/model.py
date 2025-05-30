import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['label'].notna()].reset_index(drop=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])

    X = df[[f'f{i+1}' for i in range(14)]].astype(np.float32)
    y = to_categorical(df['label_encoded'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, le, scaler

def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_pipeline(csv_path='data/haralick_dataset.csv', model_path='model/trained/haralick_model.h5'):
    X, y, label_encoder, scaler = load_and_preprocess(csv_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = build_model(input_dim=X.shape[1], output_dim=y.shape[1])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint(model_path, save_best_only=True)

    # Treinamento
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    # Curvas de Aprendizado
    pd.DataFrame(history.history).to_csv("model/trained/history.csv", index=False)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss por Época'); plt.xlabel('Épocas'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Accuracy por Época'); plt.xlabel('Épocas'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True)

    plt.tight_layout(); plt.show()

    # Avaliação Final
    y_pred = model.predict(X_val)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_val, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_labels, y_pred_labels))
    print("F1-macro:", f1_score(y_true_labels, y_pred_labels, average='macro'))

    # Persistência de artefatos
    joblib.dump(label_encoder, 'model/trained/label_encoder.joblib')
    joblib.dump(scaler, 'model/trained/scaler.joblib')

    return model, label_encoder

if __name__ == '__main__':
    train_pipeline()
