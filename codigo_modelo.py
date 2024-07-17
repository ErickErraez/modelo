import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import cv2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string

# Directorio de datos de imágenes
IMG_HEIGHT = 128
IMG_WIDTH = 128

def load_images_and_labels(folder_name):
    data_folder = f'./data/{folder_name}'
    images = []
    labels = []

    emotions_mapping = {
        'angry': 0,
        'disgusted': 0,
        'fearful': 0,
        'happy': 1,
        'neutral': 2,
        'sad': 0,
        'surprised': 1
    }

    for emotion in emotions_mapping:
        emotion_folder = os.path.join(data_folder, emotion)
        for image_name in os.listdir(emotion_folder):
            image_path = os.path.join(emotion_folder, image_name)
            image = cv2.imread(image_path)  # Lee la imagen usando OpenCV
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))  # Ajusta tamaño si es necesario
            images.append(image)
            labels.append(emotions_mapping[emotion])

    return np.array(images), np.array(labels)

# Mapear emociones a categorías reducidas
emotion_to_category = {
    'angry': 'negative',
    'disgusted': 'negative',
    'fearful': 'negative',
    'happy': 'positive',
    'neutral': 'neutral',
    'sad': 'negative',
    'surprised': 'positive'
}

# Función para cargar y preprocesar imágenes y texto
def load_and_preprocess_data():
    # Cargar imágenes y etiquetas
    train_images, train_labels = load_images_and_labels('train')
    test_images, test_labels = load_images_and_labels('test')

    # Cargar datos de texto y preprocesarlos
    data = pd.read_csv('./data/Social Media Emotion Dataset.csv')
    data['sentiment'] = data['label'].apply(map_to_sentiment)

    data['cleaned_text'] = data['text'].apply(preprocess_text)

    # Crear un objeto TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Ajustar y transformar los datos limpios
    X = vectorizer.fit_transform(data['cleaned_text'])

    # Obtener las nuevas etiquetas (sentimientos)
    y = data['sentiment']

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return train_images, train_labels, test_images, test_labels, X_train, X_test, y_train, y_test

def preprocess_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = text.translate(str.maketrans('', '', string.punctuation))  # Eliminar signos de puntuación
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]  # Eliminar stop words
    return ' '.join(words)

def map_to_sentiment(label):
    if label in ['Happy', 'Surprise']:
        return 'positive'
    elif label in ['Angry', 'Sad']:
        return 'negative'
    else:  # Neutral
        return 'neutral'

# Función para construir el modelo de red neuronal
def build_model(input_shape_images, max_words):
    # Capa de entrada para imágenes
    input_images = layers.Input(shape=input_shape_images)
    x_images = layers.Conv2D(32, (3, 3), activation='relu')(input_images)
    x_images = layers.MaxPooling2D((2, 2))(x_images)
    x_images = layers.Conv2D(64, (3, 3), activation='relu')(x_images)
    x_images = layers.MaxPooling2D((2, 2))(x_images)
    x_images = layers.Flatten()(x_images)
    x_images = layers.Dense(64, activation='relu')(x_images)

    # Capa de entrada para texto
    input_text = layers.Input(shape=(max_words,))
    x_text = layers.Dense(64, activation='relu')(input_text)

    # Combinación de capas de imágenes y texto
    combined = layers.concatenate([x_images, x_text])
    combined = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(3, activation='softmax')(combined)  # 3 categorías: positivo, negativo, neutral

    model = models.Model(inputs=[input_images, input_text], outputs=output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Función para entrenar el modelo con imágenes y texto
def train_model(model, train_images, train_labels, X_train, y_train, epochs=10):
    history = model.fit([train_images, X_train], train_labels, epochs=epochs, validation_split=0.2)
    return history

def evaluate_model(model, text, image_url):
    if text.strip() == '' and image_url.strip() != '':
        # Solo hay URL de imagen, llamar a función de evaluación de imagen
        positive_percent, negative_percent, neutral_percent = evaluate_model_image(model, image_url)
    elif text.strip() != '' and image_url.strip() == '':
        # Solo hay texto, llamar a función de evaluación de texto
        positive_percent, negative_percent, neutral_percent = evaluate_model_text(model, text)
    else:
        # Ambos o ninguno están presentes (tratamiento opcional según tu flujo)
        print("Debe proporcionar solo texto o solo una URL de imagen válida.")
        return None

    return positive_percent, negative_percent, neutral_percent

def evaluate_model_image(model, image_url):
    # Aquí simula la carga y evaluación de la imagen desde la URL
    # En un entorno real, deberías cargar la imagen desde la URL y luego preprocesarla según tu modelo
    print(f"Evaluando imagen desde URL: {image_url}")
    # Simulación de predicciones aleatorias para demostración
    predictions = np.random.rand(10, 3)  # Ejemplo de predicciones aleatorias
    predicted_labels = np.argmax(predictions, axis=1)

    # Calcular porcentajes basados en las predicciones
    total_samples = len(predicted_labels)
    positive_percentage = (np.sum(predicted_labels == 1) / total_samples) * 100
    negative_percentage = (np.sum(predicted_labels == 0) / total_samples) * 100
    neutral_percentage = (np.sum(predicted_labels == 2) / total_samples) * 100

    return positive_percentage, negative_percentage, neutral_percentage

def evaluate_model_text(model, text):
    # Aquí simula la evaluación de texto
    print(f"Evaluando texto: {text}")
    # Simulación de predicciones aleatorias para demostración
    predictions = np.random.rand(10, 3)  # Ejemplo de predicciones aleatorias
    predicted_labels = np.argmax(predictions, axis=1)

    # Calcular porcentajes basados en las predicciones
    total_samples = len(predicted_labels)
    positive_percentage = (np.sum(predicted_labels == 1) / total_samples) * 100
    negative_percentage = (np.sum(predicted_labels == 0) / total_samples) * 100
    neutral_percentage = (np.sum(predicted_labels == 2) / total_samples) * 100

    return positive_percentage, negative_percentage, neutral_percentage

# Función para visualizar el rendimiento del modelo
def plot_performance(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Función principal para ejecutar todo el flujo
def main():
    # Paso 1: Cargar y preprocesar datos de imágenes y texto
    train_images, train_labels, test_images, test_labels, X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Paso 2: Construir modelo combinado
    model = build_model(input_shape_images=train_images.shape[1:], max_words=X_train.shape[1])

    # Paso 3: Entrenar modelo con imágenes y texto
    history = train_model(model, train_images, train_labels, X_train, y_train)

    # Guardar modelo entrenado
    model.save('modelo_completo.h5')

    # Ejemplo de evaluación con texto y URL de imagen
    text_input = ""
    image_input = ""

    positive_percent, negative_percent, neutral_percent = evaluate_model(model, text_input, image_input)
    print(f"Porcentaje de positivo: {positive_percent}%")
    print(f"Porcentaje de negativo: {negative_percent}%")
    print(f"Porcentaje de neutro: {neutral_percent}%")

    # Visualizar porcentajes en un gráfico
    categories = ['Positive', 'Negative', 'Neutral']
    percentages = [positive_percent, negative_percent, neutral_percent]

    plt.figure(figsize=(6, 4))
    plt.bar(categories, percentages, color=['green', 'red', 'blue'])
    plt.xlabel('Sentiment Category')
    plt.ylabel('Percentage')
    plt.title('Sentiment Analysis Results')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.show()

    # Paso 5: Visualizar rendimiento del modelo
    plot_performance(history)

# Ejecutar el programa principal
if __name__ == "__main__":
    main()
