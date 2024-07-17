
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


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

def main():
    loaded_model = load_model('modelo_completo.h5')
    # Ejemplo de evaluación con texto y URL de imagen
    text_input = ""
    image_input = "./data/test/imagenprueba.jpg"

    positive_percent, negative_percent, neutral_percent = evaluate_model(loaded_model, text_input, image_input)
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

# Ejecutar el programa principal
if __name__ == "__main__":
    main()
