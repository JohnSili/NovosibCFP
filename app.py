import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Загрузите модели из файлов anomaly_model.pkl и y_model.pkl
with open('anomaly_model.pkl', 'rb') as f:
    anomaly_model = pickle.load(f)

with open('y_model.pkl', 'rb') as f:
    y_model = pickle.load(f)

# # Загрузите датасет из файла dataset.csv
# dataset = pd.read_csv('dataset.csv')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Получите данные из запроса
        data = request.get_json()

        # Обработайте данные
        # ...

        # Используйте модели для предсказания результатов
        anomaly_result = anomaly_model.predict(data)
        y_result = y_model.predict(data)

        # Верните результаты в формате JSON
        return jsonify({'anomaly_result': anomaly_result.tolist(), 'y_result': y_result.tolist()})
    else:
        return 'Добро пожаловать в мое приложение!'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')