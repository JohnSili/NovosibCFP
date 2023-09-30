import io
import joblib
import pandas as pd
from flask import Flask, make_response, request, jsonify, render_template
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

app = Flask(__name__)

# Загрузите модели из файлов anomaly_model.pkl и y_model.pkl
# with open('anomaly_model.pkl', 'rb') as f:
#     anomaly_model = pickle.load(f)

y_model = joblib.load('y_model.pkl')

# # Загрузите датасет из файла dataset.csv
# dataset = pd.read_csv('dataset.csv')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Получите файл из запроса
        file = request.files['file']
        # Прочитайте данные из файла
        df = pd.read_csv(file)
        df['ts'] = pd.to_datetime(df['ts'])

        df['Weekday_reg'] = df['ts'].dt.weekday
        df['Hour_reg'] = df['ts'].dt.hour
        df['Minute_reg'] = df['ts'].dt.minute
        
        features = ['departure_terminal_encoded','checkin_terminal_encoded','Weekday_reg', 'Hour_reg','Minute_reg', 'DayTime','local_or_transfer_binary','config']                                                                                                                                     
        X = pd.DataFrame(data=df[features])

        target = ['VolumePerMinute','pax_arrival_profile']
        y = pd.DataFrame(data=df[target])
        # X = df[features].iloc[-1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Используйте модели для предсказания результатов
        # anomaly_result = anomaly_model.predict(data)
        y_pred_test = y_model.predict(X_test)

        anomaly_model_test = IsolationForest(contamination=0.05, random_state=42)
        anomaly_pred_test = anomaly_model_test.fit_predict(X_test)
        anomalies_X_test = X_test[anomaly_pred_test == -1]

        absolute_diff = abs(y_pred_test - y_test).sum(axis=1)

        anomalies_influence = absolute_diff[anomalies_X_test.index]
        # plt.switch_backend('agg')

        for idx in anomalies_influence.head(4).index:
            # Получаем данные x для этой аномалии
            anomaly_x = X_test.loc[idx]
            # Исключаем влияние столбца 'config'
            feature_influence = y_model.feature_importances_ * anomaly_x
            feature_influence['config'] = 0
            
            plt.figure(figsize=(10, 4))
            plt.plot(feature_influence, label=f'Аномалия в строке {idx}')
            plt.xlabel('Время')
            plt.ylabel('Влияние признака')
            plt.title(f'Влияние признака на предсказание для аномалии в строке {idx}')
            plt.legend()       
            plt.show()
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            response = make_response(buffer.getvalue())
            response.headers['Content-Type'] = 'image/png'
            return response
                
        
    else:
        return render_template('index.html')

@app.errorhandler(Exception)
def handle_exception(e):
    # Вывести информацию об ошибке в логи сервера
    app.logger.error(str(e))
    # Вернуть сообщение об ошибке в ответе сервера
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')