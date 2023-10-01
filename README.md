# NovosibCFP

#Моделька
В файле модель_+_вывод графиков - находится модель random forest, прогнозирующая параметры BSM, также там находится другая модель, которая позволяет искать аномалии в данных и вывод графиков, какие аномалии повлияли на предсказание конкретных данных

#Docker
В папке docker находятся файлы для работы докера: Dockerfile, requirements.txt, app.py. index.html находится в папке docker/templates.
В файле Dockerfile находятся необходимые настройки для создания образа. 
Файл requirements.txt содержит список библиотек, необходимых для работы mvp.
app.py - основной файл, в котором находится загрузка файла с сервера и его отправка на вход модели. На выходе получаем график аномалий.
index.html - файл с фронтом.
