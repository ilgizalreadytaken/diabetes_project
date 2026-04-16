# Диагностика диабета (ML проект)

## Описание проекта
Проект представляет собой систему машинного обучения для диагностики диабета на основе медицинских показателей пациента. Модель предсказывает наличие или отсутствие диабета (0 или 1).

## Цель
Разработать модель машинного обучения, которая:
- обучается на медицинских данных
- делает предсказания
- предоставляет простой веб-интерфейс

## Датасет
Используется датасет Pima Indians Diabetes Dataset.

Признаки:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Целевая переменная:
- Outcome (0 или 1)

## Технологии
- Python
- Pandas
- NumPy
- Scikit-learn
- Gradio
- Pytest

## Модель
Используется Logistic Regression.

Средняя точность: ~0.75–0.76

## Запуск проекта

Установка зависимостей:
pip install -r requirements.txt

Обучение модели:
python train.py

Запуск интерфейса:
python app.py

## Тестирование

Запуск тестов:
pytest tests

Ожидаемый результат:
3 passed

## Структура проекта
diabetes_project/
├── data/
├── models/
├── src/
├── app.py
├── train.py
├── tests/
├── requirements.txt
└── README.md