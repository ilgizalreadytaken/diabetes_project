import pickle
import numpy as np
import os


# путь к модели
MODEL_PATH = "model.pkl"


# 1. Проверка загрузки модели
def test_model_load():
    assert os.path.exists(MODEL_PATH)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    assert model is not None


# 2. Проверка предсказания (не падает и возвращает 1 значение)
def test_prediction_runs():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    sample = np.array([[2, 120, 70, 20, 80, 25.0, 0.5, 30]])
    pred = model.predict(sample)

    assert pred is not None
    assert len(pred) == 1


# 3. Проверка формата ответа (0 или 1)
def test_prediction_format():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    sample = np.array([[2, 120, 70, 20, 80, 25.0, 0.5, 30]])
    pred = model.predict(sample)[0]

    assert pred in [0, 1]