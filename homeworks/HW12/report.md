# HW12 - временные ряды в PyTorch: baseline-модели и GRU

## 1. Кратко: что сделано
В домашней работе рассматривается задача one-step-ahead прогнозирования временного ряда по датасету `S12-hw-dataset.csv`. Реализованы temporal split без перемешивания, три baseline-подхода (`B1`, `B2`, `B3`) и рекуррентная модель `GRU` (`R1`) в PyTorch. После запуска ноутбука сохраняются таблица результатов, конфиг лучшей нейросетевой модели, веса и основные графики.

## 2. Среда и воспроизводимость
- Python 3.11
- Основные библиотеки: `torch`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- Устройство: `cuda`, если доступно, иначе `cpu`
- Seed: `42`
- Запуск: открыть [HW12.ipynb](/C:/AiCourseMirea/aie-demidov/homeworks/HW12/HW12.ipynb) и выполнить `Run All`
- Входной файл: положить `S12-hw-dataset.csv` в папку [data](/C:/AiCourseMirea/aie-demidov/homeworks/HW12/data)

## 3. Данные и постановка задачи
Используется временной ряд с обязательными колонками `date` и `target`. Колонка `date` преобразуется в формат `datetime`, после чего данные сортируются по времени. Целевая переменная для прогноза: `target`.

Постановка задачи: предсказать значение ряда на один шаг вперёд (`horizon = 1`) без утечки информации из будущего. Для оценки качества используются `MAE`, `RMSE` и `MAPE`.

## 4. Temporal split и признаки
Данные делятся по времени в пропорции `70% / 15% / 15%` на `train / val / test`. Перемешивание не применяется.

Для baseline-моделей используются:
- лаги `t-1`, `t-7`, `t-14`, `t-28`
- скользящие средние по окнам `7` и `14`
- календарные признаки `dayofweek` и `month`

Для `GRU` используется только история самой целевой переменной: окно длины `28` наблюдений.

## 5. Модели и эксперименты
Проведены следующие эксперименты:

- `B1 (naive-last)`: прогноз равен последнему известному значению ряда
- `B2 (moving-average)`: прогноз равен среднему по последним 7 значениям
- `B3 (ridge-lags)`: `Ridge` на лагах, rolling statistics и календарных признаках
- `R1 (gru-forecast)`: `GRU` в PyTorch с `hidden_size=64`, `num_layers=2`, `dropout=0.2`

Ключевые гиперпараметры `R1`:
- `window_size = 28`
- `batch_size = 32`
- `optimizer = Adam`
- `learning_rate = 5e-4`
- `loss = MSELoss`
- `epochs = 30`

## 6. Результаты
Файлы с результатами после запуска ноутбука:

- Таблица результатов: [artifacts/runs.csv](/C:/AiCourseMirea/aie-demidov/homeworks/HW12/artifacts/runs.csv)
- Веса лучшей GRU: [artifacts/best_gru.pt](/C:/AiCourseMirea/aie-demidov/homeworks/HW12/artifacts/best_gru.pt)
- Конфиг лучшей GRU: [artifacts/best_gru_config.json](/C:/AiCourseMirea/aie-demidov/homeworks/HW12/artifacts/best_gru_config.json)
- Финальная оценка на test: [artifacts/final_test_evaluation.json](/C:/AiCourseMirea/aie-demidov/homeworks/HW12/artifacts/final_test_evaluation.json)
- График temporal split: [artifacts/figures/series_split.png](/C:/AiCourseMirea/aie-demidov/homeworks/HW12/artifacts/figures/series_split.png)
- Сравнение baseline-моделей: [artifacts/figures/baselines_compare.png](/C:/AiCourseMirea/aie-demidov/homeworks/HW12/artifacts/figures/baselines_compare.png)
- Кривые обучения GRU: [artifacts/figures/gru_learning_curves.png](/C:/AiCourseMirea/aie-demidov/homeworks/HW12/artifacts/figures/gru_learning_curves.png)
- Прогноз лучшей модели на test: [artifacts/figures/best_forecast_test.png](/C:/AiCourseMirea/aie-demidov/homeworks/HW12/artifacts/figures/best_forecast_test.png)

Итоговые численные значения нужно смотреть в `runs.csv` и `final_test_evaluation.json`, так как они зависят от конкретного содержимого датасета и вычисляются при запуске ноутбука.

## 7. Анализ
Baseline `B1` задаёт нижнюю границу качества и показывает, насколько ряд инерционный. `B2` сглаживает шум, но может хуже реагировать на быстрые изменения. `B3` обычно даёт более сильный результат за счёт использования лагов и простых календарных признаков.

`GRU` полезна тем, что сама учится извлекать закономерности из окна наблюдений, не требуя ручного задания большого количества признаков. Если `R1` обгоняет `B3`, это означает, что последовательная нейросетевая модель смогла лучше уловить динамику ряда. Если разрыв небольшой, то для данного датасета линейная модель на лагах уже является сильным и дешёвым baseline.

## 8. Итоговый вывод
В проекте реализован воспроизводимый пайплайн для прогноза временного ряда: от корректного temporal split до сравнения baseline-моделей и `GRU`. После `Run All` репозиторий получает ту же основную структуру артефактов, что и в референсном примере.

## 9. Приложение (опционально)
При желании можно добавить:
- дополнительные baseline-модели
- другой размер окна для `GRU`
- сравнение `GRU` и `LSTM`
- анализ ошибок на отдельных участках test-отрезка
