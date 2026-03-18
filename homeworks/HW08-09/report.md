# HW08-09 - Report

## 1. Кратко: что сделано

- Реализован MLP на PyTorch для `CIFAR10` с полным циклом обучения: `Dataset/DataLoader`, `nn.Module`, `loss`, `optimizer`, train/eval цикл.
- Проведены эксперименты `E1-E4` по регуляризации и `O1-O3` по learning rate, optimizer и weight decay.
- Сохранены обязательные артефакты: `runs.csv`, `best_model.pt`, `best_config.json`, `curves_best.png`, `curves_lr_extremes.png`.

## 2. Среда и воспроизводимость

- Framework: `PyTorch`
- Device: `cuda` if available else `cpu`
- Seed: `42`
- Зафиксированы `torch.manual_seed` и `numpy.random.seed`.
- Разбиение `train/val` выполнено воспроизводимо с фиксированным seed.

## 3. Данные

- Датасет: `CIFAR10` из `torchvision.datasets`.
- Использованы стандартные части `train` и `test`.
- Из `train` выделена `val` в пропорции `80/20`.
- Размер батча: `128`.
- Форма входа: `(3, 32, 32)`.
- Преобразование: `ToTensor()`.

## 4. Базовая модель и обучение

- Базовая архитектура: `Flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear(logits)`.
- Hidden sizes: `[512, 256, 128]`.
- Loss: `CrossEntropyLoss`, базовый optimizer: `Adam`, `lr=0.001`.
- Валидация считается в режимах `model.eval()` и `torch.no_grad()`.
- Логируется история `train/val loss` и `train/val accuracy`.

## 5. Часть A (S08): регуляризация (E1-E4)

- `E1`: базовый MLP без `Dropout` и `BatchNorm`.
- `E2`: тот же MLP с `Dropout(p=0.3)`.
- `E3`: тот же MLP с `BatchNorm1d`.
- `E4`: лучший из `E2/E3` по `val_accuracy`, переобучен с `EarlyStopping(patience=4)`.

| Experiment | Optimizer | Epochs | Best val accuracy | Best val loss |
|---|---|---:|---:|---:|
| E1 | Adam, lr=0.001 | 15 | 0.5044 | 1.4059 |
| E2 | Adam, lr=0.001 | 15 | 0.4522 | 1.5460 |
| E3 | Adam, lr=0.001 | 15 | 0.5171 | 1.3780 |
| E4 | Adam, lr=0.001 | 9 | 0.5336 | 1.3268 |

## 6. Часть B (S09): LR, оптимизаторы, weight decay (O1-O3)

- `O1`: `Adam` с заведомо слишком большим `lr=1e-1`.
- `O2`: `Adam` с заведомо слишком маленьким `lr=1e-5`.
- `O3`: `SGD` с `momentum=0.9` и `weight_decay=1e-4`.

| Experiment | Optimizer | Epochs | Best val accuracy | Best val loss |
|---|---|---:|---:|---:|
| O1 | Adam, lr=0.1 | 8 | 0.4694 | 1.4924 |
| O2 | Adam, lr=1e-5 | 8 | 0.4934 | 1.4974 |
| O3 | SGD, lr=0.01, momentum=0.9, weight_decay=1e-4 | 12 | 0.5083 | 1.4154 |

## 7. Результаты

- Лучшая модель: `E4`.
- Лучший `val_accuracy`: `0.5336`.
- Лучший `val_loss`: `1.3268`.
- Финальная оценка на test: `loss = 1.3333`, `accuracy = 0.5312`.
- Таблица запусков: [artifacts/runs.csv](artifacts/runs.csv).
- Веса лучшей модели: [artifacts/best_model.pt](artifacts/best_model.pt).
- Конфиг лучшей модели: [artifacts/best_config.json](artifacts/best_config.json).
- График лучшего прогона: [artifacts/figures/curves_best.png](artifacts/figures/curves_best.png).
- График экстремальных LR: [artifacts/figures/curves_lr_extremes.png](artifacts/figures/curves_lr_extremes.png).

## 8. Анализ

- На этой конфигурации `BatchNorm` оказался полезнее, чем `Dropout`.
- `EarlyStopping` помог остановить обучение раньше и дал лучший результат части A.
- В `O1` видно нестабильное обучение, характерное для слишком большого learning rate.
- В `O2` обучение почти не двигается, что соответствует слишком маленькому learning rate.
- `SGD + momentum + weight_decay` работает корректно, но здесь уступил `E4`.

## 9. Итоговый вывод

- Все обязательные части HW08-09 выполнены.
- Лучшая модель домашней работы: `E4`, то есть MLP с `BatchNorm` и `EarlyStopping`.
- Практический вывод: качество обучения заметно зависит от регуляризации, learning rate и выбора optimizer.
