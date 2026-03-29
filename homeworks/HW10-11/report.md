# HW10-11 - Report

## 1. Кратко: что сделано
- Часть A: классификация на `STL10` с экспериментами `C1-C4`.
- Часть B: segmentation на `OxfordIIITPet` с режимами `V1-V2`.
- Все результаты сохранены в [artifacts/runs.csv](artifacts/runs.csv).

## 2. Среда и воспроизводимость
- Framework: `PyTorch` + `torchvision`
- Seed: `42`
- Device: `cuda`, если доступен, иначе `cpu`

## 3. Данные
- Датасет части A: `STL10` из `torchvision.datasets.STL10`.
- Split части A: `train` разбит на `train/val` в пропорции `80/20`, `test` использован для финальной оценки лучшей модели.
- Датасет части B: `OxfordIIITPet` из `torchvision.datasets.OxfordIIITPet`.
- Foreground для segmentation: пиксели питомца, где trimap `!= 2`.

## 4. Часть A: модели и обучение (C1-C4)
- `C1`: simple CNN без аугментаций.
- `C2`: та же simple CNN, но с аугментациями `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`.
- `C3`: `ResNet18` с pretrained weights, обучается только классификационная голова.
- `C4`: `ResNet18` с pretrained weights, partial fine-tuning для `layer4 + fc`.
- Loss: `CrossEntropyLoss`.
- Основная метрика: `accuracy`.
- Артефакты: [artifacts/best_classifier.pt](artifacts/best_classifier.pt), [artifacts/best_classifier_config.json](artifacts/best_classifier_config.json), [artifacts/figures/classification_curves_best.png](artifacts/figures/classification_curves_best.png), [artifacts/figures/classification_compare.png](artifacts/figures/classification_compare.png), [artifacts/figures/augmentations_preview.png](artifacts/figures/augmentations_preview.png).

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)
- Трек: `segmentation`.
- Модель: pretrained `DeepLabV3_ResNet50`.
- `V1`: threshold `0.5`.
- `V2`: threshold `0.7`.
- Метрики: `mean_iou`, `precision`, `recall`.
- Артефакты: [artifacts/figures/segmentation_examples.png](artifacts/figures/segmentation_examples.png), [artifacts/figures/segmentation_metrics.png](artifacts/figures/segmentation_metrics.png).

## 6. Результаты
- Лучшая модель части A по `best_val_accuracy`: `C4` с `val_accuracy=0.9570`.
- Финальная `test_accuracy` лучшей модели: `0.9481`.
- `V1`: precision=`0.9575`, recall=`0.8665`, mean_iou=`0.8325`.
- `V2`: precision=`0.9697`, recall=`0.8087`, mean_iou=`0.7871`.
- Полная таблица результатов: [artifacts/runs.csv](artifacts/runs.csv).

## 7. Анализ
- Аугментации в `C2` улучшают устойчивость модели по сравнению с базовым `C1`.
- Transfer learning на `ResNet18` даёт заметный прирост относительно простой CNN.
- Partial fine-tuning в `C4` оказался лучше, чем head-only обучение в `C3`.
- Для segmentation увеличение threshold с `0.5` до `0.7` повышает precision, но снижает recall и `mean_iou`.

## 8. Итоговый вывод
- Обязательный минимум HW10-11 выполнен.
- В части A показан эффект CNN, аугментаций и transfer learning.
- В части B выполнен запуск готового segmentation pipeline, визуализация результатов и расчёт базовых метрик.
