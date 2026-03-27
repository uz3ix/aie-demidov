# HW10-11 - Report

## 1. Кратко: что сделано
- Часть A: классификация на `STL10` с экспериментами `C1-C4`.
- Часть B: segmentation на `OxfordIIITPet` с режимами `V1-V2`.
- Все результаты сохранены в [artifacts/runs.csv](artifacts/runs.csv).

## 2. Среда и воспроизводимость
- Framework: `PyTorch` + `torchvision`
- Seed: `42`
- Device: `cuda`, если доступен, иначе `cpu`

## 3. Часть A: классификация изображений
- Датасет: `STL10` из `torchvision.datasets.STL10`.
- Эксперименты: `C1` simple CNN, `C2` simple CNN + augmentations, `C3` ResNet18 head-only, `C4` ResNet18 partial fine-tune.
- Лучшая модель по `best_val_accuracy`: `C4` с `val_accuracy=0.9570`.
- Финальная `test_accuracy` лучшей модели: `0.9481`.
- Артефакты: [artifacts/best_classifier.pt](artifacts/best_classifier.pt), [artifacts/best_classifier_config.json](artifacts/best_classifier_config.json), [artifacts/figures/classification_curves_best.png](artifacts/figures/classification_curves_best.png), [artifacts/figures/classification_compare.png](artifacts/figures/classification_compare.png), [artifacts/figures/augmentations_preview.png](artifacts/figures/augmentations_preview.png).

## 4. Часть B: segmentation track
- Датасет: `OxfordIIITPet` из `torchvision.datasets.OxfordIIITPet`.
- Foreground: пиксели питомца, где trimap `!= 2`.
- Модель: pretrained `DeepLabV3_ResNet50`.
- `V1`: threshold `0.5`, precision=`0.9575`, recall=`0.8665`, mean_iou=`0.8325`.
- `V2`: threshold `0.7`, precision=`0.9697`, recall=`0.8087`, mean_iou=`0.7871`.
- Артефакты: [artifacts/figures/segmentation_examples.png](artifacts/figures/segmentation_examples.png), [artifacts/figures/segmentation_metrics.png](artifacts/figures/segmentation_metrics.png).

## 5. Итоговые артефакты
- Ноутбук: [HW10-11.ipynb](HW10-11.ipynb)
- Отчёт: [report.md](report.md)
- Таблица запусков: [artifacts/runs.csv](artifacts/runs.csv)

## 6. Вывод
- Выполнен обязательный минимум HW10-11.
- Аугментации и partial fine-tuning сравниваются через `C1-C4`, а segmentation оценивается в двух режимах `V1-V2`.
