# EDA-CLI для анализа csv файлов
## доступные команды 
```
eda-cli overview data.csv          # Краткий обзор датасета
eda-cli report data.csv            # Полный EDA-отчет с графиками
```

## Краткий обзор датасета 

Выводит: 
- Количество строк
- Таблица по колонкам: типы, пропуски, уникальные значения, статистика

## Полный EDA-отчет с графиками

Генерирует каталог reports/ с:

- report.md — основной Markdown-отчёт
- summary.csv, missing.csv, correlation.csv — таблички
- duplicate_rows.csv* — дубликаты строк (если есть)
- top_categories/*.csv — топ-значения категорий
- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png

Параметры `report`: 
| Параметр | Дефолт значение | На что влияет |
|----------|-----------------|---------------|
|`--out-dir`| reports | Каталог для отчета|
|`--max-hist-columns`| 6 | Максимальное количество гистрограмм числовых колонок |
|`--top-k-categories`| 5 | Топ n значений для категорий |
|`--title` | "EDA-отчет" | Заголовок |
|`--min-missing-share` | 0.3 | Порог проблемных пропусков |

Пример вызова с параметрами: 
```
eda-cli report data/example.csv --out-dir my_report --max-hist-columns 8 --top-k-categories 10 --title "Анализ данных клиентов" --min-missing-share 0.25
```