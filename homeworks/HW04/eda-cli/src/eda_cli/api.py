from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Dict, Optional, Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import (
    compute_quality_flags,
    correlation_matrix,
    missing_table,
    summarize_dataset,
    top_categories,
    DatasetSummary,
)

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.2.0",
    description=(
        "HTTP-сервис-заглушка для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных вместо настоящей ML-модели."
    ),
    docs_url="/docs",
    redoc_url=None,
)

# --------------------

class QualityRequest(BaseModel):
    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(..., ge=0.0, le=1.0, description="Максимальная доля пропусков")
    numeric_cols: int = Field(..., ge=0, description="Количество числовых колонок")
    categorical_cols: int = Field(..., ge=0, description="Количество категориальных колонок")


class QualityResponse(BaseModel):
    ok_for_model: bool = Field(..., description="True если датасет качественный")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Оценка качества")
    message: str = Field(..., description="Пояснение решения")
    latency_ms: float = Field(..., ge=0.0, description="Время обработки, мс")
    flags: Optional[Dict[str, bool]] = Field(default=None, description="Булевы флаги")
    dataset_shape: Optional[Dict[str, int]] = Field(default=None, description="Размеры датасета")



class QualityFlagsResponse(BaseModel):
    """Ответ с полным набором флагов качества."""
    
    flags: Dict[str, Any] = Field(
        ...,
        description="Полный набор флагов качества, включая числовые значения"
    )
    dataset_shape: Dict[str, int] = Field(
        ...,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}"
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды"
    )


# ---------- Системный эндпоинт ----------

@app.get("/health", tags=["system"])
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.2.0",
    }


# ----------  /quality  ----------

@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    start = perf_counter()
    score = 1.0
    score -= req.max_missing_share
    
    if req.n_rows < 1000:
        score -= 0.2
    if req.n_cols > 100:
        score -= 0.1
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05
        
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7
    
    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
    else:
        message = "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    print(
        f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
        f"max_missing_share={req.max_missing_share:.3f} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv ----------

@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных.")

    # Используем EDA-ядро
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # ИСПРАВЛЕНИЕ: передаём все необходимые параметры
    flags_all = compute_quality_flags(df, summary, missing_df)

    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Оставляем только булевы флаги для компактности
    flags_bool: Dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])

    print(
        f"[quality-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- /quality-flags-from-csv ----------

@app.post(
    "/quality-flags-from-csv",
    response_model=QualityFlagsResponse,
    tags=["quality"],
    summary="Полный набор флагов качества из CSV-файла",
)
async def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityFlagsResponse:
    """
    Эндпоинт для получения полного набора флагов качества из CSV-файла.
    Возвращает все флаги, включая те, что были добавлены в HW03.
    """
    
    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных.")

    # Используем EDA-ядро
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Получаем все флаги качества
    flags_all = compute_quality_flags(df, summary, missing_df)

    latency_ms = (perf_counter() - start) * 1000.0

    n_rows = int(df.shape[0])
    n_cols = int(df.shape[1])

    print(
        f"[quality-flags-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} "
        f"flags_count={len(flags_all)} latency_ms={latency_ms:.1f} ms"
    )

    return QualityFlagsResponse(
        flags=flags_all,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
        latency_ms=latency_ms,
    )


@app.post(
    "/head",
    tags=["quality"],
    summary="Первые N строк датасета",
)
async def get_head(
    file: UploadFile = File(...),
    n: int = 10
) -> Dict[str, Any]:
    """
    Возвращает первые N строк CSV-файла в JSON-формате.
    """
    
    if n < 1 or n > 1000:
        raise HTTPException(status_code=400, detail="Параметр n должен быть от 1 до 1000")

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл.")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных.")

    # Берем первые n строк
    head_df = df.head(n)
    
    # Конвертируем в словарь для JSON
    return {
        "n_rows": n,
        "total_rows": int(df.shape[0]),
        "data": head_df.to_dict(orient="records")
    }