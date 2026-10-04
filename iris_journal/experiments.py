"""Группировка образцов и сохранение результатов сравнения методов."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import ReferenceSample
from .domain import FEATURE_COLUMNS, ValidationError
from .models import write_json_atomic
from .training import SPLIT_RANDOM_STATE


logger = logging.getLogger(__name__)


class ExperimentError(ValueError):
    """Отчёт имеет неподдерживаемый вид, повреждён или недоступен для чтения."""

    pass


def cluster_reference_samples(samples: list[ReferenceSample]) -> dict:
    """Группирует измерения K-means и возвращает строки, центры и показатели качества.

    Все четыре признака масштабируются; виды не участвуют в обучении.
    Силуэт вычисляется в масштабированном пространстве, ARI — по известным видам
    после группировки, а центры возвращаются в сантиметрах.
    """
    if len(samples) < 4 or len({sample.source_id for sample in samples}) != len(samples):
        raise ValidationError("Для группировки нужны уникальные образцы с измерениями.")
    for sample in samples:
        sample.measurements.validate()
    features = pd.DataFrame(
        [sample.measurements.feature_vector() for sample in samples], columns=FEATURE_COLUMNS,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clusterer", KMeans(n_clusters=3, n_init=10, random_state=SPLIT_RANDOM_STATE)),
    ])
    # Метки видов используются только после группировки для сопоставления результата.
    groups = pipeline.fit_predict(features)
    scaled = pipeline.named_steps["scaler"].transform(features)
    # Силуэт должен оценивать расстояния в том же пространстве, что использовал K-means.
    silhouette = float(silhouette_score(scaled, groups))
    adjusted_rand = float(adjusted_rand_score([sample.species.value for sample in samples], groups))
    # Для отчёта возвращаем центры из масштабированных координат в сантиметры.
    centers = pipeline.named_steps["scaler"].inverse_transform(pipeline.named_steps["clusterer"].cluster_centers_)
    report = {
        "algorithm": "kmeans", "clusters": 3, "random_state": SPLIT_RANDOM_STATE,
        "silhouette": silhouette, "adjusted_rand": adjusted_rand,
        "rows": [
            {
                "source_id": sample.source_id, "cluster_id": int(group),
                "actual_species": sample.species.value,
                **dict(zip(FEATURE_COLUMNS, sample.measurements.feature_vector())),
            }
            for sample, group in zip(samples, groups)
        ],
        "centers": [
            {"cluster_id": index, **{key: float(value) for key, value in zip(FEATURE_COLUMNS, center)}}
            for index, center in enumerate(centers)
        ],
    }
    logger.info(
        "Группировка завершена: образцов=%s, силуэт=%.4f, ARI=%.4f", len(samples), silhouette, adjusted_rand,
    )
    return report


class ExperimentStore:
    """Хранит самостоятельные JSON-отчёты и указатели последних расчётов каждого вида."""

    def __init__(self, directory: Path):
        """Запоминает папку отчётов; создаёт её только при первом сохранении."""
        self.directory = directory

    def _kind(self, kind: str) -> None:
        """Разрешает только отчёты сравнения классификаторов и группировки."""
        if kind not in ("comparison", "clustering"):
            raise ExperimentError("Неизвестный вид отчёта.")

    def save(self, kind: str, content: dict) -> dict:
        """Сохраняет новый отчёт и затем атомарно обновляет указатель последнего расчёта.

        Добавляет идентификатор, время UTC и версию формата. При ошибке записи
        полного отчёта прежний указатель остаётся доступным.
        """
        self._kind(kind)
        report = dict(
            content, run_id=uuid4().hex, kind=kind, format_version=1,
            created_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.directory / f"{report['run_id']}.json", report)
        # Указатель обновляется после полного отчёта, чтобы читатель не увидел
        # ссылку на ещё не записанный результат.
        write_json_atomic(self.directory / f"latest_{kind}.json", {"run_id": report["run_id"]})
        logger.info("Отчёт сохранён: вид=%s, id=%s", kind, report["run_id"])
        return report

    def latest(self, kind: str) -> dict | None:
        """Возвращает последний проверенный отчёт или None, если расчётов ещё нет.

        Проверяет идентификатор, вид, версию и обязательный состав отчёта.
        Повреждённый указатель или файл вызывает ExperimentError.
        """
        self._kind(kind)
        pointer = self.directory / f"latest_{kind}.json"
        if not pointer.exists():
            return None
        try:
            run_id = json.loads(pointer.read_text(encoding="utf-8"))["run_id"]
            if UUID(run_id).hex != run_id:
                raise ValueError("Некорректный идентификатор отчёта")
            report = json.loads((self.directory / f"{run_id}.json").read_text(encoding="utf-8"))
            if report["kind"] != kind or report["run_id"] != run_id or report["format_version"] != 1:
                raise ValueError("Некорректный формат отчёта")
            if kind == "comparison":
                if not report["model_ids"] or report["recommended_model_id"] not in report["model_ids"]:
                    raise ValueError("Некорректный состав сравнения")
            elif not report["rows"] or not report["centers"]:
                raise ValueError("Пустой результат группировки")
            return report
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            logger.exception("Не удалось открыть отчёт: вид=%s", kind)
            raise ExperimentError("Сохранённый отчёт недоступен. Запустите расчёт заново.") from error
