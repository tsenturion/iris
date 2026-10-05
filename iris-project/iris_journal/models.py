"""Сохранённые модели и прогнозы имеют независимый от интерфейса формат."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import platform
from uuid import UUID, uuid4

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.pipeline import Pipeline

from .domain import FEATURE_COLUMNS, Measurements, Prediction, Species
from .training import TrainingResult


logger = logging.getLogger(__name__)
MODEL_FORMAT_VERSION = 1


class ModelError(ValueError):
    """Модель, её описание или среда несовместимы с приложением либо недоступны."""

    pass


def environment_versions() -> dict[str, str]:
    """Возвращает версии Python и библиотек, от которых зависит сохранённый конвейер."""
    return {
        "python": platform.python_version(), "scikit-learn": sklearn.__version__,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__, "joblib": joblib.__version__,
    }


@dataclass(frozen=True)
class ModelInfo:
    """Описание версии модели для воспроизводимости, проверки среды и показа качества.

    Содержит признаки, классы, параметры, строки обучения и теста.
    Подробная оценка и подбор необязательны для ранее сохранённых описаний.
    """

    model_id: str
    created_at: str
    algorithm: str
    parameters: dict
    accuracy: float
    training_ids: tuple[int, ...]
    evaluation_ids: tuple[int, ...]
    random_state: int
    feature_columns: tuple[str, ...]
    classes: tuple[str, ...]
    environment: dict[str, str]
    dataset_file: str
    format_version: int = MODEL_FORMAT_VERSION
    evaluation: dict | None = None
    selection: dict | None = None


@dataclass(frozen=True)
class LoadedModel:
    """Проверенный обученный конвейер и его описание, готовые к применению без fit."""

    info: ModelInfo
    pipeline: Pipeline

    def predict(self, measurements: Measurements) -> Prediction:
        """Определяет один цветок через общий пакетный путь и возвращает самостоятельный прогноз."""
        return self.predict_many([measurements])[0]

    def predict_many(self, measurements: list[Measurements]) -> list[Prediction]:
        """Получает прогнозы для списка измерений одним вызовом predict_proba.

        Проверяет все значения до применения, задаёт фиксированный порядок столбцов
        и сопоставляет оценки с classes_ конвейера. Результаты получают общее время UTC;
        обучение и сохранение в этой операции не выполняются.
        """
        if not measurements:
            raise ModelError("Нет измерений для определения вида.")
        for values in measurements:
            values.validate()
        frame = pd.DataFrame([values.feature_vector() for values in measurements], columns=FEATURE_COLUMNS)
        all_scores = self.pipeline.predict_proba(frame)
        # Порядок оценок задаёт обученный конвейер, а не порядок перечисления Species.
        classes = self.pipeline.classes_
        created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        results = []
        for values, scores in zip(measurements, all_scores):
            prediction = Prediction(
                model_id=self.info.model_id, algorithm=self.info.algorithm, created_at=created_at,
                measurements=values, predicted_species=Species(str(classes[int(np.argmax(scores))])),
                probabilities={Species(str(label)): float(score) for label, score in zip(classes, scores)},
            )
            prediction.validate()
            results.append(prediction)
        logger.info("Получены прогнозы: модель=%s, образцов=%s", self.info.model_id, len(results))
        return results


def write_json_atomic(path: Path, value: dict) -> None:
    """Публикует JSON заменой временного файла в той же папке.

    Читатель получает прежний или новый полный документ вместо частично записанного.
    Родительская папка должна существовать; временный файл убирается и при ошибке.
    """
    # Та же папка обеспечивает замену в одной файловой системе; уникальное имя
    # не даёт одновременным сохранениям разделить один временный файл.
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ModelStore:
    """Хранилище независимых версий конвейеров и атомарного указателя активной модели."""

    def __init__(self, artifacts_dir: Path):
        """Задаёт папку версий и путь активного указателя внутри каталога артефактов."""
        self.directory = artifacts_dir / "models"
        self.active_path = artifacts_dir / "active_model.json"

    def _model_directory(self, model_id: str) -> Path:
        """Возвращает папку канонического UUID модели и отклоняет произвольные пути."""
        try:
            if UUID(model_id).hex != model_id:
                raise ValueError
        except (ValueError, TypeError, AttributeError) as error:
            raise ModelError("Некорректный идентификатор модели.") from error
        return self.directory / model_id

    def save(self, result: TrainingResult, dataset_file: str) -> ModelInfo:
        """Сохраняет обученный конвейер и его описание как новую самостоятельную версию.

        Проверяет разбиение и признаки, добавляет сведения о среде и оценке.
        Возвращает ModelInfo; активация выполняется отдельно после успешной загрузки.
        """
        if not math.isfinite(result.accuracy) or not 0 <= result.accuracy <= 1:
            raise ModelError("Некорректный результат проверки модели.")
        if not result.training_ids or not result.evaluation_ids or set(result.training_ids) & set(result.evaluation_ids):
            raise ModelError("Обучающие и проверочные образцы должны быть разделены.")
        if tuple(result.pipeline.feature_names_in_) != FEATURE_COLUMNS:
            raise ModelError("Признаки обученной модели не соответствуют формату проекта.")
        model_id = uuid4().hex
        info = ModelInfo(
            model_id=model_id, created_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            algorithm=result.algorithm, parameters=result.parameters, accuracy=result.accuracy,
            training_ids=result.training_ids, evaluation_ids=result.evaluation_ids,
            random_state=result.random_state, feature_columns=FEATURE_COLUMNS,
            classes=tuple(str(label) for label in result.pipeline.classes_),
            environment=environment_versions(), dataset_file=dataset_file,
            # JSON преобразует кортежи в списки. Нормализация сразу делает
            # описание в памяти таким же, как после чтения сохранённого файла.
            evaluation=json.loads(json.dumps(asdict(result.evaluation))) if result.evaluation is not None else None,
            selection=result.selection,
        )
        directory = self._model_directory(model_id)
        directory.mkdir(parents=True)
        temporary = directory / "pipeline.joblib.tmp"
        try:
            joblib.dump(result.pipeline, temporary, compress=3)
            temporary.replace(directory / "pipeline.joblib")
            write_json_atomic(directory / "model.json", asdict(info))
        except Exception:
            logger.exception("Не удалось сохранить обученную модель: %s", model_id)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        logger.info("Модель сохранена: %s", model_id)
        return info

    def get_info(self, model_id: str) -> ModelInfo:
        """Читает метаданные модели и проверяет идентификатор, формат, признаки, виды и accuracy."""
        directory = self._model_directory(model_id)
        try:
            data = json.loads((directory / "model.json").read_text(encoding="utf-8"))
            for name in ("training_ids", "evaluation_ids", "feature_columns", "classes"):
                data[name] = tuple(data[name])
            info = ModelInfo(**data)
            if info.model_id != model_id or info.format_version != MODEL_FORMAT_VERSION:
                raise ValueError("Не совпадает идентификатор или версия формата")
            if info.feature_columns != FEATURE_COLUMNS or set(info.classes) != {species.value for species in Species}:
                raise ValueError("Не совпадают признаки или виды")
            if not math.isfinite(info.accuracy) or not 0 <= info.accuracy <= 1:
                raise ValueError("Некорректная доля верных ответов")
            return info
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.exception("Не удалось прочитать описание модели: %s", model_id)
            raise ModelError("Не удалось прочитать сохранённую модель. Обучите модель заново.") from error

    def active_info(self) -> ModelInfo | None:
        """Читает описание активной версии или возвращает None при отсутствии указателя."""
        if not self.active_path.exists():
            return None
        try:
            active = json.loads(self.active_path.read_text(encoding="utf-8"))
            return self.get_info(active["model_id"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            logger.exception("Не удалось открыть активную модель")
            raise ModelError("Активная модель недоступна. Обучите модель заново.") from error

    def load(self, model_id: str) -> LoadedModel:
        """Загружает локальный конвейер после проверки метаданных и совпадения версий среды.

        Сверяет тип, признаки и порядок классов конвейера с описанием.
        Файлы joblib должны происходить из этого приложения: десериализация не является
        безопасной проверкой чужой модели. При несовместимости вызывает ModelError.
        """
        info = self.get_info(model_id)
        if info.environment != environment_versions():
            logger.warning("Среда сохранённой модели изменилась: %s", model_id)
            raise ModelError("Версии зависимостей изменились. Обучите модель заново.")
        # Проверка метаданных не делает pickle безопасным. Здесь допустимы только
        # конвейеры, созданные приложением в его локальном хранилище.
        try:
            pipeline = joblib.load(self._model_directory(model_id) / "pipeline.joblib")
            if not isinstance(pipeline, Pipeline) or not callable(getattr(pipeline, "predict_proba", None)):
                raise ValueError("Некорректный конвейер")
            if tuple(pipeline.feature_names_in_) != info.feature_columns:
                raise ValueError("Некорректные признаки конвейера")
            if tuple(str(label) for label in pipeline.classes_) != info.classes:
                raise ValueError("Некорректные виды конвейера")
        except Exception as error:
            logger.exception("Не удалось загрузить конвейер модели: %s", model_id)
            raise ModelError("Файл обученной модели недоступен или повреждён. Обучите модель заново.") from error
        logger.info("Конвейер модели загружен: %s", model_id)
        return LoadedModel(info, pipeline)

    def activate(self, model_id: str) -> None:
        """Проверяет загрузку выбранной модели и затем атомарно переключает активный указатель."""
        # Ошибка загрузки возникает до замены указателя и сохраняет прежний выбор.
        self.load(model_id)
        self.active_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.active_path, {"model_id": model_id})
        logger.info("Активная модель изменена: %s", model_id)

    def load_active(self) -> LoadedModel:
        """Возвращает готовую активную модель или поясняющую ошибку при отсутствии обучения."""
        info = self.active_info()
        if info is None:
            raise ModelError("Модель ещё не обучена. Откройте раздел «Модель» и запустите обучение.")
        return self.load(info.model_id)

    def list_models(self) -> list[ModelInfo]:
        """Возвращает доступные описания от новых к старым, записывая пропуски повреждённых в лог."""
        if not self.directory.exists():
            return []
        result = []
        for path in self.directory.glob("*/model.json"):
            try:
                result.append(self.get_info(path.parent.name))
            except ModelError:
                logger.warning("Пропущено недоступное описание модели: %s", path.parent.name)
        return sorted(result, key=lambda info: info.created_at, reverse=True)
