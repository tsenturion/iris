"""Жизненный цикл модели и сохранение прогнозов без изменения выводов человека."""

import logging
from pathlib import Path

from .dataset import load_reference_dataset
from .domain import FEATURE_COLUMNS, FEATURE_LABELS, Measurements, Observation, Prediction, StoredPrediction, ValidationError
from .batch import BatchResult, ClassifiedRow, parse_measurements_csv
from .experiments import cluster_reference_samples, ExperimentStore
from .models import ModelError, ModelInfo, ModelStore
from .repository import SampleRepository
from .training import compare_classifiers, train_baseline


logger = logging.getLogger(__name__)


class ModelService:
    """Связывает локальный Iris, модели, отчёты экспериментов и пользовательский журнал."""

    def __init__(self, store: ModelStore, dataset_path: Path, repository: SampleRepository):
        """Подключает хранилища и путь к Iris, размещая отчёты рядом с артефактами моделей."""
        self.store = store
        self.dataset_path = dataset_path
        self.repository = repository
        self.experiments = ExperimentStore(store.active_path.parent / "experiments")

    def prepare(self) -> ModelInfo:
        """Возвращает готовую активную модель либо подготавливает рекомендованный классификатор.

        Доступный конвейер с подходящей средой сохраняется без повторного обучения.
        Если загрузка невозможна, сравниваются три метода и активируется рекомендация
        по кросс-валидации. Данные берутся только из локального справочного Iris.
        """
        try:
            info = self.store.load_active().info
        except ModelError:
            logger.info("Требуется подготовка активной модели")
        else:
            logger.info("Готовая модель сохранена активной: %s", info.model_id)
            return info
        comparison = self.compare()
        identifier = comparison["recommended_model_id"]
        self.store.activate(identifier)
        return self.store.get_info(identifier)

    def train(self) -> ModelInfo:
        """Обучает базовую модель на локальном Iris, сохраняет новую версию и активирует её."""
        try:
            samples = load_reference_dataset(self.dataset_path)
            result = train_baseline(samples)
            info = self.store.save(result, self.dataset_path.name)
            self.store.activate(info.model_id)
        except Exception:
            logger.exception("Не удалось завершить обучение и активацию модели")
            raise
        return info

    def compare(self) -> dict:
        """Сравнивает три метода и сохраняет их модели и общий отчёт с рекомендацией.

        Активная модель не меняется. Указатель отчёта публикуется после сохранения
        всех конвейеров, чтобы интерфейс получил полный результат сравнения.
        """
        try:
            samples = load_reference_dataset(self.dataset_path)
            compared = compare_classifiers(samples)
            models = [self.store.save(result, self.dataset_path.name) for result in compared.results]
            report = self.experiments.save("comparison", {
                "dataset_file": self.dataset_path.name,
                "model_ids": [info.model_id for info in models],
                "recommended_model_id": models[compared.recommended_index].model_id,
                "selection_metric": "f1_macro", "cv_folds": 5,
            })
        except Exception:
            logger.exception("Не удалось завершить сравнение методов")
            raise
        logger.info("Сравнение завершено: рекомендованная модель=%s", report["recommended_model_id"])
        return report

    def cluster(self) -> dict:
        """Группирует локальные справочные образцы и сохраняет отдельный отчёт K-means."""
        try:
            samples = load_reference_dataset(self.dataset_path)
            content = cluster_reference_samples(samples)
            return self.experiments.save("clustering", dict(content, dataset_file=self.dataset_path.name))
        except Exception:
            logger.exception("Не удалось завершить группировку образцов")
            raise

    def predict_measurements(self, measurements: Measurements) -> Prediction:
        """Применяет активный конвейер к проверенным измерениям без записи в журнал."""
        measurements.validate()
        return self.store.load_active().predict(measurements)

    def predict_sample(self, sample_id: str) -> StoredPrediction:
        """Определяет вид прочитанной редакции образца и сохраняет связанный с ней прогноз.

        Прогноз не меняет наблюдение или человеческие определения. Если запись
        параллельно исправлена, результат по-прежнему относится к прочитанному снимку.
        """
        sample = self.repository.get(sample_id)
        if sample is None:
            raise ValidationError("Образец для определения не найден.")
        prediction = self.predict_measurements(sample.observation.measurements)
        return self.repository.save_prediction(sample, prediction)

    def reference_ranges(self) -> dict[str, tuple[float, float]]:
        """Возвращает минимумы и максимумы четырёх признаков локального Iris.

        Если справочный файл недоступен или неверен, возвращает пустой словарь:
        отсутствие справочных предупреждений не блокирует готовый классификатор.
        """
        try:
            samples = load_reference_dataset(self.dataset_path)
        except (OSError, ValidationError):
            logger.warning("Справочные диапазоны измерений недоступны")
            return {}
        return {
            name: (min(getattr(sample.measurements, name) for sample in samples),
                   max(getattr(sample.measurements, name) for sample in samples))
            for name in FEATURE_COLUMNS
        }

    def measurement_warnings(
        self, measurements: Measurements, ranges: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[str, ...]:
        """Возвращает замечания о выходе значений за справочные диапазоны, не меняя измерения.

        Готовый ranges позволяет повторно использовать диапазоны для всего пакета.
        Выход за границы предлагает проверить единицы и не запрещает прогноз.
        """
        ranges = self.reference_ranges() if ranges is None else ranges
        warnings = []
        for name, (minimum, maximum) in ranges.items():
            if not minimum <= getattr(measurements, name) <= maximum:
                warnings.append(
                    f"{FEATURE_LABELS[name]}: значение вне диапазона Iris ({minimum:g}–{maximum:g}). Проверьте измерение и единицы."
                )
        return tuple(warnings)

    def classify_csv(self, payload: bytes) -> BatchResult:
        """Проверяет CSV целиком и определяет виды всех строк одной загруженной моделью.

        Возвращает пакет результатов для просмотра, экспорта или явного сохранения.
        Неверный файл не передаётся модели; записи в журнал здесь не создаются.
        """
        # Полная проверка предшествует загрузке конвейера: ошибочный файл
        # не получает частичных результатов и не требует доступной модели.
        rows = parse_measurements_csv(payload)
        model = self.store.load_active()
        predictions = model.predict_many([row.observation.measurements for row in rows])
        ranges = self.reference_ranges()
        result = BatchResult(model.info.model_id, tuple(
            ClassifiedRow(row.line, row.observation, prediction,
                          self.measurement_warnings(row.observation.measurements, ranges))
            for row, prediction in zip(rows, predictions)
        ))
        logger.info("Пакетное определение завершено: образцов=%s, модель=%s", len(rows), result.model_id)
        return result

    def save_new_prediction(self, observation: Observation, prediction: Prediction) -> StoredPrediction:
        """Сохраняет новый образец и его прогноз вместе в одной транзакции."""
        return self.repository.save_new_predictions([(observation, prediction)])[0]

    def save_batch(self, result: BatchResult) -> list[StoredPrediction]:
        """Сохраняет весь проверенный пакет образцов и прогнозов либо откатывает его целиком."""
        return self.repository.save_new_predictions([(row.observation, row.prediction) for row in result.rows])
