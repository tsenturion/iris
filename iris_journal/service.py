"""Сохранение наблюдений и экспорт журнала с отдельными машинными прогнозами."""

import logging

from .domain import Observation, Sample, Species
from .csv_tools import encode_csv
from .repository import OBSERVATION_COLUMNS, SampleRepository, _to_record


logger = logging.getLogger(__name__)


class JournalService:
    """Операции сохранения и экспорта, общие для интерфейса журнала."""

    def __init__(self, repository: SampleRepository):
        """Подключает репозиторий, не открывая дополнительное постоянное соединение."""
        self.repository = repository

    def save(self, observation: Observation, editing: Sample | None = None) -> Sample:
        """Передаёт наблюдение на создание или правку с ожидаемой редакцией выбранного снимка."""
        return self.repository.save(
            observation,
            sample_id=editing.id if editing else None,
            expected_version=editing.version if editing else None,
        )

    def export_csv(self, samples: list[Sample]) -> bytes:
        """Экспортирует выбранные записи с человеческими полями и отдельным последним прогнозом.

        Для прогноза указывает редакцию, актуальность, модель и оценки видов.
        Записи без результата имеют пустые машинные столбцы; текстовые формулы экранируются.
        """
        columns = (
            "id", "version", "created_at", "updated_at", *OBSERVATION_COLUMNS,
            "predicted_species", "prediction_sample_version", "prediction_is_current",
            "prediction_model_id", "prediction_algorithm", "prediction_created_at",
            *(f"score_{species.value}" for species in Species),
        )
        predictions = self.repository.latest_predictions()
        rows = []
        for sample in samples:
            row = dict(
                id=sample.id, version=sample.version, created_at=sample.created_at,
                updated_at=sample.updated_at, **_to_record(sample.observation),
            )
            if saved := predictions.get(sample.id):
                prediction = saved.prediction
                row.update(
                    predicted_species=prediction.predicted_species.value,
                    prediction_sample_version=saved.sample_version,
                    prediction_is_current=saved.sample_version == sample.version,
                    prediction_model_id=prediction.model_id,
                    prediction_algorithm=prediction.algorithm,
                    prediction_created_at=prediction.created_at,
                    **{f"score_{species.value}": prediction.probabilities[species] for species in Species},
                )
            rows.append(row)
        logger.info("Подготовлен экспорт журнала: записей=%s", len(samples))
        return encode_csv(columns, rows)
