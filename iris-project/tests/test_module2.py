"""Проверки базовой модели, её сохранения, применения, команд и связи прогноза с редакцией."""

from contextlib import closing, redirect_stdout
from dataclasses import replace
from datetime import date
from io import StringIO
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from iris_journal.__main__ import main as cli_main
from iris_journal.dataset import load_reference_dataset
from iris_journal.domain import Measurements, Observation, Species, ValidationError
from iris_journal.logging_setup import configure_logging
from iris_journal.model_service import ModelService
from iris_journal.models import ModelError, ModelStore
from iris_journal.repository import SampleRepository
from iris_journal.settings import PROJECT_ROOT
from iris_journal.training import train_baseline


class ModelTests(unittest.TestCase):
    """Проверяет жизненный цикл классификатора и сохранность пользовательских определений."""

    @classmethod
    def setUpClass(cls):
        """Один раз обучает общий базовый конвейер для проверок сохранения и загрузки."""
        cls.dataset_path = PROJECT_ROOT / "data" / "datasets" / "iris.csv"
        cls.samples = load_reference_dataset(cls.dataset_path)
        cls.result = train_baseline(cls.samples)

    def setUp(self):
        """Создаёт изолированные хранилища и активирует копию общего обученного конвейера."""
        self.folder = TemporaryDirectory()
        self.root = Path(self.folder.name)
        configure_logging(self.root / "logs")
        self.store = ModelStore(self.root / "artifacts")
        self.repository = SampleRepository(self.root / "journal.sqlite3")
        self.service = ModelService(self.store, self.dataset_path, self.repository)
        self.info = self.store.save(self.result, self.dataset_path.name)
        self.store.activate(self.info.model_id)

    def tearDown(self):
        """Закрывает обработчики логов перед удалением временного окружения проверки."""
        logger = logging.getLogger("iris_journal")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        self.folder.cleanup()

    def add_sample(self):
        """Сохраняет образец с разными предположением и подтверждением для проверки независимости полей."""
        return self.repository.save(Observation(
            sample_code="ML-001", observed_on=date(2026, 10, 5),
            measurements=Measurements(5.1, 3.5, 1.4, 0.2),
            hypothesized_species=Species.VERSICOLOR, hypothesis_basis="Предварительный осмотр",
            confirmed_species=Species.SETOSA, confirmation_basis="Паспорт коллекции",
        ))

    def test_scaler_is_fitted_only_on_training_rows(self):
        """Сверяет параметры scaler с обучающей частью и проверяет независимость тестовых строк."""
        self.assertEqual(len(self.result.training_ids), 120)
        self.assertEqual(len(self.result.evaluation_ids), 30)
        self.assertFalse(set(self.result.training_ids) & set(self.result.evaluation_ids))
        by_id = {sample.source_id: sample for sample in self.samples}
        features = np.array([by_id[identifier].measurements.feature_vector() for identifier in self.result.training_ids])
        scaler = self.result.pipeline.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, features.mean(axis=0))
        self.assertEqual(int(scaler.n_samples_seen_), 120)
        self.assertEqual(scaler.n_features_in_, 4)
        self.assertGreaterEqual(self.result.accuracy, 0.8)

    def test_reload_does_not_train_again_and_preserves_predictions(self):
        """Запрещает fit при загрузке и сравнивает вид и оценки до и после открытия модели."""
        measurements = Measurements(5.1, 3.5, 1.4, 0.2)
        before = self.service.predict_measurements(measurements)
        fresh_store = ModelStore(self.root / "artifacts")
        with patch("iris_journal.training.Pipeline.fit", side_effect=AssertionError("Повторное обучение")):
            after = fresh_store.load_active().predict(measurements)
        self.assertEqual(after.model_id, before.model_id)
        self.assertEqual(after.predicted_species, Species.SETOSA)
        self.assertEqual(after.probabilities, before.probabilities)
        self.assertAlmostEqual(sum(after.probabilities.values()), 1)

    def test_new_training_keeps_previous_model(self):
        """Проверяет новую активную версию и доступность ранее сохранённого конвейера."""
        trained = self.service.train()
        self.assertNotEqual(trained.model_id, self.info.model_id)
        self.assertEqual(self.store.active_info(), trained)
        self.assertEqual(len(self.store.list_models()), 2)
        self.assertEqual(self.store.load(self.info.model_id).info, self.info)

    def test_failed_training_does_not_replace_active_model(self):
        """Проверяет сохранение активного указателя при отсутствии данных для нового обучения."""
        broken = ModelService(self.store, self.root / "missing.csv", self.repository)
        with self.assertRaises(FileNotFoundError):
            broken.train()
        self.assertEqual(self.store.active_info().model_id, self.info.model_id)

    def test_prediction_is_bound_to_revision_and_keeps_human_decisions(self):
        """Проверяет неизменность прогноза прежних измерений и человеческих полей после правки."""
        sample = self.add_sample()
        stored = self.service.predict_sample(sample.id)
        self.assertEqual(self.repository.get(sample.id), sample)
        self.assertEqual(stored.sample_version, 1)
        changed = self.repository.save(
            replace(sample.observation, measurements=Measurements(6.4, 3.2, 4.5, 1.5)),
            sample_id=sample.id, expected_version=sample.version,
        )
        reopened = SampleRepository(self.repository.path)
        restored = reopened.list_predictions(sample.id)[0]
        self.assertEqual(restored, stored)
        self.assertEqual(restored.prediction.measurements, sample.observation.measurements)
        self.assertEqual(reopened.get(sample.id), changed)
        self.assertEqual(changed.observation.hypothesized_species, Species.VERSICOLOR)
        self.assertEqual(changed.observation.confirmed_species, Species.SETOSA)

    def test_prediction_with_other_measurements_cannot_be_attached(self):
        """Проверяет запрет привязки прогноза с чужими измерениями к редакции образца."""
        sample = self.add_sample()
        prediction = self.service.predict_measurements(Measurements(6.4, 3.2, 4.5, 1.5))
        with self.assertRaises(ValidationError):
            self.repository.save_prediction(sample, prediction)
        self.assertEqual(self.repository.list_predictions(sample.id), [])

    def test_migration_from_module1_preserves_samples_and_history(self):
        """Проверяет добавление таблицы прогнозов к схеме 1 с сохранением записей и снимков."""
        sample = self.add_sample()
        # Воспроизводим схему первого модуля на изолированной базе, сохраняя
        # реальные записи и снимки для проверки их сохранности после миграции.
        with closing(sqlite3.connect(self.repository.path)) as connection:
            connection.execute("DROP TABLE predictions")
            connection.execute("PRAGMA user_version = 1")
            connection.commit()
        migrated = SampleRepository(self.repository.path)
        self.assertEqual(migrated.get(sample.id), sample)
        self.assertEqual(len(migrated.history(sample.id)), 1)
        self.assertEqual(migrated.list_predictions(sample.id), [])
        with closing(sqlite3.connect(self.repository.path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)

    def test_unknown_schema_version_is_not_rewritten(self):
        """Проверяет отказ неизвестной версии базы без изменения её номера схемы."""
        with closing(sqlite3.connect(self.repository.path)) as connection:
            connection.execute("PRAGMA user_version = 99")
            connection.commit()
        with self.assertRaises(RuntimeError):
            SampleRepository(self.repository.path)
        with closing(sqlite3.connect(self.repository.path)) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 99)

    def test_changed_environment_is_reported_before_loading_pipeline(self):
        """Проверяет, что несовпадение зависимостей выявляется до десериализации joblib."""
        path = self.store.directory / self.info.model_id / "model.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["environment"]["scikit-learn"] = "0.0"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with patch("iris_journal.models.joblib.load", side_effect=AssertionError("Нельзя загружать")):
            with self.assertRaisesRegex(ModelError, "Версии зависимостей"):
                self.store.load_active()

    def test_corrupt_model_has_actionable_error(self):
        """Проверяет понятное сообщение при повреждённом файле сохранённого конвейера."""
        path = self.store.directory / self.info.model_id / "pipeline.joblib"
        path.write_bytes(b"broken model")
        with self.assertRaisesRegex(ModelError, "повреждён"):
            self.store.load_active()

    def test_cli_can_predict_and_store_result(self):
        """Проверяет команду определения сохранённого образца и отдельную запись результата."""
        sample = self.add_sample()
        with patch.dict(os.environ, {
            "IRIS_DATA_DIR": str(self.root), "IRIS_LOG_DIR": str(self.root / "logs"),
        }):
            output = StringIO()
            with redirect_stdout(output):
                code = cli_main(["predict", "ML-001"])
        self.assertEqual(code, 0)
        self.assertIn("Iris setosa", output.getvalue())
        self.assertEqual(len(self.repository.list_predictions(sample.id)), 1)
        self.assertEqual(self.repository.get(sample.id), sample)

    def test_cli_training_uses_local_dataset(self):
        """Проверяет обучение через команду на копии локального Iris и создание активной модели."""
        data_dir = self.root / "cli"
        dataset_dir = data_dir / "datasets"
        dataset_dir.mkdir(parents=True)
        shutil.copyfile(self.dataset_path, dataset_dir / "iris.csv")
        with patch.dict(os.environ, {
            "IRIS_DATA_DIR": str(data_dir), "IRIS_LOG_DIR": str(self.root / "logs"),
        }):
            with redirect_stdout(StringIO()) as output:
                code = cli_main(["train"])
            self.assertEqual(code, 0)
            self.assertIn("120", output.getvalue())
        self.assertIsNotNone(ModelStore(data_dir / "artifacts").active_info())
