"""Проверки общего разбиения, подбора без утечки, отчётов и группировки без меток видов."""

from dataclasses import replace
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

from iris_journal.dataset import load_reference_dataset
from iris_journal.domain import FEATURE_COLUMNS, Measurements, Species
from iris_journal.experiments import cluster_reference_samples, ExperimentError, ExperimentStore
from iris_journal.logging_setup import configure_logging
from iris_journal.model_service import ModelService
from iris_journal.models import ModelStore
from iris_journal.repository import SampleRepository
from iris_journal.settings import PROJECT_ROOT
from iris_journal.training import compare_classifiers, split_reference_samples


class MethodsTests(unittest.TestCase):
    """Проверяет корректность сравнения методов и независимого сохранения экспериментов."""

    @classmethod
    def setUpClass(cls):
        """Один раз сравнивает три метода на общем Iris для повторного использования результатов."""
        cls.dataset_path = PROJECT_ROOT / "data" / "datasets" / "iris.csv"
        cls.samples = load_reference_dataset(cls.dataset_path)
        cls.compared = compare_classifiers(cls.samples)

    def setUp(self):
        """Создаёт временные хранилища и исходную активную модель для проверки переключений."""
        self.folder = TemporaryDirectory()
        self.root = Path(self.folder.name)
        configure_logging(self.root / "logs")
        self.store = ModelStore(self.root / "artifacts")
        self.repository = SampleRepository(self.root / "journal.sqlite3")
        self.service = ModelService(self.store, self.dataset_path, self.repository)
        self.initial = self.store.save(self.compared.results[0], self.dataset_path.name)
        self.store.activate(self.initial.model_id)

    def tearDown(self):
        """Закрывает файлы логирования и освобождает временные данные проверки."""
        logger = logging.getLogger("iris_journal")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        self.folder.cleanup()

    def publish_comparison(self):
        """Публикует готовое сравнение через сервис, не повторяя поиск параметров."""
        with patch("iris_journal.model_service.compare_classifiers", return_value=self.compared):
            return self.service.compare()

    def test_algorithms_use_identical_split_and_consistent_reports(self):
        """Сверяет строки всех алгоритмов, метрики, матрицы и списки ошибок."""
        self.assertEqual({result.algorithm for result in self.compared.results}, {"logistic_regression", "knn", "decision_tree"})
        first = self.compared.results[0]
        for result in self.compared.results:
            self.assertEqual(result.training_ids, first.training_ids)
            self.assertEqual(result.evaluation_ids, first.evaluation_ids)
            self.assertEqual(len(result.training_ids), 120)
            self.assertEqual(len(result.evaluation_ids), 30)
            self.assertEqual(tuple(result.pipeline.feature_names_in_), FEATURE_COLUMNS)
            report = result.evaluation
            matrix = np.array(report.confusion_matrix)
            self.assertEqual(int(matrix.sum()), 30)
            self.assertAlmostEqual(float(matrix.trace() / 30), report.accuracy)
            self.assertEqual(len(report.errors), 30 - int(matrix.trace()))
            self.assertEqual(sum(int(values["support"]) for values in report.per_class.values()), 30)
            self.assertEqual({row["source_id"] for row in report.errors} - set(result.evaluation_ids), set())
            self.assertEqual(result.selection["folds"], 5)

    def test_selection_and_scaling_do_not_use_test_rows(self):
        """Проверяет обучение scaler только на обучающих строках и независимость рекомендации от теста."""
        split = split_reference_samples(self.samples)
        train_ids = set(split.training.identifiers)
        seen_scaler_rows = []
        original_fit = StandardScaler.fit

        def recording_fit(scaler, features, *args, **kwargs):
            """Запоминает номера строк каждого fit scaler и затем вызывает его исходную реализацию."""
            seen_scaler_rows.append(set(features.index))
            return original_fit(scaler, features, *args, **kwargs)

        wrong_winner = (self.compared.recommended_index + 1) % 3
        # Заведомо лучший результат теста отдаём другому алгоритму: рекомендация
        # должна остаться прежней, если выбор действительно основан только на CV.
        fake_reports = [
            replace(result.evaluation, accuracy=1.0 if index == wrong_winner else 0.0,
                    macro_f1=1.0 if index == wrong_winner else 0.0)
            for index, result in enumerate(self.compared.results)
        ]
        with patch.object(StandardScaler, "fit", new=recording_fit):
            with patch("iris_journal.training.evaluate_pipeline", side_effect=fake_reports):
                compared = compare_classifiers(self.samples)
        self.assertEqual(compared.recommended_index, self.compared.recommended_index)
        self.assertNotEqual(compared.recommended_index, wrong_winner)
        self.assertTrue(seen_scaler_rows)
        self.assertTrue(all(identifiers <= train_ids for identifiers in seen_scaler_rows))
        # 96 строк — обучение в одном разбиении 4/5 от 120; 120 — итоговый refit.
        self.assertIn(96, {len(identifiers) for identifiers in seen_scaler_rows})
        self.assertIn(120, {len(identifiers) for identifiers in seen_scaler_rows})

    def test_comparison_and_all_pipelines_survive_reopening(self):
        """Проверяет повторное чтение отчёта, применение трёх конвейеров и явную активацию."""
        comparison = self.publish_comparison()
        fresh_experiments = ExperimentStore(self.root / "artifacts" / "experiments")
        self.assertEqual(fresh_experiments.latest("comparison"), comparison)
        self.assertEqual(self.store.active_info().model_id, self.initial.model_id)
        for model_id in comparison["model_ids"]:
            loaded = ModelStore(self.root / "artifacts").load(model_id)
            prediction = loaded.predict(Measurements(5.1, 3.5, 1.4, 0.2))
            self.assertEqual(prediction.predicted_species, Species.SETOSA)
            self.assertEqual(len(loaded.info.evaluation["confusion_matrix"]), 3)
            self.assertEqual(loaded.info.selection["folds"], 5)
        self.store.activate(comparison["recommended_model_id"])
        self.assertEqual(self.store.active_info().model_id, comparison["recommended_model_id"])

    def test_failed_comparison_keeps_previous_report_and_active_model(self):
        """Проверяет прежний отчёт и активную модель при сбое сохранения нового сравнения."""
        previous = self.publish_comparison()
        with patch("iris_journal.model_service.compare_classifiers", return_value=self.compared):
            with patch.object(self.store, "save", side_effect=OSError("Недоступен диск")):
                with self.assertRaises(OSError):
                    self.service.compare()
        self.assertEqual(self.service.experiments.latest("comparison"), previous)
        self.assertEqual(self.store.active_info().model_id, self.initial.model_id)

    def test_model_from_module2_remains_readable(self):
        """Проверяет загрузку описания без необязательных подробной оценки и подбора."""
        path = self.store.directory / self.initial.model_id / "model.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.pop("evaluation")
        metadata.pop("selection")
        path.write_text(json.dumps(metadata), encoding="utf-8")
        loaded = self.store.load_active()
        self.assertIsNone(loaded.info.evaluation)
        self.assertEqual(loaded.predict(Measurements(5.1, 3.5, 1.4, 0.2)).predicted_species, Species.SETOSA)

    def test_cluster_membership_does_not_depend_on_known_species(self):
        """Меняет известные виды и проверяет, что группы, центры и силуэт не изменяются."""
        original = cluster_reference_samples(self.samples)
        changed_labels = cluster_reference_samples([replace(sample, species=Species.SETOSA) for sample in self.samples])
        self.assertEqual(
            [row["cluster_id"] for row in original["rows"]],
            [row["cluster_id"] for row in changed_labels["rows"]],
        )
        self.assertEqual(original["centers"], changed_labels["centers"])
        self.assertEqual(original["silhouette"], changed_labels["silhouette"])
        self.assertEqual(changed_labels["adjusted_rand"], 0.0)

    def test_clustering_is_saved_separately_from_active_classifier(self):
        """Проверяет сохранение K-means как отчёта без замены активного классификатора."""
        report = self.service.cluster()
        self.assertEqual(len(report["rows"]), 150)
        self.assertEqual({row["cluster_id"] for row in report["rows"]}, {0, 1, 2})
        self.assertTrue(-1 <= report["silhouette"] <= 1)
        self.assertTrue(-1 <= report["adjusted_rand"] <= 1)
        self.assertEqual(self.service.experiments.latest("clustering"), report)
        self.assertEqual(self.store.active_info().model_id, self.initial.model_id)

    def test_corrupt_report_pointer_is_reported(self):
        """Проверяет отказ указателю с произвольным путём вместо канонического UUID отчёта."""
        self.publish_comparison()
        pointer = self.service.experiments.directory / "latest_comparison.json"
        pointer.write_text('{"run_id": "../../invalid"}', encoding="utf-8")
        with self.assertRaises(ExperimentError):
            self.service.experiments.latest("comparison")
