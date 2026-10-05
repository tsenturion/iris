"""Проверки наблюдений, редакций, экспорта, справочного Iris и срока хранения диагностики."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import csv
from io import StringIO
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from iris_journal.dataset import load_reference_dataset
from iris_journal.domain import ConflictError, Measurements, Observation, Species, ValidationError
from iris_journal.logging_setup import DailyLogHandler
from iris_journal.repository import SampleRepository
from iris_journal.service import JournalService
from iris_journal.settings import PROJECT_ROOT


def make_observation(**changes) -> Observation:
    """Создаёт корректное наблюдение с возможностью заменить отдельные поля для проверки."""
    values = dict(
        sample_code="IR-001", observed_on=date(2026, 10, 5),
        measurements=Measurements(5.1, 3.5, 1.4, 0.2), location="Коллекция № 1",
    )
    values.update(changes)
    return Observation(**values)


class JournalTests(unittest.TestCase):
    """Проверяет сохранность наблюдений и ограничения журнала в отдельной временной базе."""

    def setUp(self):
        """Создаёт временную SQLite-базу и сервис для изоляции одной проверки."""
        self.folder = TemporaryDirectory()
        self.path = Path(self.folder.name) / "journal.sqlite3"
        self.repository = SampleRepository(self.path)
        self.service = JournalService(self.repository)

    def tearDown(self):
        """Освобождает временные файлы завершённой проверки журнала."""
        self.folder.cleanup()

    def test_reopen_preserves_observation(self):
        """Проверяет, что повторное открытие базы возвращает то же наблюдение и статус."""
        saved = self.service.save(make_observation())
        reopened = SampleRepository(self.path)
        self.assertEqual(reopened.get(saved.id), saved)
        self.assertEqual(len(reopened.list_samples()), 1)
        self.assertEqual(saved.status, "Требует определения")

    def test_hypothesis_is_not_confirmed_label(self):
        """Проверяет, что предположение человека не заполняет подтверждённый вид."""
        saved = self.service.save(make_observation(
            hypothesized_species=Species.SETOSA, hypothesis_basis="Короткий лепесток",
        ))
        self.assertIsNone(saved.observation.confirmed_species)
        self.assertEqual(saved.status, "Есть предположение")

    def test_revision_preserves_previous_measurements_and_reasoning(self):
        """Проверяет сохранение прежних измерений и обоснований в истории после правки."""
        first = self.service.save(make_observation(
            hypothesized_species=Species.VERSICOLOR, hypothesis_basis="Предварительное сравнение",
        ))
        second = self.service.save(replace(
            first.observation, measurements=Measurements(5.2, 3.5, 1.4, 0.2),
            confirmed_species=Species.SETOSA, confirmation_basis="Паспорт коллекции",
        ), first)
        history = self.repository.history(first.id)
        self.assertEqual(second.version, 2)
        self.assertEqual(second.status, "Подтверждён")
        self.assertEqual([row["version"] for row in history], [2, 1])
        self.assertEqual(history[1]["sepal_length_cm"], 5.1)
        self.assertEqual(history[1]["hypothesis_basis"], "Предварительное сравнение")
        self.assertIsNone(history[1]["confirmed_species"])
        self.assertEqual(history[0]["confirmed_species"], "setosa")

    def test_stale_edit_does_not_overwrite_new_revision(self):
        """Проверяет отказ устаревшей форме без потери более новой редакции."""
        first = self.service.save(make_observation())
        current = self.service.save(replace(first.observation, notes="Первая правка"), first)
        with self.assertRaises(ConflictError):
            self.service.save(replace(first.observation, notes="Устаревшая правка"), first)
        self.assertEqual(self.repository.get(first.id), current)
        self.assertEqual(len(self.repository.history(first.id)), 2)

    def test_duplicate_code_does_not_create_extra_sample(self):
        """Проверяет уникальность номера после удаления пробелов и без учёта регистра."""
        first = self.service.save(make_observation())
        with self.assertRaises(ConflictError):
            self.service.save(make_observation(sample_code="  ir-001  "))
        self.assertEqual(len(self.repository.list_samples()), 1)
        self.assertEqual(len(self.repository.history(first.id)), 1)

    def test_missing_confirmation_basis_is_rejected(self):
        """Проверяет запрет подтверждённого вида без независимого основания."""
        with self.assertRaises(ValidationError):
            self.service.save(make_observation(confirmed_species=Species.SETOSA))
        self.assertEqual(self.repository.list_samples(), [])

    def test_invalid_measurements_are_rejected(self):
        """Проверяет отказ отсутствующим, логическим, неположительным и нечисловым измерениям."""
        for value in (None, 0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    self.service.save(make_observation(measurements=Measurements(value, 3.5, 1.4, 0.2)))
        self.assertEqual(self.repository.list_samples(), [])

    def test_export_preserves_russian_text_and_separate_labels(self):
        """Проверяет кириллицу, многострочные заметки и разделение человеческих полей в CSV."""
        saved = self.service.save(make_observation(
            notes="Строка 1\nСтрока 2", hypothesized_species=Species.SETOSA,
            hypothesis_basis="Сравнение с коллекцией",
        ))
        exported = self.service.export_csv([saved])
        self.assertTrue(exported.startswith(b"\xef\xbb\xbf"))
        row = next(csv.DictReader(StringIO(exported.decode("utf-8-sig"))))
        self.assertEqual(row["notes"], saved.observation.notes)
        self.assertEqual(row["hypothesized_species"], "setosa")
        self.assertEqual(row["confirmed_species"], "")
        self.assertEqual(float(row["petal_width_cm"]), 0.2)

    def test_export_escapes_spreadsheet_formulas(self):
        """Проверяет, что текстовые значения с признаками формулы экранируются при экспорте."""
        saved = self.service.save(make_observation(sample_code="=1+1", notes=" @SUM(1,2)"))
        row = next(csv.DictReader(StringIO(self.service.export_csv([saved]).decode("utf-8-sig"))))
        self.assertEqual(row["sample_code"], "'=1+1")
        self.assertTrue(row["notes"].startswith("'"))


class DatasetAndLoggingTests(unittest.TestCase):
    """Проверяет целостность справочных данных и выборочную очистку суточных логов."""

    def test_downloaded_dataset_has_expected_shape_and_labels(self):
        """Проверяет размер Iris, уникальные номера, порядок измерений и баланс трёх видов."""
        rows = load_reference_dataset(PROJECT_ROOT / "data" / "datasets" / "iris.csv")
        self.assertEqual(len(rows), 150)
        self.assertEqual(len({row.source_id for row in rows}), 150)
        self.assertEqual(rows[0].measurements.feature_vector(), (5.1, 3.5, 1.4, 0.2))
        for species in Species:
            self.assertEqual(sum(row.species == species for row in rows), 50)

    def test_corrupt_dataset_is_rejected(self):
        """Проверяет отказ набору с NaN вместо измерения, без восстановления данных по догадке."""
        with TemporaryDirectory() as folder:
            path = Path(folder) / "iris.csv"
            content = (PROJECT_ROOT / "data" / "datasets" / "iris.csv").read_text(encoding="utf-8")
            path.write_text(content.replace("1,5.1,", "1,nan,", 1), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_reference_dataset(path)

    def test_log_retention_removes_only_old_application_logs(self):
        """Проверяет удаление старых логов приложения с сохранением свежих и чужих файлов."""
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            old_day = (datetime.now(timezone.utc) - timedelta(days=31)).date()
            old_log = directory / f"iris-{old_day.isoformat()}.log"
            old_log.write_text("Старый журнал", encoding="utf-8")
            recent_log = directory / f"iris-{datetime.now(timezone.utc).date().isoformat()}.log"
            recent_log.write_text("Текущий журнал\n", encoding="utf-8")
            unrelated = directory / "other.log"
            unrelated.write_text("Другой журнал", encoding="utf-8")
            handler = DailyLogHandler(directory)
            try:
                handler.emit(logging.LogRecord("iris_journal", logging.INFO, "", 0, "Проверка записи", (), None))
                self.assertFalse(old_log.exists())
                self.assertTrue(unrelated.exists())
                self.assertIn("Проверка записи", recent_log.read_text(encoding="utf-8"))
            finally:
                handler.close()
