"""Проверки завершённого кейса: CSV, атомарная запись, резервная копия и сценарии интерфейса."""

from contextlib import closing, redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import date
import csv
from io import BytesIO, StringIO
import logging
import os
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from iris_journal.__main__ import main as cli_main
from iris_journal.batch import (
    BatchValidationError, MAX_CSV_BYTES, export_predictions, parse_measurements_csv, template_csv,
)
from iris_journal.dataset import load_reference_dataset
from iris_journal.domain import ConflictError, FEATURE_COLUMNS, Measurements, Species, ValidationError
from iris_journal.logging_setup import configure_logging
from iris_journal.model_service import ModelService
from iris_journal.models import ModelStore
from iris_journal.repository import SampleRepository
from iris_journal.service import JournalService
from iris_journal.settings import PROJECT_ROOT
from iris_journal.training import train_baseline


def close_test_logs():
    """Закрывает обработчики приложения, чтобы Windows разрешила удалить временные файлы."""
    logger = logging.getLogger("iris_journal")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


class CsvTests(unittest.TestCase):
    """Проверяет входные форматы CSV, ограничения и сбор всех ошибок до применения модели."""

    def test_template_and_optional_metadata(self):
        """Проверяет шаблон, дату по умолчанию и генерируемые номера при отсутствии метаданных."""
        rows = parse_measurements_csv(template_csv())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].observation.measurements, Measurements(5.1, 3.5, 1.4, 0.2))
        self.assertIsNone(rows[0].observation.confirmed_species)
        self.assertIsInstance(rows[0].observation.observed_on, date)
        payload = (",".join(FEATURE_COLUMNS) + "\n5.1,3.5,1.4,0.2\n").encode()
        self.assertTrue(parse_measurements_csv(payload)[0].observation.sample_code.startswith("CSV-"))

    def test_cp1251_decimal_comma_and_multiline_notes(self):
        """Проверяет Windows-1251, десятичную запятую и физический номер многострочной записи."""
        text = (
            "sample_code;" + ";".join(FEATURE_COLUMNS) + ";observed_on;location;notes\n"
            'РУ-1;5,1;3,5;1,4;0,2;2026-10-05;Коллекция;"Строка 1\nСтрока 2"\n'
        )
        row = parse_measurements_csv(text.encode("cp1251"))[0]
        self.assertEqual(row.line, 3)
        self.assertEqual(row.observation.sample_code, "РУ-1")
        self.assertEqual(row.observation.measurements.sepal_length_cm, 5.1)
        self.assertEqual(row.observation.notes, "Строка 1\nСтрока 2")

    def test_invalid_headers_empty_size_and_row_limit(self):
        """Проверяет ограничения байтов и строк, а также неверные и повторные столбцы."""
        header = ",".join(FEATURE_COLUMNS)
        invalid = [
            b"", b"\x00", b"x" * (MAX_CSV_BYTES + 1),
            (header + "\n").encode(), (header + ",species\n5,3,1,.2,setosa\n").encode(),
            (header + ",sepal_length_cm\n5,3,1,.2,5\n").encode(),
            b"sepal_length_cm,sepal_width_cm\n5,3\n",
        ]
        for payload in invalid:
            with self.subTest(payload=payload[:80]):
                with self.assertRaises(ValidationError):
                    parse_measurements_csv(payload)
        with patch("iris_journal.batch.MAX_BATCH_ROWS", 1), self.assertRaises(ValidationError):
            parse_measurements_csv(template_csv())

    def test_reports_all_invalid_rows_and_duplicate_codes(self):
        """Проверяет общий перечень ошибок значений, дат, формы строк и повторных номеров."""
        text = (
            "sample_code," + ",".join(FEATURE_COLUMNS) + ",observed_on\n"
            "A,5,3,1,0.2,\n"
            "a,5,3,1,0.2,\n"
            "B,nan,3,1,0.2,\n"
            "C,5,3,1,0.2,2026-99-99\n"
            "D,5,3,1\n"
        )
        with self.assertRaises(BatchValidationError) as caught:
            parse_measurements_csv(text.encode())
        self.assertEqual([issue.line for issue in caught.exception.issues], [3, 4, 5, 6])
        broken = (",".join(FEATURE_COLUMNS) + '\n"5,3,1,0.2\n').encode()
        with self.assertRaises(BatchValidationError):
            parse_measurements_csv(broken)


class ApplicationTests(unittest.TestCase):
    """Проверяет пакетный прогноз, транзакции, экспорт, резервное восстановление и команды."""

    @classmethod
    def setUpClass(cls):
        """Обучает общий базовый конвейер один раз для изолированных проверок приложения."""
        cls.dataset = PROJECT_ROOT / "data" / "datasets" / "iris.csv"
        cls.trained = train_baseline(load_reference_dataset(cls.dataset))

    def setUp(self):
        """Создаёт копию Iris, отдельный журнал, логи и активную модель во временной папке."""
        self.folder = TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.data = self.root / "data"
        (self.data / "datasets").mkdir(parents=True)
        shutil.copyfile(self.dataset, self.data / "datasets" / "iris.csv")
        configure_logging(self.root / "logs")
        self.repository = SampleRepository(self.data / "journal.sqlite3")
        self.store = ModelStore(self.data / "artifacts")
        info = self.store.save(self.trained, "iris.csv")
        self.store.activate(info.model_id)
        self.service = ModelService(self.store, self.data / "datasets" / "iris.csv", self.repository)

    def tearDown(self):
        """Закрывает диагностические файлы и удаляет только временное окружение проверки."""
        close_test_logs()
        self.folder.cleanup()

    def test_batch_uses_one_loaded_pipeline_without_training(self):
        """Проверяет одну загрузку, один пакетный predict_proba и отсутствие fit и записи в журнал."""
        model = self.store.load_active()
        with patch.object(self.store, "load_active", return_value=model) as load, patch(
            "iris_journal.models.Pipeline.fit", side_effect=AssertionError("Обучение при определении"),
        ), patch.object(model.pipeline, "predict_proba", wraps=model.pipeline.predict_proba) as predict:
            result = self.service.classify_csv(template_csv())
        load.assert_called_once()
        predict.assert_called_once()
        self.assertEqual(len(predict.call_args.args[0]), 2)
        self.assertEqual(result.rows[0].prediction.predicted_species, Species.SETOSA)
        self.assertEqual(result.rows[1].prediction.predicted_species, Species.VERSICOLOR)
        self.assertEqual(self.repository.list_samples(), [])

    def test_invalid_csv_does_not_load_model(self):
        """Запрещает загрузку конвейера и проверяет, что ошибочный CSV отклоняется раньше."""
        with patch.object(self.store, "load_active", side_effect=AssertionError("Загрузка при неверном CSV")):
            with self.assertRaises(ValidationError):
                self.service.classify_csv(b"bad\n1\n")

    def test_save_batch_and_reopen_without_human_labels(self):
        """Проверяет сохранение и повторное чтение пакета без автоматически заданных человеческих видов."""
        result = self.service.classify_csv(template_csv())
        saved = self.service.save_batch(result)
        reopened = SampleRepository(self.repository.path)
        self.assertEqual(len(reopened.list_samples()), 2)
        for item in saved:
            sample = reopened.get(item.sample_id)
            self.assertIsNone(sample.observation.hypothesized_species)
            self.assertIsNone(sample.observation.confirmed_species)
            self.assertEqual(reopened.list_predictions(sample.id), [item])
        rows = list(csv.DictReader(StringIO(export_predictions(result).decode("utf-8-sig"))))
        self.assertEqual(rows[0]["predicted_species"], "setosa")
        self.assertEqual(rows[0]["model_id"], result.model_id)
        self.assertAlmostEqual(sum(float(rows[0][f"score_{species.value}"]) for species in Species), 1)

    def test_conflict_rolls_back_every_new_sample_and_prediction(self):
        """Создаёт конфликт во второй строке и проверяет полный откат образцов, снимков и прогнозов."""
        result = self.service.classify_csv(template_csv())
        existing = self.repository.save(result.rows[1].observation)
        with self.assertRaises(ConflictError):
            self.service.save_batch(result)
        self.assertEqual(self.repository.list_samples(), [existing])
        self.assertEqual(self.repository.latest_predictions(), {})
        with closing(sqlite3.connect(self.repository.path)) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM sample_revisions").fetchone()[0], 1)

    def test_duplicate_russian_code_is_rejected_across_uploads(self):
        """Проверяет уникальность кириллического номера при разном регистре между загрузками."""
        item = self.service.classify_csv(template_csv()).rows[0]
        original = replace(item.observation, sample_code="ОБРАЗЕЦ-1")
        self.service.save_new_prediction(original, item.prediction)
        with self.assertRaises(ConflictError):
            self.service.save_new_prediction(replace(original, sample_code="образец-1"), item.prediction)
        self.assertEqual(len(self.repository.list_samples()), 1)

    def test_failed_prediction_insert_rolls_back_single_sample(self):
        """Имитирует сбой вставки прогноза и проверяет откат образца и запись причины в лог."""
        item = self.service.classify_csv(template_csv()).rows[0]
        with patch.object(self.repository, "_prediction_in_connection", side_effect=sqlite3.OperationalError("Проверка сбоя")):
            with self.assertRaises(sqlite3.OperationalError):
                self.service.save_new_prediction(item.observation, item.prediction)
        self.assertEqual(self.repository.list_samples(), [])
        self.assertEqual(self.repository.latest_predictions(), {})
        log = next((self.root / "logs").glob("iris-*.log")).read_text(encoding="utf-8")
        self.assertIn("Проверка сбоя", log)

    def test_revision_history_latest_export_and_backup(self):
        """Проверяет редакции, приоритет новых прогнозов, актуальность CSV и восстановление SQLite-копии."""
        item = self.service.classify_csv(template_csv()).rows[0]
        first_prediction = self.service.save_new_prediction(item.observation, item.prediction)
        first = self.repository.get(first_prediction.sample_id)
        current = self.repository.save(replace(
            first.observation, measurements=Measurements(6.4, 3.2, 4.5, 1.5),
            confirmed_species=Species.VERSICOLOR, confirmation_basis="Паспорт коллекции",
        ), sample_id=first.id, expected_version=first.version)
        journal = JournalService(self.repository)
        old = next(csv.DictReader(StringIO(journal.export_csv([current]).decode("utf-8-sig"))))
        self.assertEqual(old["prediction_is_current"], "False")
        new_prediction = self.service.predict_sample(first.id)
        # Поздняя вставка результата редакции 1 не должна скрыть прогноз редакции 2.
        self.repository.save_prediction(first, item.prediction)
        self.assertEqual(self.repository.list_predictions(first.id)[0], new_prediction)
        self.assertEqual(self.repository.latest_predictions()[first.id], new_prediction)
        self.assertEqual(self.repository.get(first.id), current)
        exported = next(csv.DictReader(StringIO(journal.export_csv([current]).decode("utf-8-sig"))))
        self.assertEqual(exported["prediction_sample_version"], "2")
        self.assertEqual(exported["confirmed_species"], "versicolor")
        self.assertEqual(exported["prediction_is_current"], "True")
        backup_path = self.root / "restored" / "journal.sqlite3"
        backup_path.parent.mkdir()
        backup_path.write_bytes(self.repository.backup_bytes())
        restored = SampleRepository(backup_path)
        self.assertEqual(restored.get(first.id), current)
        self.assertEqual(len(restored.history(first.id)), 2)
        self.assertEqual(len(restored.list_predictions(first.id)), 3)

    def test_measurement_warning_does_not_change_values(self):
        """Проверяет предупреждение о единицах при больших значениях без автоматического исправления."""
        values = Measurements(51, 35, 14, 2)
        warnings = self.service.measurement_warnings(values)
        self.assertGreater(len(warnings), 0)
        self.assertTrue(any("единицы" in warning for warning in warnings))
        self.assertEqual(self.service.predict_measurements(values).measurements, values)

    def test_prepare_creates_model_and_preserves_existing_choice(self):
        """Проверяет первое сравнение, повторное использование модели и восстановление повреждённого указателя."""
        fresh = ModelStore(self.root / "new-artifacts")
        service = ModelService(fresh, self.service.dataset_path, self.repository)
        info = service.prepare()
        self.assertEqual(len(fresh.list_models()), 3)
        self.assertEqual(fresh.load_active().info, info)
        with patch.object(service, "compare", side_effect=AssertionError("Повторное обучение")):
            self.assertEqual(service.prepare(), info)
        fresh.active_path.write_text("broken", encoding="utf-8")
        repaired = service.prepare()
        self.assertNotEqual(repaired.model_id, info.model_id)
        self.assertEqual(fresh.load_active().info, repaired)

    def test_cli_csv_and_protection_from_overwrite(self):
        """Проверяет CSV-команду, отсутствие записи в журнал и защиту существующего файла результатов."""
        source = self.root / "input.csv"
        destination = self.root / "result.csv"
        source.write_bytes(template_csv())
        environment = {"IRIS_DATA_DIR": str(self.data), "IRIS_LOG_DIR": str(self.root / "logs")}
        with patch.dict(os.environ, environment), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(cli_main(["predict-csv", str(source), "--output", str(destination)]), 0)
            original = destination.read_bytes()
            self.assertEqual(cli_main(["predict-csv", str(source), "--output", str(destination)]), 1)
        self.assertEqual(destination.read_bytes(), original)
        self.assertEqual(self.repository.list_samples(), [])


class PredictionInterfaceTests(unittest.TestCase):
    """Проверяет пользовательские сценарии определения, сохранения и подтверждения через AppTest."""

    def setUp(self):
        """Запускает интерфейс с копией Iris и отдельными данными в изменённом окружении."""
        self.folder = TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.data = self.root / "data"
        (self.data / "datasets").mkdir(parents=True)
        shutil.copyfile(PROJECT_ROOT / "data" / "datasets" / "iris.csv", self.data / "datasets" / "iris.csv")
        self.environment = patch.dict(os.environ, {
            "IRIS_DATA_DIR": str(self.data), "IRIS_LOG_DIR": str(self.root / "logs"),
        })
        self.environment.start()
        self.app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=30).run()
        self.repository = SampleRepository(self.data / "journal.sqlite3")

    def tearDown(self):
        """Восстанавливает окружение, закрывает логи и освобождает временные данные интерфейса."""
        self.environment.stop()
        close_test_logs()
        self.folder.cleanup()

    def prepare(self):
        """Готовит модель через начальную страницу и проверяет успешное построение интерфейса."""
        self.assertEqual(self.app.radio("page").value, "Определить вид")
        self.app.button("prepare_model").click().run()
        self.assertFalse(self.app.exception)

    def test_single_save_confirm_edit_predict_and_reopen(self):
        """Проходит определение, запись, подтверждение, обновление прогноза и повторный запуск журнала."""
        self.prepare()
        for name, value in zip(FEATURE_COLUMNS, (5.1, 3.5, 1.4, 0.2)):
            self.app.number_input(f"single:{name}").set_value(value)
        next(button for button in self.app.button if button.label == "Определить вид").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Iris setosa" in item.value for item in self.app.success))
        self.app.text_input("single_save_code").set_value("CASE-001")
        next(button for button in self.app.button if button.label == "Сохранить образец и прогноз").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.repository.list_samples()), 1)
        first = self.repository.list_samples()[0]
        self.assertEqual(len(self.repository.list_predictions(first.id)), 1)
        self.assertFalse(any(button.label == "Сохранить образец и прогноз" for button in self.app.button))
        self.app.radio("page").set_value("Журнал").run()
        self.app.selectbox("selected_sample").set_value(first.id).run()
        prefix = f"edit:{first.id}:1"
        self.app.selectbox(f"{prefix}:confirmed").set_value(Species.SETOSA)
        self.app.text_area(f"{prefix}:confirmation_basis").set_value("Паспорт коллекции")
        next(button for button in self.app.button if button.label == "Сохранить изменения").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("редакции 1" in item.value for item in self.app.warning))
        self.app.button(f"predict_sample:{first.id}").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("совпадает" in item.value for item in self.app.info))
        self.app.button("prepare_backup").click().run()
        self.assertFalse(self.app.exception)
        self.assertGreater(len(self.app.session_state["journal_backup"][1]), 0)
        reopened = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=30).run()
        reopened.radio("page").set_value("Журнал").run()
        reopened.selectbox("selected_sample").set_value(first.id).run()
        self.assertFalse(reopened.exception)
        self.assertTrue(any("совпадает" in item.value for item in reopened.info))
        self.assertEqual(len(self.repository.list_predictions(first.id)), 2)
        self.assertEqual(self.repository.get(first.id).observation.confirmed_species, Species.SETOSA)

    def test_csv_preview_save_and_reopen(self):
        """Проверяет просмотр CSV до записи, сохранение пакета и отсутствие повторной кнопки."""
        self.prepare()
        # AppTest не предоставляет ввод файла. Подменяем только источник байтов,
        # оставляя настоящие обработку CSV, кнопки, состояние сеанса и транзакции.
        with patch("iris_journal.ui_predictions.st.file_uploader", return_value=BytesIO(template_csv())):
            self.app.run()
            self.app.button("classify_csv").click().run()
            self.assertFalse(self.app.exception)
            self.assertEqual(len(self.repository.list_samples()), 0)
            self.app.button("save_batch").click().run()
            self.assertFalse(self.app.exception)
            self.assertEqual(len(self.repository.list_samples()), 2)
            self.assertFalse(any(button.key == "save_batch" for button in self.app.button))
        self.app.radio("page").set_value("Журнал").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.dataframe[0].value), 2)

    def test_empty_measurements_show_validation(self):
        """Проверяет понятный отказ пустой форме определения без аварии и записи в базу."""
        self.prepare()
        next(button for button in self.app.button if button.label == "Определить вид").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.error)
        self.assertEqual(self.repository.list_samples(), [])
