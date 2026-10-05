"""Проверки форм журнала, сводки, обучения и просмотра сохранённых экспериментов в Streamlit."""

import logging
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from iris_journal.domain import Species
from iris_journal.repository import SampleRepository
from iris_journal.models import ModelStore
from iris_journal.settings import PROJECT_ROOT


class InterfaceTests(unittest.TestCase):
    """Проверяет разделы интерфейса в отдельном размещении данных без изменения пользовательского журнала."""

    def setUp(self):
        """Копирует Iris во временную папку, задаёт пути окружением и открывает новую форму."""
        self.folder = TemporaryDirectory()
        root = Path(self.folder.name)
        self.data_dir = root / "data"
        (self.data_dir / "datasets").mkdir(parents=True)
        shutil.copyfile(
            PROJECT_ROOT / "data" / "datasets" / "iris.csv", self.data_dir / "datasets" / "iris.csv",
        )
        self.environment = patch.dict(os.environ, {
            "IRIS_DATA_DIR": str(self.data_dir), "IRIS_LOG_DIR": str(root / "logs"),
        })
        self.environment.start()
        self.app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=30).run()
        self.app.radio("page").set_value("Новый образец").run()

    def tearDown(self):
        """Восстанавливает окружение и закрывает логи перед удалением временных данных."""
        self.environment.stop()
        logger = logging.getLogger("iris_journal")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        self.folder.cleanup()

    def fill_sample(self):
        """Заполняет номер и четыре измерения корректного первого образца в форме."""
        self.app.text_input("new:0:code").set_value("UI-001")
        for name, value in (
            ("sepal_length_cm", 5.1), ("sepal_width_cm", 3.5),
            ("petal_length_cm", 1.4), ("petal_width_cm", 0.2),
        ):
            self.app.number_input(f"new:0:{name}").set_value(value)

    def test_empty_form_is_rejected_without_crash(self):
        """Проверяет сообщение об ошибке пустого наблюдения без исключения и вставки записи."""
        self.assertFalse(self.app.exception)
        self.app.button[0].click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.error)
        self.assertEqual(SampleRepository(self.data_dir / "journal.sqlite3").list_samples(), [])

    def test_create_edit_history_and_summary(self):
        """Проверяет добавление, очистку формы, подтверждение, историю и показатели сводки."""
        self.assertFalse(self.app.exception)
        self.fill_sample()
        self.app.selectbox("new:0:hypothesis").set_value(Species.SETOSA)
        self.app.text_area("new:0:hypothesis_basis").set_value("Короткий лепесток")
        self.app.button[0].click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.success)
        self.assertEqual(self.app.text_input("new:1:code").value, "")
        self.assertIsNone(self.app.number_input("new:1:sepal_length_cm").value)
        repository = SampleRepository(self.data_dir / "journal.sqlite3")
        first = repository.list_samples()[0]
        self.app.radio("page").set_value("Журнал").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.dataframe[0].value), 1)
        self.app.selectbox("selected_sample").set_value(first.id).run()
        self.assertFalse(self.app.exception)
        prefix = f"edit:{first.id}:1"
        self.app.selectbox(f"{prefix}:confirmed").set_value(Species.SETOSA)
        self.app.text_area(f"{prefix}:confirmation_basis").set_value("Паспорт коллекции")
        next(button for button in self.app.button if button.label == "Сохранить изменения").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.success)
        self.assertEqual(repository.get(first.id).observation.confirmed_species, Species.SETOSA)
        self.assertEqual(len(repository.history(first.id)), 2)
        self.app.radio("page").set_value("Сводка").run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.metric[0].value, "1")
        self.assertEqual(self.app.metric[1].value, "1")

    def test_model_can_be_trained_and_reopened(self):
        """Проверяет обучение по кнопке и чтение той же модели и истории после нового сеанса."""
        self.app.radio("page").set_value("Модель").run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.info)
        self.app.button("train_baseline").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.success)
        self.assertEqual(self.app.metric[1].value, "120")
        self.assertEqual(self.app.metric[2].value, "30")
        reopened = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=30).run()
        reopened.radio("page").set_value("Модель").run()
        self.assertFalse(reopened.exception)
        self.assertEqual(reopened.metric[1].value, "120")
        history = next(element for element in reopened.dataframe if "Модель" in element.value.columns)
        self.assertEqual(len(history.value), 1)

    def test_missing_dataset_is_shown_without_crash(self):
        """Проверяет сообщение об отсутствии Iris при обучении без падения интерфейса."""
        (self.data_dir / "datasets" / "iris.csv").unlink()
        self.app.radio("page").set_value("Модель").run()
        self.app.button("train_baseline").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("Набор Iris не найден" in error.value for error in self.app.error))

    def test_compare_choose_and_reopen_models(self):
        """Проверяет сравнение, явный выбор классификатора и сохранение выбора между сеансами."""
        self.app.radio("page").set_value("Сравнение моделей").run()
        self.assertFalse(self.app.exception)
        self.app.button("compare_classifiers").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.success)
        comparison = next(element for element in self.app.dataframe if "Алгоритм" in element.value.columns)
        self.assertEqual(len(comparison.value), 3)
        self.assertEqual(int(comparison.value["Рекомендована"].sum()), 1)
        self.assertIsNone(ModelStore(self.data_dir / "artifacts").active_info())
        self.app.button("activate_compared_model").click().run()
        self.assertFalse(self.app.exception)
        active = ModelStore(self.data_dir / "artifacts").active_info()
        self.assertIsNotNone(active)
        reopened = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=30).run()
        reopened.radio("page").set_value("Сравнение моделей").run()
        self.assertFalse(reopened.exception)
        saved = next(element for element in reopened.dataframe if "Алгоритм" in element.value.columns)
        self.assertEqual(len(saved.value), 3)
        self.assertEqual(int(saved.value["Активна"].sum()), 1)

    def test_clustering_page_can_calculate_and_reopen_report(self):
        """Проверяет расчёт и повторное открытие группировки без активации классификатора."""
        self.app.radio("page").set_value("Сравнение моделей").run()
        self.app.button("cluster_samples").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(self.app.success)
        groups = next(element for element in self.app.dataframe if "Номер в Iris" in element.value.columns)
        self.assertEqual(len(groups.value), 150)
        self.assertIsNone(ModelStore(self.data_dir / "artifacts").active_info())
        reopened = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=30).run()
        reopened.radio("page").set_value("Сравнение моделей").run()
        self.assertFalse(reopened.exception)
        groups = next(element for element in reopened.dataframe if "Номер в Iris" in element.value.columns)
        self.assertEqual(len(groups.value), 150)
