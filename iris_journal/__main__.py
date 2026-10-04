"""Подготовка приложения, управление моделями и определение образцов из CSV или журнала."""

import argparse
import logging
from pathlib import Path
import sqlite3
import sys

from .domain import ValidationError
from .batch import MAX_CSV_BYTES, export_predictions
from .experiments import ExperimentError
from .logging_setup import configure_logging
from .model_service import ModelService
from .models import ModelError, ModelStore
from .repository import SampleRepository
from .settings import Settings
from .training import ALGORITHM_LABELS


def main(argv: list[str] | None = None) -> int:
    """Выполняет команду подготовки, обучения или определения из командной строки.

    argv позволяет передать аргументы явно; None использует аргументы процесса.
    Успешная операция возвращает 0, ошибка выполнения — 1 с записью в лог.
    CSV-результат записывается только в новый файл; прогноз из журнала сохраняется
    отдельно от человеческого определения.
    """
    parser = argparse.ArgumentParser(description="Управление моделью определения видов ирисов")
    commands = parser.add_subparsers(dest="command", required=True, title="Команды")
    commands.add_parser("prepare", help="Подготовить модель при первом запуске; сохранить готовую активную модель")
    commands.add_parser("train", help="Обучить, проверить и сохранить базовую модель")
    commands.add_parser("model", help="Показать активную модель")
    comparison_parser = commands.add_parser("compare", help="Сравнить классификаторы и сохранить отчёт")
    comparison_parser.add_argument(
        "--activate-recommended", action="store_true", help="Использовать рекомендованную по кросс-валидации модель",
    )
    commands.add_parser("cluster", help="Сгруппировать образцы Iris без использования меток видов")
    prediction_parser = commands.add_parser("predict", help="Определить вид сохранённого образца")
    prediction_parser.add_argument("sample_code", help="Номер образца в журнале")
    csv_parser = commands.add_parser("predict-csv", help="Определить виды из CSV и записать новый файл результатов")
    csv_parser.add_argument("input", type=Path, help="CSV с четырьмя измерениями в сантиметрах")
    csv_parser.add_argument("--output", type=Path, required=True, help="Путь к новому CSV результатов")
    arguments = parser.parse_args(argv)
    settings = Settings.from_environment()
    configure_logging(settings.log_dir)
    logger = logging.getLogger(__name__)
    try:
        repository = SampleRepository(settings.database_path)
        service = ModelService(ModelStore(settings.artifacts_dir), settings.dataset_path, repository)
        if arguments.command == "prepare":
            info = service.prepare()
            print(f"Приложение готово. Активная модель: {ALGORITHM_LABELS[info.algorithm]} ({info.model_id}).")
        elif arguments.command == "predict-csv":
            if arguments.output.exists():
                raise ValidationError("Файл результатов уже существует. Укажите новое имя.")
            if arguments.input.stat().st_size > MAX_CSV_BYTES:
                raise ValidationError("Размер CSV должен быть не более 2 МБ.")
            result = service.classify_csv(arguments.input.read_bytes())
            content = export_predictions(result)
            with arguments.output.open("xb") as output:
                output.write(content)
            print(f"Определено образцов: {len(result.rows)}. Результаты: {arguments.output.resolve()}")
        elif arguments.command == "train":
            info = service.train()
            print(f"Модель обучена и активирована: {info.model_id}")
            print(f"Обучение: {len(info.training_ids)} образцов; проверка: {len(info.evaluation_ids)}.")
            print(f"Доля верных определений при проверке: {info.accuracy:.1%}")
        elif arguments.command == "compare":
            comparison = service.compare()
            for model_id in comparison["model_ids"]:
                info = service.store.get_info(model_id)
                print(
                    f"{ALGORITHM_LABELS[info.algorithm]}: F1 на кросс-валидации {info.selection['mean_score']:.3f}; "
                    f"верных на тесте {info.accuracy:.1%}; F1 на тесте {info.evaluation['macro_f1']:.3f}"
                )
            recommended_id = comparison["recommended_model_id"]
            recommended = service.store.get_info(recommended_id)
            print(f"Рекомендована: {ALGORITHM_LABELS[recommended.algorithm]} ({recommended_id})")
            if arguments.activate_recommended:
                service.store.activate(recommended_id)
                print("Рекомендованная модель активирована.")
        elif arguments.command == "cluster":
            report = service.cluster()
            print(f"Сгруппировано {len(report['rows'])} образцов; групп: {report['clusters']}.")
            print(f"Силуэт: {report['silhouette']:.3f}; ARI: {report['adjusted_rand']:.3f}")
        elif arguments.command == "model":
            info = service.store.active_info()
            if info is None:
                raise ModelError("Модель ещё не обучена. Выполните команду train.")
            service.store.load(info.model_id)
            print(f"Активная модель: {info.model_id}")
            print(f"Алгоритм: {info.algorithm}; доля верных определений при проверке: {info.accuracy:.1%}")
        else:
            code = arguments.sample_code.strip().casefold()
            sample = next((
                sample for sample in repository.list_samples()
                if sample.observation.sample_code.casefold() == code
            ), None)
            if sample is None:
                raise ValidationError("Образец с указанным номером не найден.")
            stored = service.predict_sample(sample.id)
            result = stored.prediction
            print(f"Образец {sample.observation.sample_code}, редакция {stored.sample_version}: {result.predicted_species.label}")
            for species, score in result.probabilities.items():
                print(f"  {species.label}: {score:.1%}")
            print("Прогноз сохранён отдельно от определения человека.")
    except (ModelError, ValidationError, ExperimentError, OSError, sqlite3.Error, ValueError) as error:
        logger.exception("Команда %s не выполнена", arguments.command)
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
