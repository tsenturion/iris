"""Проверка и единый формат локального справочного набора Iris."""

import csv
from dataclasses import dataclass
from io import StringIO
import logging
from pathlib import Path

from .domain import FEATURE_COLUMNS, Measurements, Species, ValidationError


logger = logging.getLogger(__name__)
DATASET_COLUMNS = ("source_id", *FEATURE_COLUMNS, "species")
SOURCE_COLUMNS = (
    "Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species",
)


@dataclass(frozen=True)
class ReferenceSample:
    """Размеченный справочный образец: номер исходной строки, измерения и известный вид."""

    source_id: int
    measurements: Measurements
    species: Species


def _parse_rows(reader: csv.DictReader, source_format: bool) -> list[ReferenceSample]:
    """Разбирает исходный или нормализованный Iris и проверяет целостность набора.

    source_format выбирает имена столбцов и удаление префикса Iris- у вида.
    Требуются уникальные положительные номера и ровно 150 образцов, по 50 на вид.
    Ошибки значений преобразуются в ValidationError с номером строки.
    """
    expected = SOURCE_COLUMNS if source_format else DATASET_COLUMNS
    if tuple(reader.fieldnames or ()) != expected:
        raise ValidationError("Столбцы набора Iris не соответствуют формату проекта.")
    result = []
    identifiers = set()
    for line, row in enumerate(reader, start=2):
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Некорректное число полей")
            identifier = int(row[expected[0]])
            measurements = Measurements(**{
                name: float(row[column]) for name, column in zip(FEATURE_COLUMNS, expected[1:5])
            })
            measurements.validate()
            label = row[expected[-1]]
            if source_format:
                label = label.removeprefix("Iris-")
            species = Species(label)
            if identifier <= 0 or identifier in identifiers:
                raise ValueError("Некорректный или повторный номер")
            identifiers.add(identifier)
        except (ValueError, TypeError) as error:
            raise ValidationError(f"Ошибка в строке {line} набора Iris: {error}") from error
        result.append(ReferenceSample(identifier, measurements, species))
    if len(result) != 150 or any(sum(row.species == species for row in result) != 50 for species in Species):
        raise ValidationError("Ожидалось 150 образцов Iris: по 50 каждого вида.")
    return result


def load_reference_dataset(path: Path) -> list[ReferenceSample]:
    """Читает локальный нормализованный CSV и возвращает проверенные справочные образцы."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        samples = _parse_rows(csv.DictReader(stream), source_format=False)
    logger.info("Справочный набор Iris проверен: образцов=%s", len(samples))
    return samples


def normalize_source_csv(source: Path, target: Path) -> list[ReferenceSample]:
    """Преобразует CSV источника в формат проекта, сохраняя измерения и номера строк.

    Сначала проверяется весь исходный набор, затем записываются канонические
    названия столбцов и метки видов. Скачивание данных не выполняется.
    """
    with source.open(encoding="utf-8-sig", newline="") as stream:
        samples = _parse_rows(csv.DictReader(stream), source_format=True)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=DATASET_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for sample in samples:
        writer.writerow({
            "source_id": sample.source_id,
            **dict(zip(FEATURE_COLUMNS, sample.measurements.feature_vector())),
            "species": sample.species.value,
        })
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(buffer.getvalue(), encoding="utf-8")
    logger.info("Набор Iris нормализован: образцов=%s", len(samples))
    return samples
