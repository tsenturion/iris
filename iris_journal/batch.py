"""Проверка входного CSV и результаты определения группы новых образцов."""

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from io import StringIO
import logging
from uuid import uuid4

from .csv_tools import encode_csv
from .domain import FEATURE_COLUMNS, FEATURE_LABELS, Measurements, Observation, Prediction, Species, ValidationError


logger = logging.getLogger(__name__)
MAX_CSV_BYTES = 2 * 1024 * 1024
MAX_BATCH_ROWS = 5000
OPTIONAL_COLUMNS = ("sample_code", "observed_on", "location", "notes")
IMPORT_COLUMNS = ("sample_code", *FEATURE_COLUMNS, "observed_on", "location", "notes")


@dataclass(frozen=True)
class CsvIssue:
    """Ошибка входного CSV с физическим номером строки и пояснением для пользователя."""

    line: int
    message: str


class BatchValidationError(ValidationError):
    """Содержит все найденные ошибки строк, чтобы исправить файл за один проход."""

    def __init__(self, issues: list[CsvIssue]):
        """Сохраняет перечень ошибок и формирует общее сообщение о неудачной проверке."""
        self.issues = tuple(issues)
        super().__init__(f"Ошибок в CSV: {len(issues)}. Исправьте указанные строки и повторите загрузку.")


@dataclass(frozen=True)
class InputRow:
    """Проверенное наблюдение и номер строки исходного CSV, из которой оно получено."""

    line: int
    observation: Observation


@dataclass(frozen=True)
class ClassifiedRow:
    """Наблюдение с прогнозом и предупреждениями о выходе измерений за диапазон Iris."""

    line: int
    observation: Observation
    prediction: Prediction
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class BatchResult:
    """Результаты одного пакета, полученные общей активной моделью."""

    model_id: str
    rows: tuple[ClassifiedRow, ...]


def parse_measurements_csv(payload: bytes) -> list[InputRow]:
    """Проверяет весь входной CSV и возвращает наблюдения с номерами строк.

    Принимает байты UTF-8 или Windows-1251 с четырьмя обязательными измерениями.
    Отсутствующие номера генерируются, пустые даты заменяются текущей датой.
    Ошибки структуры и ограничения размера вызывают ValidationError; ошибки
    отдельных строк собираются в BatchValidationError. Модель и база не используются.
    """
    if not payload or len(payload) > MAX_CSV_BYTES:
        raise ValidationError("Загрузите непустой CSV размером не более 2 МБ.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = payload.decode("cp1251")
        except UnicodeDecodeError as error:
            raise ValidationError("Не удалось прочитать кодировку CSV. Сохраните файл в UTF-8.") from error
    if "\x00" in text:
        raise ValidationError("Файл содержит нулевые байты. Загрузите текстовый CSV.")
    # Разделитель ищется в заголовке: десятичные запятые в данных не должны
    # превратить файл с разделителем ';' в файл с разделителем ','.
    first_line = text.splitlines()[0] if text.splitlines() else ""
    delimiters = [delimiter for delimiter in (",", ";", "\t") if delimiter in first_line]
    if not delimiters:
        raise ValidationError("В CSV не найден разделитель столбцов: запятая, точка с запятой или табуляция.")
    delimiter = max(delimiters, key=first_line.count)
    reader = csv.DictReader(StringIO(text, newline=""), delimiter=delimiter, strict=True)
    try:
        headers = [name.strip() for name in reader.fieldnames or []]
    except csv.Error as error:
        raise ValidationError("Не удалось прочитать заголовок CSV.") from error
    if len(set(headers)) != len(headers):
        raise ValidationError("Названия столбцов CSV должны быть уникальными.")
    missing = set(FEATURE_COLUMNS) - set(headers)
    unknown = set(headers) - set(FEATURE_COLUMNS) - set(OPTIONAL_COLUMNS)
    if missing:
        raise ValidationError("В CSV отсутствуют столбцы: " + ", ".join(sorted(missing)))
    if unknown:
        raise ValidationError("Неизвестные столбцы CSV: " + ", ".join(sorted(unknown)))
    reader.fieldnames = headers
    default_date = datetime.now(timezone(timedelta(hours=5))).date()
    prefix = "CSV-" + uuid4().hex[:8].upper()
    rows, issues, codes = [], [], set()
    try:
        for ordinal, row in enumerate(reader, start=1):
            if ordinal > MAX_BATCH_ROWS:
                raise ValidationError("В одной загрузке допускается не более 5000 образцов.")
            # line_num считает физические строки, включая переносы внутри кавычек.
            line = reader.line_num
            try:
                if None in row or any(value is None for value in row.values()):
                    raise ValidationError("Количество значений не соответствует заголовку.")
                measurements = {}
                for name in FEATURE_COLUMNS:
                    try:
                        measurements[name] = float(row[name].strip().replace(",", "."))
                    except ValueError as error:
                        raise ValidationError(f"{FEATURE_LABELS[name]}: укажите число.") from error
                raw_date = row.get("observed_on", "").strip()
                try:
                    observed_on = date.fromisoformat(raw_date) if raw_date else default_date
                except ValueError as error:
                    raise ValidationError("Дата наблюдения должна быть в формате ГГГГ-ММ-ДД.") from error
                observation = Observation(
                    sample_code=row.get("sample_code", "").strip() or f"{prefix}-{ordinal:04d}",
                    observed_on=observed_on, measurements=Measurements(**measurements),
                    location=row.get("location", "").strip(), notes=row.get("notes", "").strip(),
                )
                observation.validate()
                code = observation.sample_code.casefold()
                if code in codes:
                    raise ValidationError(f"Повторный номер образца: {observation.sample_code}.")
                codes.add(code)
                rows.append(InputRow(line, observation))
            except ValidationError as error:
                # Продолжаем проверять следующие строки, чтобы пользователь
                # получил весь перечень ошибок вместо последовательных отказов.
                issues.append(CsvIssue(line, str(error)))
    except csv.Error as error:
        issues.append(CsvIssue(reader.line_num, "Нарушено оформление CSV: " + str(error)))
    if issues:
        logger.warning("CSV отклонён: ошибок=%s", len(issues))
        raise BatchValidationError(issues)
    if not rows:
        raise ValidationError("CSV содержит заголовок, но не содержит образцов.")
    logger.info("CSV проверен: образцов=%s", len(rows))
    return rows


def template_csv() -> bytes:
    """Возвращает CSV-шаблон с двумя примерами измерений и всеми допустимыми столбцами."""
    return encode_csv(IMPORT_COLUMNS, [
        dict(sample_code="NEW-001", sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2),
        dict(sample_code="NEW-002", sepal_length_cm=6.4, sepal_width_cm=3.2, petal_length_cm=4.5, petal_width_cm=1.5),
    ])


def export_predictions(result: BatchResult) -> bytes:
    """Формирует CSV результатов с измерениями, оценками классов, моделью и замечаниями.

    Человеческие определения не добавляются: машинный результат не подтверждает вид.
    Текстовые ячейки экранируются общей функцией экспорта.
    """
    columns = (
        *IMPORT_COLUMNS, "predicted_species", *(f"score_{species.value}" for species in Species),
        "model_id", "algorithm", "predicted_at", "measurement_warnings",
    )
    rows = []
    for item in result.rows:
        observation, prediction = item.observation, item.prediction
        rows.append({
            "sample_code": observation.sample_code, "observed_on": observation.observed_on.isoformat(),
            "location": observation.location, "notes": observation.notes,
            **dict(zip(FEATURE_COLUMNS, observation.measurements.feature_vector())),
            "predicted_species": prediction.predicted_species.value,
            **{f"score_{species.value}": prediction.probabilities[species] for species in Species},
            "model_id": prediction.model_id, "algorithm": prediction.algorithm,
            "predicted_at": prediction.created_at, "measurement_warnings": " ".join(item.warnings),
        })
    return encode_csv(columns, rows)
