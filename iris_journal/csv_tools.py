"""CSV для внешних программ: UTF-8 с BOM и безопасные текстовые ячейки."""

import csv
from io import StringIO


def safe_cell(value):
    """Экранирует текст, который табличная программа может интерпретировать как формулу."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def encode_csv(columns, rows) -> bytes:
    """Записывает строки в CSV с заданным порядком столбцов и возвращает UTF-8 с BOM.

    Отсутствующие значения становятся пустыми ячейками, текст проходит safe_cell.
    BOM помогает табличным программам Windows правильно распознать кириллицу.
    """
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: safe_cell(row.get(name)) for name in columns})
    return buffer.getvalue().encode("utf-8-sig")
