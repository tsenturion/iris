"""Суточные журналы очищаются при запуске и смене даты, срок хранения — 30 дней."""

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from threading import RLock
import time


class DailyLogHandler(logging.Handler):
    """Пишет суточные UTF-8-логи в UTC и удаляет собственные журналы старше срока хранения.

    Блокировка согласует запись, смену файла и закрытие между потоками приложения.
    Чужие файлы в папке логов не участвуют в очистке.
    """

    def __init__(self, directory: Path):
        """Создаёт папку, блокировку и первый суточный файл с очисткой устаревших логов."""
        super().__init__()
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._day = ""
        self._stream = None
        self._guard = RLock()
        self._rotate()

    def _rotate(self) -> None:
        """При смене даты закрывает прежний файл, очищает старые логи и открывает текущий.

        Вызывается при создании обработчика или под блокировкой во время записи.
        Ошибки удаления отдельных файлов не мешают открыть новый журнал.
        """
        now = datetime.now(timezone.utc)
        day = now.date().isoformat()
        if day == self._day:
            return
        if self._stream is not None:
            self._stream.close()
        cutoff = now - timedelta(days=30)
        # Ограничиваем очистку именами приложения и датой в имени, чтобы
        # не затронуть чужие файлы и не зависеть от времени копирования лога.
        for path in self.directory.glob("iris-????-??-??.log"):
            try:
                log_date = datetime.strptime(path.stem[5:], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if log_date <= cutoff:
                    path.unlink()
            except (ValueError, OSError):
                continue
        self._stream = (self.directory / f"iris-{day}.log").open("a", encoding="utf-8")
        self._day = day

    def emit(self, record: logging.LogRecord) -> None:
        """Записывает событие под блокировкой и сразу сбрасывает буфер на диск."""
        try:
            with self._guard:
                self._rotate()
                self._stream.write(self.format(record) + "\n")
                self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Освобождает открытый поток под блокировкой и закрывает обработчик logging."""
        with self._guard:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        super().close()


def configure_logging(directory: Path) -> logging.Logger:
    """Настраивает общий логгер приложения и возвращает его без дублирующих обработчиков.

    Повторный вызов с той же папкой использует текущий DailyLogHandler.
    При смене папки прежний обработчик закрывается; время форматируется в UTC.
    """
    logger = logging.getLogger("iris_journal")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    resolved = directory.resolve()
    for handler in list(logger.handlers):
        if isinstance(handler, DailyLogHandler):
            if handler.directory == resolved:
                return logger
            logger.removeHandler(handler)
            handler.close()
    handler = DailyLogHandler(resolved)
    formatter = logging.Formatter("%(asctime)s UTC | %(levelname)s | %(name)s | %(message)s")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Журнал приложения открыт")
    return logger
