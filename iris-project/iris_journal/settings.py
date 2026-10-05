"""Все пути вычисляются относительно проекта и не зависят от рабочей папки."""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Абсолютные каталоги данных и диагностики для одного размещения приложения."""

    data_dir: Path
    log_dir: Path

    @classmethod
    def from_environment(cls) -> "Settings":
        """Читает IRIS_DATA_DIR и IRIS_LOG_DIR либо использует каталоги рядом с проектом."""
        return cls(
            data_dir=Path(os.environ.get("IRIS_DATA_DIR", PROJECT_ROOT / "data")).resolve(),
            log_dir=Path(os.environ.get("IRIS_LOG_DIR", PROJECT_ROOT / "logs")).resolve(),
        )

    @property
    def database_path(self) -> Path:
        """Возвращает путь SQLite-журнала с наблюдениями, редакциями и прогнозами."""
        return self.data_dir / "journal.sqlite3"

    @property
    def dataset_path(self) -> Path:
        """Возвращает путь локального нормализованного Iris для обучения и справочных диапазонов."""
        return self.data_dir / "datasets" / "iris.csv"

    @property
    def artifacts_dir(self) -> Path:
        """Возвращает каталог версий моделей, активного указателя и отчётов экспериментов."""
        return self.data_dir / "artifacts"
