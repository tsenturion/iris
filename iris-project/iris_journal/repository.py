"""SQLite хранит текущую запись и неизменяемые снимки всех её редакций."""

from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from uuid import uuid4

from .domain import (
    ConflictError, FEATURE_COLUMNS, Measurements, Observation, Prediction,
    Sample, Species, StoredPrediction, ValidationError,
)


logger = logging.getLogger(__name__)
OBSERVATION_COLUMNS = (
    "sample_code", "observed_on", *FEATURE_COLUMNS, "location", "notes",
    "hypothesized_species", "hypothesis_basis", "confirmed_species", "confirmation_basis",
)


def _to_record(observation: Observation) -> dict:
    """Разворачивает наблюдение в поля SQLite: измерения отдельно, дата в ISO, номер без пробелов."""
    record = asdict(observation)
    record.update(record.pop("measurements"))
    record["sample_code"] = observation.sample_code.strip()
    record["observed_on"] = observation.observed_on.isoformat()
    return record


def _to_sample(row: sqlite3.Row) -> Sample:
    """Восстанавливает сохранённую запись SQLite в типизированное наблюдение с редакцией."""
    record = dict(row)
    observation = Observation(
        sample_code=record["sample_code"],
        observed_on=date.fromisoformat(record["observed_on"]),
        measurements=Measurements(**{key: record[key] for key in FEATURE_COLUMNS}),
        location=record["location"],
        notes=record["notes"],
        hypothesized_species=Species(record["hypothesized_species"]) if record["hypothesized_species"] else None,
        hypothesis_basis=record["hypothesis_basis"],
        confirmed_species=Species(record["confirmed_species"]) if record["confirmed_species"] else None,
        confirmation_basis=record["confirmation_basis"],
    )
    return Sample(record["id"], record["version"], record["created_at"], record["updated_at"], observation)


def _to_prediction(row: sqlite3.Row) -> StoredPrediction:
    """Восстанавливает прогноз из JSON-полей SQLite и связывает его с редакцией образца."""
    return StoredPrediction(
        id=row["id"], sample_id=row["sample_id"], sample_version=row["sample_version"],
        prediction=Prediction(
            model_id=row["model_id"], algorithm=row["algorithm"], created_at=row["created_at"],
            measurements=Measurements(**json.loads(row["measurements_json"])),
            predicted_species=Species(row["predicted_species"]),
            probabilities={Species(key): value for key, value in json.loads(row["probabilities_json"]).items()},
        ),
    )


class SampleRepository:
    """Хранит наблюдения, неизменяемые редакции и прогнозы в SQLite с короткими соединениями.

    Каждый вызов открывает своё соединение. Составные сохранения выполняются
    в транзакциях, а внешние ключи удерживают связь прогноза с исходными измерениями.
    """

    def __init__(self, path: Path):
        """Открывает хранилище в WAL и создаёт или обновляет поддерживаемую схему в транзакции."""
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            # Схема и её номер публикуются вместе; неудачная миграция откатит оба.
            connection.execute("BEGIN IMMEDIATE")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2):
                raise RuntimeError("Версия базы данных не поддерживается этим приложением.")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    sample_code TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    observed_on TEXT NOT NULL,
                    sepal_length_cm REAL NOT NULL CHECK(sepal_length_cm > 0),
                    sepal_width_cm REAL NOT NULL CHECK(sepal_width_cm > 0),
                    petal_length_cm REAL NOT NULL CHECK(petal_length_cm > 0),
                    petal_width_cm REAL NOT NULL CHECK(petal_width_cm > 0),
                    location TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    hypothesized_species TEXT,
                    hypothesis_basis TEXT NOT NULL,
                    confirmed_species TEXT,
                    confirmation_basis TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS sample_revisions (
                    sample_id TEXT NOT NULL REFERENCES samples(id),
                    version INTEGER NOT NULL,
                    saved_at TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    PRIMARY KEY (sample_id, version)
                )
            """)
            if version < 2:
                # Ссылка на снимок сохраняет исходные измерения, даже если samples
                # уже содержит исправленную редакцию того же образца.
                connection.execute("""
                    CREATE TABLE predictions (
                        id TEXT PRIMARY KEY,
                        sample_id TEXT NOT NULL,
                        sample_version INTEGER NOT NULL,
                        model_id TEXT NOT NULL,
                        algorithm TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        predicted_species TEXT NOT NULL,
                        probabilities_json TEXT NOT NULL,
                        measurements_json TEXT NOT NULL,
                        FOREIGN KEY (sample_id, sample_version)
                            REFERENCES sample_revisions(sample_id, version)
                    )
                """)
                connection.execute(
                    "CREATE INDEX predictions_sample_idx ON predictions(sample_id, created_at)"
                )
                connection.execute("PRAGMA user_version = 2")
                logger.info("Схема хранилища обновлена: %s -> 2", version)
        logger.info("Хранилище образцов открыто: %s", path)

    @contextmanager
    def _connection(self):
        """Выдаёт соединение с внешними ключами, фиксирует успех и откатывает исключения.

        Соединение всегда закрывается; ошибки SQLite записываются в диагностику.
        Такой контекст можно использовать из разных сеансов без общего открытого подключения.
        """
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        except sqlite3.IntegrityError:
            raise
        except sqlite3.Error:
            logger.exception("Ошибка операции с хранилищем образцов")
            raise
        finally:
            connection.close()

    def save(
        self,
        observation: Observation,
        *,
        sample_id: str | None = None,
        expected_version: int | None = None,
    ) -> Sample:
        """Сохраняет наблюдение и снимок новой редакции под блокировкой записи.

        Без sample_id создаёт запись; для правки нужны идентификатор и expected_version.
        Повторный номер или устаревшая редакция вызывают ConflictError без потери данных.
        """
        observation.validate()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            saved = self._save_in_connection(connection, observation, sample_id, expected_version)
        logger.info("Образец сохранён: id=%s, версия=%s", saved.id, saved.version)
        return saved

    def _save_in_connection(
        self, connection: sqlite3.Connection, observation: Observation,
        sample_id: str | None = None, expected_version: int | None = None,
        known_codes: dict[str, str] | None = None,
    ) -> Sample:
        """Сохраняет запись и снимок внутри уже открытой транзакции вызывающего кода.

        Проверяет номер через casefold и редакцию через условный UPDATE.
        known_codes переиспользуется и дополняется при пакетной загрузке;
        метод самостоятельно не фиксирует транзакцию.
        """
        record = _to_record(observation)
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        is_new = sample_id is None
        sample_id = sample_id or str(uuid4())
        if known_codes is None:
            # SQLite NOCASE учитывает прежде всего ASCII. casefold нужен также
            # для кириллицы; вызывающий код уже удерживает блокировку записи.
            known_codes = {row["sample_code"].casefold(): row["id"] for row in connection.execute("SELECT id, sample_code FROM samples")}
        code = record["sample_code"].casefold()
        if code in known_codes and known_codes[code] != sample_id:
            logger.warning("Отклонено сохранение образца с повторным номером")
            raise ConflictError(f"Образец с номером «{record['sample_code']}» уже существует.")
        try:
            if is_new:
                version = 1
                columns = ("id", "version", "created_at", "updated_at", *OBSERVATION_COLUMNS)
                values = dict(id=sample_id, version=version, created_at=now, updated_at=now, **record)
                placeholders = ", ".join(f":{column}" for column in columns)
                connection.execute(
                    f"INSERT INTO samples ({', '.join(columns)}) VALUES ({placeholders})", values,
                )
            else:
                if expected_version is None:
                    raise ConflictError("Для редактирования необходимо указать версию записи.")
                version = expected_version + 1
                assignments = ", ".join(f"{column} = :{column}" for column in OBSERVATION_COLUMNS)
                # Проверка версии входит в сам UPDATE, поэтому между проверкой
                # и записью невозможно незаметно затереть чужую редакцию.
                changed = connection.execute(
                    f"UPDATE samples SET {assignments}, version = :version, updated_at = :updated_at "
                    "WHERE id = :id AND version = :expected_version",
                    dict(record, version=version, updated_at=now, id=sample_id, expected_version=expected_version),
                ).rowcount
                if changed != 1:
                    logger.warning("Конфликт редакций: образец=%s, версия=%s", sample_id, expected_version)
                    raise ConflictError("Запись уже изменена. Откройте её заново и повторите правку.")
            # Текущая запись и снимок относятся к одной транзакции: история
            # не останется без записи, а запись — без соответствующей редакции.
            connection.execute(
                "INSERT INTO sample_revisions VALUES (?, ?, ?, ?)",
                (sample_id, version, now, json.dumps(record, ensure_ascii=False)),
            )
            row = connection.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
        except sqlite3.IntegrityError as error:
            if "samples.sample_code" in str(error):
                logger.warning("Отклонено сохранение образца с повторным номером")
                raise ConflictError(f"Образец с номером «{observation.sample_code.strip()}» уже существует.") from error
            logger.exception("Нарушено ограничение целостности хранилища образцов")
            raise
        known_codes[code] = sample_id
        return _to_sample(row)

    def get(self, sample_id: str) -> Sample | None:
        """Возвращает текущую редакцию образца по идентификатору или None, если его нет."""
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
        return _to_sample(row) if row is not None else None

    def list_samples(self) -> list[Sample]:
        """Возвращает текущие записи в порядке последнего изменения, начиная с новых."""
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM samples ORDER BY updated_at DESC, id").fetchall()
        return [_to_sample(row) for row in rows]

    def history(self, sample_id: str) -> list[dict]:
        """Возвращает полные снимки редакций выбранного образца от новых к старым."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT version, saved_at, snapshot_json FROM sample_revisions "
                "WHERE sample_id = ? ORDER BY version DESC", (sample_id,),
            ).fetchall()
        return [
            dict(version=row["version"], saved_at=row["saved_at"], **json.loads(row["snapshot_json"]))
            for row in rows
        ]

    def save_prediction(self, sample: Sample, prediction: Prediction) -> StoredPrediction:
        """Сохраняет прогноз для указанного снимка образца без изменения наблюдения."""
        prediction.validate()
        with self._connection() as connection:
            saved = self._prediction_in_connection(connection, sample, prediction)
        logger.info(
            "Прогноз сохранён: образец=%s, редакция=%s, модель=%s",
            sample.id, sample.version, prediction.model_id,
        )
        return saved

    def _prediction_in_connection(
        self, connection: sqlite3.Connection, sample: Sample, prediction: Prediction,
    ) -> StoredPrediction:
        """Вставляет прогноз в текущую транзакцию после сверки с неизменяемым снимком.

        Сравнивает измерения с sample_revisions, а не с редактируемой таблицей samples.
        Это позволяет сохранить результат старой редакции после параллельной правки.
        """
        identifier = str(uuid4())
        revision = connection.execute(
            "SELECT snapshot_json FROM sample_revisions WHERE sample_id = ? AND version = ?",
            (sample.id, sample.version),
        ).fetchone()
        if revision is None:
            raise ConflictError("Редакция образца для сохранения прогноза не найдена.")
        snapshot = json.loads(revision["snapshot_json"])
        expected = Measurements(**{name: snapshot[name] for name in FEATURE_COLUMNS})
        if prediction.measurements != expected:
            raise ValidationError("Измерения прогноза не соответствуют выбранной редакции образца.")
        connection.execute(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                identifier, sample.id, sample.version, prediction.model_id,
                prediction.algorithm, prediction.created_at, prediction.predicted_species.value,
                json.dumps(prediction.probabilities), json.dumps(asdict(prediction.measurements)),
            ),
        )
        return StoredPrediction(identifier, sample.id, sample.version, prediction)

    def save_new_predictions(self, items: list[tuple[Observation, Prediction]]) -> list[StoredPrediction]:
        """Атомарно сохраняет новые образцы, их редакции и соответствующие прогнозы.

        До записи проверяет все значения, совпадение измерений и номера внутри пакета.
        Затем под общей блокировкой сверяет существующие номера; любая ошибка
        откатывает весь пакет, включая уже вставленные строки.
        """
        if not items:
            raise ValidationError("Нет образцов для сохранения.")
        codes = set()
        for observation, prediction in items:
            observation.validate()
            prediction.validate()
            if observation.measurements != prediction.measurements:
                raise ValidationError("Измерения образца и прогноза должны совпадать.")
            code = observation.sample_code.strip().casefold()
            if code in codes:
                raise ConflictError(f"Повторный номер образца: {observation.sample_code}.")
            codes.add(code)
        saved = []
        with self._connection() as connection:
            # Блокировка охватывает сверку номеров и весь пакет. Исключение
            # внутри контекста откатывает и уже вставленные образцы с прогнозами.
            connection.execute("BEGIN IMMEDIATE")
            known_codes = {row["sample_code"].casefold(): row["id"] for row in connection.execute("SELECT id, sample_code FROM samples")}
            for observation, prediction in items:
                sample = self._save_in_connection(connection, observation, known_codes=known_codes)
                saved.append(self._prediction_in_connection(connection, sample, prediction))
        logger.info("Образцы с прогнозами сохранены одной транзакцией: записей=%s", len(saved))
        return saved

    def backup_bytes(self) -> bytes:
        """Возвращает согласованный SQLite-файл журнала через онлайн-API backup.

        Копия включает изменения WAL, редакции и прогнозы, но не внешние модели.
        Временная база закрывается перед чтением байтов и удаляется после получения копии.
        """
        with TemporaryDirectory(prefix="iris-backup-") as folder:
            path = Path(folder) / "journal.sqlite3"
            with self._connection() as source, closing(sqlite3.connect(path)) as target:
                # Прямое копирование файла не учло бы актуальные страницы WAL.
                source.backup(target)
            # Оба соединения закрыты до чтения и удаления временного файла в Windows.
            data = path.read_bytes()
        logger.info("Подготовлена резервная копия журнала: байт=%s", len(data))
        return data

    def list_predictions(self, sample_id: str) -> list[StoredPrediction]:
        """Возвращает прогнозы по убыванию редакции, внутри неё — от поздних вставок к ранним."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM predictions WHERE sample_id = ? ORDER BY sample_version DESC, rowid DESC",
                (sample_id,),
            ).fetchall()
        return [_to_prediction(row) for row in rows]

    def latest_predictions(self) -> dict[str, StoredPrediction]:
        """Возвращает по одному последнему прогнозу для самой новой доступной редакции каждого образца.

        Поздняя вставка результата старой редакции не скрывает результат новой.
        Совпадение с текущей редакцией отдельно проверяется интерфейсом и экспортом.
        """
        with self._connection() as connection:
            # ROW_NUMBER выбирает одну строку внутри каждого образца. Сначала
            # важна редакция, затем порядок вставки, а не время старого прогноза.
            rows = connection.execute("""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY sample_id ORDER BY sample_version DESC, rowid DESC
                    ) AS position FROM predictions
                ) WHERE position = 1
            """).fetchall()
        return {row["sample_id"]: _to_prediction(row) for row in rows}
