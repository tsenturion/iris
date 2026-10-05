"""Измерения хранятся отдельно от предположений и подтверждённых определений."""

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
import math


# Один порядок нужен при обучении, загрузке CSV и применении модели: перестановка
# измерений дала бы формально допустимый вектор с другим физическим смыслом.
FEATURE_COLUMNS = (
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
)
FEATURE_LABELS = {
    "sepal_length_cm": "Длина чашелистика, см",
    "sepal_width_cm": "Ширина чашелистика, см",
    "petal_length_cm": "Длина лепестка, см",
    "petal_width_cm": "Ширина лепестка, см",
}


class ValidationError(ValueError):
    """Данные нарушают правила приложения; сообщение можно показать пользователю."""

    pass


class ConflictError(ValueError):
    """Сохранение конфликтует с существующим номером или более новой редакцией записи."""

    pass


class Species(StrEnum):
    """Три поддерживаемых вида со стабильными значениями для CSV, SQLite и моделей."""

    SETOSA = "setosa"
    VERSICOLOR = "versicolor"
    VIRGINICA = "virginica"

    @property
    def label(self) -> str:
        """Возвращает полное название вида для отображения в интерфейсе."""
        return f"Iris {self.value}"


@dataclass(frozen=True)
class Measurements:
    """Четыре измерения одного цветка в сантиметрах, неизменяемые после создания."""

    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

    def validate(self) -> None:
        """Отклоняет отсутствующие, логические, нечисловые, бесконечные и неположительные значения."""
        for name in FEATURE_COLUMNS:
            value = getattr(self, name)
            # bool наследуется от int, но True не является измерением цветка.
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValidationError(
                    f"{FEATURE_LABELS[name]}: укажите конечное число больше нуля."
                )

    def feature_vector(self) -> tuple[float, ...]:
        """Возвращает числа в общем для обучения и прогноза порядке FEATURE_COLUMNS."""
        return tuple(float(getattr(self, name)) for name in FEATURE_COLUMNS)


@dataclass(frozen=True)
class Observation:
    """Наблюдение человека: измерения, метаданные и два независимых уровня определения.

    Предполагаемый и подтверждённый виды не являются машинным прогнозом.
    Подтверждение требует основания, а заметки и основания не входят в признаки модели.
    """

    sample_code: str
    observed_on: date
    measurements: Measurements
    location: str = ""
    notes: str = ""
    hypothesized_species: Species | None = None
    hypothesis_basis: str = ""
    confirmed_species: Species | None = None
    confirmation_basis: str = ""

    def validate(self) -> None:
        """Проверяет поля наблюдения и согласованность вида с его обоснованием.

        Проверка уникальности номера и редакции выполняется репозиторием при сохранении.
        Здесь проверяются номер, дата, измерения, поддерживаемые виды и длины текстов.
        """
        if not self.sample_code.strip() or len(self.sample_code.strip()) > 100:
            raise ValidationError("Номер образца должен содержать от 1 до 100 символов.")
        if any(ord(character) < 32 for character in self.sample_code):
            raise ValidationError("Номер образца не должен содержать переносы строк или управляющие символы.")
        if not isinstance(self.observed_on, date):
            raise ValidationError("Укажите дату наблюдения.")
        self.measurements.validate()
        for species in (self.hypothesized_species, self.confirmed_species):
            if species is not None and not isinstance(species, Species):
                raise ValidationError("Выберите один из трёх поддерживаемых видов ириса.")
        if self.confirmed_species is not None and not self.confirmation_basis.strip():
            raise ValidationError("Для подтверждённого вида укажите основание определения.")
        if self.hypothesized_species is None and self.hypothesis_basis.strip():
            raise ValidationError("Выберите предполагаемый вид или очистите его обоснование.")
        if self.confirmed_species is None and self.confirmation_basis.strip():
            raise ValidationError("Выберите подтверждённый вид или очистите основание.")
        for text in (
            self.location,
            self.notes,
            self.hypothesis_basis,
            self.confirmation_basis,
        ):
            if len(text) > 5000:
                raise ValidationError("Текстовое поле должно содержать не более 5000 символов.")


@dataclass(frozen=True)
class Sample:
    """Сохранённое наблюдение с идентификатором, редакцией и временем создания и правки."""

    id: str
    version: int
    created_at: str
    updated_at: str
    observation: Observation

    @property
    def status(self) -> str:
        """Определяет статус только по человеческим полям, отдавая приоритет подтверждению."""
        if self.observation.confirmed_species is not None:
            return "Подтверждён"
        if self.observation.hypothesized_species is not None:
            return "Есть предположение"
        return "Требует определения"


@dataclass(frozen=True)
class Prediction:
    """Результат модели с исходными измерениями, оценками всех классов и временем.

    Содержит сведения о конкретной модели и не изменяет определение человека.
    Привязка к сохранённому образцу появляется в StoredPrediction.
    """

    model_id: str
    algorithm: str
    created_at: str
    measurements: Measurements
    predicted_species: Species
    probabilities: dict[Species, float]

    def validate(self) -> None:
        """Проверяет измерения, оценки классов, выбранный вид и время с часовым поясом.

        Оценки должны быть конечными, лежать в диапазоне 0–1 и в сумме давать единицу.
        Выбранный вид должен иметь максимальную оценку; сравнения учитывают округление.
        """
        self.measurements.validate()
        if not self.model_id or not self.algorithm:
            raise ValidationError("В прогнозе отсутствуют сведения о модели.")
        if not isinstance(self.predicted_species, Species) or set(self.probabilities) != set(Species):
            raise ValidationError("Прогноз должен содержать оценки всех трёх видов.")
        values = list(self.probabilities.values())
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValidationError("Оценки классов должны быть конечными числами от 0 до 1.")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValidationError("Сумма оценок классов должна равняться единице.")
        # Допуск учитывает погрешность вычислений, не требуя точного равенства float.
        if not math.isclose(self.probabilities[self.predicted_species], max(values), abs_tol=1e-10):
            raise ValidationError("Предсказанный вид не соответствует оценкам классов.")
        try:
            timestamp = datetime.fromisoformat(self.created_at)
        except ValueError as error:
            raise ValidationError("Некорректное время прогноза.") from error
        if timestamp.tzinfo is None:
            raise ValidationError("Время прогноза должно содержать часовой пояс.")


@dataclass(frozen=True)
class StoredPrediction:
    """Сохранённый прогноз, привязанный к неизменяемой редакции конкретного образца."""

    id: str
    sample_id: str
    sample_version: int
    prediction: Prediction
