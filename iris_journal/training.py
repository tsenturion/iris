"""Обучение и сравнение классификаторов с отдельной тестовой выборкой."""

from dataclasses import dataclass
import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .dataset import ReferenceSample
from .domain import FEATURE_COLUMNS, Species, ValidationError


logger = logging.getLogger(__name__)
BASELINE_ALGORITHM = "logistic_regression"
BASELINE_PARAMETERS = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}
SPLIT_RANDOM_STATE = 42
EVALUATION_FRACTION = 0.2
CV_FOLDS = 5
ALGORITHM_LABELS = {
    "logistic_regression": "Логистическая регрессия",
    "knn": "Ближайшие соседи",
    "decision_tree": "Дерево решений",
}


@dataclass(frozen=True)
class DataPartition:
    """Одна часть разбиения: признаки, известные виды и номера исходных строк."""

    features: pd.DataFrame
    labels: tuple[str, ...]
    identifiers: tuple[int, ...]


@dataclass(frozen=True)
class DatasetSplit:
    """Общее разбиение на обучение и независимый итоговый тест для всех алгоритмов."""

    training: DataPartition
    testing: DataPartition


@dataclass(frozen=True)
class EvaluationReport:
    """Метрики готовой модели, матрица ошибок, показатели по видам и неверные ответы.

    Порядок строк и столбцов матрицы задан labels; errors содержит только тестовые строки.
    training_accuracy служит диагностикой отличий качества на обучении и тесте.
    """

    accuracy: float
    training_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    labels: tuple[str, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_class: dict
    errors: tuple[dict, ...]


@dataclass(frozen=True)
class TrainingResult:
    """Обученный конвейер и сведения, необходимые для сохранения и проверки модели."""

    pipeline: Pipeline
    algorithm: str
    parameters: dict
    accuracy: float
    training_ids: tuple[int, ...]
    evaluation_ids: tuple[int, ...]
    random_state: int
    evaluation: EvaluationReport | None = None
    selection: dict | None = None


def split_reference_samples(samples: list[ReferenceSample]) -> DatasetSplit:
    """Создаёт воспроизводимое стратифицированное разбиение 80/20.

    Проверяет измерения, уникальные номера и присутствие всех видов.
    Номера остаются в индексе и метаданных, не добавляясь к четырём признакам модели.
    """
    if len(samples) < 15 or set(sample.species for sample in samples) != set(Species):
        raise ValidationError("Для обучения нужны размеченные образцы всех трёх видов.")
    if len({sample.source_id for sample in samples}) != len(samples):
        raise ValidationError("Номера образцов обучающего набора должны быть уникальными.")
    for sample in samples:
        sample.measurements.validate()
    # source_id используется для аудита разбиений, но остаётся индексом:
    # включение номера строки в признаки создало бы ложную закономерность.
    features = pd.DataFrame(
        [sample.measurements.feature_vector() for sample in samples], columns=FEATURE_COLUMNS,
        index=[sample.source_id for sample in samples],
    )
    labels = [sample.species.value for sample in samples]
    identifiers = [sample.source_id for sample in samples]
    try:
        train_x, evaluation_x, train_y, evaluation_y, train_ids, evaluation_ids = train_test_split(
            features, labels, identifiers, test_size=EVALUATION_FRACTION,
            random_state=SPLIT_RANDOM_STATE, stratify=labels,
        )
    except ValueError as error:
        raise ValidationError("Набор нельзя разделить с сохранением пропорций видов.") from error
    return DatasetSplit(
        DataPartition(train_x, tuple(train_y), tuple(train_ids)),
        DataPartition(evaluation_x, tuple(evaluation_y), tuple(evaluation_ids)),
    )


def evaluate_pipeline(pipeline: Pipeline, split: DatasetSplit) -> EvaluationReport:
    """Оценивает уже обученный конвейер на заданном разбиении без дополнительного fit.

    Итоговые метрики, матрица и список ошибок относятся к тестовой части.
    Дополнительно вычисляет accuracy обучения для сопоставления с тестом.
    """
    predicted = pipeline.predict(split.testing.features)
    labels = tuple(species.value for species in Species)
    # Явный список классов фиксирует порядок матрицы и включает виды,
    # которые модель могла ни разу не предсказать на небольшой тестовой части.
    report = classification_report(
        split.testing.labels, predicted, labels=list(labels), output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(split.testing.labels, predicted, labels=list(labels))
    errors = []
    for identifier, actual, answer, values in zip(
        split.testing.identifiers, split.testing.labels, predicted,
        split.testing.features.itertuples(index=False, name=None),
    ):
        if actual != answer:
            errors.append({
                "source_id": identifier, "actual_species": actual,
                "predicted_species": str(answer), "measurements": dict(zip(FEATURE_COLUMNS, values)),
            })
    return EvaluationReport(
        accuracy=float(accuracy_score(split.testing.labels, predicted)),
        training_accuracy=float(accuracy_score(split.training.labels, pipeline.predict(split.training.features))),
        macro_precision=float(report["macro avg"]["precision"]),
        macro_recall=float(report["macro avg"]["recall"]),
        macro_f1=float(report["macro avg"]["f1-score"]), labels=labels,
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
        per_class={label: report[label] for label in labels}, errors=tuple(errors),
    )


def train_baseline(samples: list[ReferenceSample]) -> TrainingResult:
    """Обучает StandardScaler и логистическую регрессию на обучающей части Iris.

    Возвращает конвейер и независимую тестовую оценку. Сохранение файлов
    и переключение активной модели остаются задачами ModelService.
    """
    split = split_reference_samples(samples)
    logger.info("Начато обучение базовой модели: образцов=%s", len(samples))
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**BASELINE_PARAMETERS)),
    ])
    pipeline.fit(split.training.features, split.training.labels)
    evaluation = evaluate_pipeline(pipeline, split)
    logger.info(
        "Базовая модель обучена: обучение=%s, проверка=%s, доля верных ответов=%.4f",
        len(split.training.identifiers), len(split.testing.identifiers), evaluation.accuracy,
    )
    return TrainingResult(
        pipeline=pipeline, algorithm=BASELINE_ALGORITHM, parameters=dict(BASELINE_PARAMETERS),
        accuracy=evaluation.accuracy, training_ids=split.training.identifiers,
        evaluation_ids=split.testing.identifiers, random_state=SPLIT_RANDOM_STATE,
        evaluation=evaluation,
    )


@dataclass(frozen=True)
class ComparedModels:
    """Результаты трёх классификаторов и индекс рекомендации, выбранной по кросс-валидации."""

    results: tuple[TrainingResult, ...]
    recommended_index: int


def compare_classifiers(samples: list[ReferenceSample]) -> ComparedModels:
    """Подбирает параметры трёх классификаторов и оценивает их на общем итоговом тесте.

    Поиск выполняется только на обучающей части с пятью одинаковыми разбиениями
    и f1_macro; масштабирование входит в Pipeline. Рекомендация фиксируется до
    тестовой оценки. Равные оценки разрешаются порядком алгоритмов в specifications.
    Возвращает результаты без записи файлов или активации.
    """
    split = split_reference_samples(samples)
    if min(split.training.labels.count(species.value) for species in Species) < CV_FOLDS:
        raise ValidationError("Для пяти разбиений нужно не менее пяти обучающих образцов каждого вида.")
    specifications = [
        (
            "logistic_regression",
            Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(**BASELINE_PARAMETERS))]),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        (
            "knn",
            Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier())]),
            {"classifier__n_neighbors": [3, 5, 7, 9], "classifier__weights": ["uniform", "distance"]},
        ),
        (
            "decision_tree",
            Pipeline([("classifier", DecisionTreeClassifier(random_state=SPLIT_RANDOM_STATE))]),
            {"classifier__max_depth": [2, 3, 4, None], "classifier__min_samples_leaf": [1, 3, 5]},
        ),
    ]
    # Материализуем разбиения один раз: все алгоритмы сравниваются на тех же строках.
    cv = list(StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=SPLIT_RANDOM_STATE,
    ).split(split.training.features, split.training.labels))
    candidates = []
    for algorithm, pipeline, grid in specifications:
        logger.info("Начат подбор параметров: алгоритм=%s, разбиений=%s", algorithm, CV_FOLDS)
        search = GridSearchCV(
            pipeline, grid, scoring="f1_macro", cv=cv, refit=True,
            n_jobs=1, error_score="raise", return_train_score=False,
        )
        # Pipeline обучает scaler внутри каждого разбиения. refit=True затем
        # обучает лучшую конфигурацию на всей обучающей части, сохраняя тест отдельно.
        search.fit(split.training.features, split.training.labels)
        best_index = search.best_index_
        selection = {
            "method": "stratified_cross_validation", "scoring": "f1_macro", "folds": CV_FOLDS,
            "mean_score": float(search.best_score_),
            "std_score": float(search.cv_results_["std_test_score"][best_index]),
            "candidates": [
                {
                    "parameters": {key.removeprefix("classifier__"): value for key, value in parameters.items()},
                    "mean_score": float(mean), "std_score": float(std), "rank": int(rank),
                }
                for parameters, mean, std, rank in zip(
                    search.cv_results_["params"], search.cv_results_["mean_test_score"],
                    search.cv_results_["std_test_score"], search.cv_results_["rank_test_score"],
                )
            ],
        }
        candidates.append((algorithm, search.best_estimator_, selection))
        logger.info("Подбор завершён: алгоритм=%s, F1 на кросс-валидации=%.4f", algorithm, selection["mean_score"])
    # Рекомендация фиксируется до оценки теста, чтобы итоговый тест не стал
    # ещё одной выборкой для подбора. При равенстве max оставляет первый метод.
    recommended = max(range(len(candidates)), key=lambda index: candidates[index][2]["mean_score"])
    results = []
    for algorithm, pipeline, selection in candidates:
        evaluation = evaluate_pipeline(pipeline, split)
        results.append(TrainingResult(
            pipeline=pipeline, algorithm=algorithm,
            parameters=pipeline.named_steps["classifier"].get_params(deep=False),
            accuracy=evaluation.accuracy, training_ids=split.training.identifiers,
            evaluation_ids=split.testing.identifiers, random_state=SPLIT_RANDOM_STATE,
            evaluation=evaluation, selection=selection,
        ))
        logger.info(
            "Тестовая оценка: алгоритм=%s, accuracy=%.4f, F1=%.4f, ошибок=%s",
            algorithm, evaluation.accuracy, evaluation.macro_f1, len(evaluation.errors),
        )
    return ComparedModels(tuple(results), recommended)
