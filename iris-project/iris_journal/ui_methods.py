"""Сравнение классификаторов, разбор ошибок и группировка коллекции."""

from dataclasses import asdict
import json
import logging
from typing import Callable

import pandas as pd
import streamlit as st

from .domain import FEATURE_COLUMNS, FEATURE_LABELS, Species, ValidationError
from .experiments import ExperimentError
from .model_service import ModelService
from .models import ModelError, ModelInfo
from .training import ALGORITHM_LABELS


logger = logging.getLogger(__name__)


def render_evaluation(info: ModelInfo) -> None:
    """Показывает сохранённые метрики, матрицу, ошибки, подбор параметров и JSON-экспорт модели."""
    report = info.evaluation
    if report is None:
        return
    columns = st.columns(3)
    columns[0].metric("Precision, среднее по видам", f"{report['macro_precision']:.3f}")
    columns[1].metric("Recall, среднее по видам", f"{report['macro_recall']:.3f}")
    columns[2].metric("F1, среднее по видам", f"{report['macro_f1']:.3f}")
    st.subheader("Матрица ошибок")
    labels = [Species(label).label for label in report["labels"]]
    matrix = pd.DataFrame(report["confusion_matrix"], index=labels, columns=labels)
    matrix.index.name = "Истинный вид"
    st.caption("Строка — истинный вид; столбец — предсказанный. Значения — количество тестовых образцов.")
    st.dataframe(matrix, width="stretch")
    st.subheader("Оценка по видам")
    st.dataframe(pd.DataFrame([
        {
            "Вид": Species(label).label, "Precision": values["precision"],
            "Recall": values["recall"], "F1": values["f1-score"],
            "Тестовых образцов": int(values["support"]),
        }
        for label, values in report["per_class"].items()
    ]).round(3), hide_index=True, width="stretch")
    st.subheader("Ошибочно определённые образцы")
    if report["errors"]:
        st.dataframe(pd.DataFrame([
            {
                "Номер в Iris": row["source_id"], "Истинный вид": Species(row["actual_species"]).label,
                "Предсказанный вид": Species(row["predicted_species"]).label,
                **{FEATURE_LABELS[name]: row["measurements"][name] for name in FEATURE_COLUMNS},
            }
            for row in report["errors"]
        ]), hide_index=True, width="stretch")
    else:
        st.info("На этой тестовой выборке модель определила все образцы верно.")
    if info.selection is not None:
        with st.expander("Результаты подбора параметров"):
            candidates = sorted(info.selection["candidates"], key=lambda candidate: candidate["rank"])
            st.dataframe(pd.DataFrame([
                {
                    "Параметры": json.dumps(candidate["parameters"], ensure_ascii=False),
                    "F1 на кросс-валидации": candidate["mean_score"],
                    "Разброс F1": candidate["std_score"], "Место": candidate["rank"],
                }
                for candidate in candidates
            ]).round(4), hide_index=True, width="stretch")
    st.download_button(
        "Скачать отчёт модели", data=json.dumps(asdict(info), ensure_ascii=False, indent=2),
        file_name=f"iris-model-{info.model_id[:12]}.json", mime="application/json",
        key=f"evaluation_download:{info.model_id}",
    )


def classification_tab(service: ModelService, format_time: Callable[[str], str]) -> None:
    """Запускает сравнение по кнопке и показывает последний сохранённый отчёт.

    Просмотр выбранной модели не делает её активной: переключение выполняется
    отдельной кнопкой после проверки загрузки конвейера.
    """
    st.write("Сравните три классификатора и выберите модель для определения новых ирисов.")
    st.caption("Параметры и рекомендация выбираются по F1 на пяти разбиениях 120 обучающих образцов. 30 тестовых образцов используются для итоговой оценки.")
    if st.button("Сравнить алгоритмы", type="primary", key="compare_classifiers"):
        try:
            with st.spinner("Подбор параметров и оценка алгоритмов…"):
                service.compare()
        except (ValidationError, ModelError, ExperimentError) as error:
            st.error(str(error))
        except FileNotFoundError:
            st.error("Набор Iris не найден. Восстановите файл данных.")
        except Exception:
            logger.exception("Ошибка сравнения из интерфейса")
            st.error("Не удалось завершить сравнение. Повторите попытку.")
        else:
            st.session_state["flash"] = "Сравнение завершено. Выберите модель для использования."
            st.rerun()
    try:
        comparison = service.experiments.latest("comparison")
        if comparison is None:
            st.info("Результаты появятся после запуска сравнения.")
            return
        models = [service.store.get_info(model_id) for model_id in comparison["model_ids"]]
    except (ExperimentError, ModelError) as error:
        st.warning(str(error))
        return
    try:
        active = service.store.active_info()
    except ModelError:
        active = None
    recommended_id = comparison["recommended_model_id"]
    st.caption(f"Последнее сравнение: {format_time(comparison['created_at'])}")
    st.dataframe(pd.DataFrame([
        {
            "Алгоритм": ALGORITHM_LABELS.get(info.algorithm, info.algorithm),
            "F1 на кросс-валидации": round(info.selection["mean_score"], 4),
            "Разброс F1": round(info.selection["std_score"], 4),
            "Верных на обучении": f"{info.evaluation['training_accuracy']:.1%}",
            "Верных на тесте": f"{info.accuracy:.1%}",
            "F1 на тесте": round(info.evaluation["macro_f1"], 4),
            "Рекомендована": info.model_id == recommended_id,
            "Активна": bool(active and info.model_id == active.model_id),
        }
        for info in sorted(models, key=lambda item: item.selection["mean_score"], reverse=True)
    ]), hide_index=True, width="stretch")
    by_id = {info.model_id: info for info in models}
    # Идентификатор расчёта в ключе создаёт новый выбор после сравнения:
    # сохранённое состояние старого списка моделей больше не подходит.
    selected_id = st.selectbox(
        "Модель для просмотра и использования", list(by_id),
        index=list(by_id).index(recommended_id),
        format_func=lambda model_id: ALGORITHM_LABELS[by_id[model_id].algorithm]
        + (" · рекомендована" if model_id == recommended_id else ""),
        key=f"compared_model:{comparison['run_id']}",
    )
    selected = by_id[selected_id]
    if st.button("Использовать выбранную модель", key="activate_compared_model"):
        try:
            service.store.activate(selected_id)
        except (ModelError, OSError) as error:
            st.error(str(error))
        else:
            st.session_state["flash"] = f"Активная модель: {ALGORITHM_LABELS[selected.algorithm]}."
            st.rerun()
    render_evaluation(selected)


def clustering_tab(service: ModelService, format_time: Callable[[str], str]) -> None:
    """Запускает K-means по кнопке и показывает сохранённые группы, центры, метрики и CSV."""
    st.write("Разделите образцы Iris на три группы по сходству измерений.")
    st.caption("K-means использует четыре масштабированных признака. Известные виды используются после расчёта для сопоставления групп.")
    if st.button("Сгруппировать образцы", type="primary", key="cluster_samples"):
        try:
            with st.spinner("Группировка образцов…"):
                service.cluster()
        except (ValidationError, ExperimentError) as error:
            st.error(str(error))
        except FileNotFoundError:
            st.error("Набор Iris не найден. Восстановите файл данных.")
        except Exception:
            logger.exception("Ошибка группировки из интерфейса")
            st.error("Не удалось сгруппировать образцы. Повторите попытку.")
        else:
            st.session_state["flash"] = "Группировка завершена и сохранена."
            st.rerun()
    try:
        report = service.experiments.latest("clustering")
    except ExperimentError as error:
        st.warning(str(error))
        return
    if report is None:
        st.info("Результаты появятся после запуска группировки.")
        return
    columns = st.columns(3)
    columns[0].metric("Образцов", len(report["rows"]))
    columns[1].metric("Разделение групп, силуэт", f"{report['silhouette']:.3f}")
    columns[2].metric("Сходство с видами, ARI", f"{report['adjusted_rand']:.3f}")
    st.caption("Силуэт оценивает разделение групп: от −1 до 1. ARI сравнивает группировку с известными видами: 1 — полное совпадение, около 0 — случайное соответствие. Это отдельные показатели группировки.")
    st.caption(f"Расчёт сохранён: {format_time(report['created_at'])}")
    rows = pd.DataFrame([
        {
            "Номер в Iris": row["source_id"], "Группа": f"Группа {row['cluster_id'] + 1}",
            "Известный вид": Species(row["actual_species"]).label,
            **{FEATURE_LABELS[name]: row[name] for name in FEATURE_COLUMNS},
        }
        for row in report["rows"]
    ])
    st.scatter_chart(rows, x=FEATURE_LABELS["petal_length_cm"], y=FEATURE_LABELS["petal_width_cm"], color="Группа")
    st.subheader("Состав групп")
    counts = pd.crosstab(rows["Группа"], rows["Известный вид"])
    st.dataframe(counts, width="stretch")
    st.subheader("Центры групп, см")
    st.dataframe(pd.DataFrame([
        {
            "Группа": f"Группа {row['cluster_id'] + 1}",
            **{FEATURE_LABELS[name]: row[name] for name in FEATURE_COLUMNS},
        }
        for row in report["centers"]
    ]).round(2), hide_index=True, width="stretch")
    with st.expander("Все образцы и группы"):
        st.dataframe(rows, hide_index=True, width="stretch")
    st.download_button(
        "Скачать группы в CSV", data=rows.to_csv(index=False).encode("utf-8-sig"),
        file_name="iris-groups.csv", mime="text/csv", key="clustering_download",
    )


def methods_page(service: ModelService, format_time: Callable[[str], str]) -> None:
    """Размещает сравнение классификаторов и группировку в отдельных вкладках Streamlit."""
    st.header("Сравнение моделей и группировка")
    classification, clustering = st.tabs(["Классификация", "Группировка"])
    with classification:
        classification_tab(service, format_time)
    with clustering:
        clustering_tab(service, format_time)
