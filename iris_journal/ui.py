"""Пользовательский интерфейс журнала наблюдений."""

from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import sqlite3

import pandas as pd
import streamlit as st

from .domain import (
    ConflictError, FEATURE_COLUMNS, FEATURE_LABELS, Measurements,
    Observation, Sample, Species, ValidationError,
)
from .logging_setup import configure_logging
from .model_service import ModelService
from .models import ModelError, ModelStore
from .repository import SampleRepository
from .service import JournalService
from .settings import Settings
from .training import ALGORITHM_LABELS
from .ui_methods import methods_page, render_evaluation
from .ui_predictions import classification_page, sample_predictions


logger = logging.getLogger(__name__)
DISPLAY_TIMEZONE = timezone(timedelta(hours=5))


@st.cache_resource
def get_service(database_path: str, log_dir: str) -> JournalService:
    """Создаёт общий сервис журнала и логирование, кешируя их по путям размещения.

    Репозиторий открывает отдельное соединение на операцию, поэтому кеширование
    сервиса не разделяет одно SQLite-подключение между пользовательскими сеансами.
    """
    configure_logging(Path(log_dir))
    return JournalService(SampleRepository(Path(database_path)))


def species_label(value: Species | None) -> str:
    """Возвращает название выбранного вида или текст для отсутствующего определения."""
    return value.label if value else "Не определён"


def local_time(value: str) -> str:
    """Преобразует сохранённое ISO-время с часовым поясом в отображение по Екатеринбургу."""
    return datetime.fromisoformat(value).astimezone(DISPLAY_TIMEZONE).strftime("%d.%m.%Y %H:%M:%S")


def sample_form(service: JournalService, editing: Sample | None = None) -> None:
    """Показывает форму создания или правки и сохраняет только после отправки.

    editing удерживает прочитанную редакцию для проверки конфликта изменений.
    Ключи новой формы меняются после успеха, чтобы очистить ввод следующего образца.
    """
    observation = editing.observation if editing else None
    generation = st.session_state.get("new_generation", 0)
    # Редакция отделяет поля правки от прежнего снимка, а generation после
    # добавления даёт пустую форму вместо повторного отображения сохранённых значений.
    prefix = f"edit:{editing.id}:{editing.version}" if editing else f"new:{generation}"
    with st.form(f"form:{prefix}"):
        st.subheader("Наблюдение")
        left, right = st.columns(2)
        sample_code = left.text_input(
            "Номер образца", value=observation.sample_code if observation else "",
            max_chars=100, placeholder="Например, IR-001", key=f"{prefix}:code",
        )
        observed_on = right.date_input(
            "Дата наблюдения", value=observation.observed_on if observation else datetime.now(DISPLAY_TIMEZONE).date(),
            format="DD.MM.YYYY", key=f"{prefix}:date",
        )
        location = st.text_input(
            "Место наблюдения", value=observation.location if observation else "",
            max_chars=5000, placeholder="Участок, коллекция или место сбора", key=f"{prefix}:location",
        )
        st.caption("Измерьте все четыре характеристики одного цветка. Единица измерения — сантиметр.")
        measurements = {}
        columns = st.columns(2)
        for index, name in enumerate(FEATURE_COLUMNS):
            measurements[name] = columns[index % 2].number_input(
                FEATURE_LABELS[name], min_value=0.01,
                value=getattr(observation.measurements, name) if observation else None,
                step=0.01, format="%.2f", key=f"{prefix}:{name}",
            )
        notes = st.text_area(
            "Дополнительные наблюдения", value=observation.notes if observation else "",
            max_chars=5000, placeholder="Состояние цветка, особенности измерения…", key=f"{prefix}:notes",
        )
        st.divider()
        st.subheader("Определение человеком")
        st.caption("Предположение и подтверждённое определение сохраняются отдельно от измерений.")
        options = [None, *Species]
        left, right = st.columns(2)
        hypothesis = left.selectbox(
            "Предполагаемый вид", options,
            index=options.index(observation.hypothesized_species) if observation else 0,
            format_func=species_label, key=f"{prefix}:hypothesis",
        )
        hypothesis_basis = left.text_area(
            "Обоснование предположения", value=observation.hypothesis_basis if observation else "",
            max_chars=5000, placeholder="Какие признаки привели к этому предположению?",
            key=f"{prefix}:hypothesis_basis",
        )
        confirmed = right.selectbox(
            "Подтверждённый вид", options,
            index=options.index(observation.confirmed_species) if observation else 0,
            format_func=species_label, key=f"{prefix}:confirmed",
        )
        confirmation_basis = right.text_area(
            "Основание подтверждения", value=observation.confirmation_basis if observation else "",
            max_chars=5000, placeholder="Определитель, паспорт коллекции или заключение специалиста",
            key=f"{prefix}:confirmation_basis",
        )
        submitted = st.form_submit_button(
            "Сохранить изменения" if editing else "Добавить образец", type="primary",
        )
    if not submitted:
        return
    try:
        saved = service.save(
            Observation(
                sample_code=sample_code, observed_on=observed_on,
                measurements=Measurements(**measurements), location=location.strip(), notes=notes.strip(),
                hypothesized_species=hypothesis, hypothesis_basis=hypothesis_basis.strip(),
                confirmed_species=confirmed, confirmation_basis=confirmation_basis.strip(),
            ), editing,
        )
    except (ValidationError, ConflictError) as error:
        logger.warning("Запись отклонена: %s", error)
        st.error(str(error))
        return
    except (sqlite3.Error, OSError):
        logger.exception("Не удалось сохранить образец из интерфейса")
        st.error("Не удалось сохранить образец. Повторите попытку.")
        return
    st.session_state["flash"] = f"Образец «{saved.observation.sample_code}» сохранён."
    if editing:
        st.session_state.pop("editing_snapshot", None)
    else:
        st.session_state["new_generation"] = generation + 1
    st.rerun()


def journal_page(service: JournalService, models: ModelService, samples: list[Sample]) -> None:
    """Показывает фильтруемый журнал, экспорт, резервную копию и выбранный образец.

    Для правки удерживает снимок, а прогнозы показывает для актуальной записи.
    История измерений и машинных результатов остаётся доступной после изменений.
    """
    st.header("Журнал образцов")
    with st.expander("Резервная копия журнала"):
        st.caption("Копия включает записи, все редакции и прогнозы. Набор Iris и модели хранятся отдельно.")
        if st.button("Создать актуальную копию", key="prepare_backup"):
            try:
                st.session_state["journal_backup"] = (
                    datetime.now(DISPLAY_TIMEZONE).strftime("%Y%m%d-%H%M%S"), service.repository.backup_bytes(),
                )
            except (OSError, sqlite3.Error):
                logger.exception("Не удалось подготовить резервную копию из интерфейса")
                st.error("Не удалось подготовить копию. Повторите попытку.")
        if backup := st.session_state.get("journal_backup"):
            st.download_button(
                f"Скачать копию от {backup[0]}", backup[1], file_name=f"iris-journal-{backup[0]}.sqlite3",
                mime="application/octet-stream", key="download_backup",
            )
    if not samples:
        st.info("Журнал пока пуст. Добавьте первый образец в разделе «Новый образец».")
        return
    left, right = st.columns([2, 1])
    query = left.text_input("Поиск по номеру или месту", key="journal_search").strip().casefold()
    status = right.selectbox(
        "Статус", ["Все", "Требует определения", "Есть предположение", "Подтверждён"],
        key="journal_status",
    )
    filtered = [
        sample for sample in samples
        if (status == "Все" or sample.status == status)
        and (not query or query in sample.observation.sample_code.casefold()
             or query in sample.observation.location.casefold())
    ]
    st.caption(f"Найдено: {len(filtered)} из {len(samples)}")
    if not filtered:
        st.info("По выбранным условиям образцов нет.")
        return
    rows = []
    predictions = service.repository.latest_predictions()
    for sample in filtered:
        observation = sample.observation
        prediction = predictions.get(sample.id)
        rows.append({
            "Номер": observation.sample_code,
            "Дата": observation.observed_on,
            "Место": observation.location,
            "Статус": sample.status,
            "Предположение": species_label(observation.hypothesized_species),
            "Подтверждённый вид": species_label(observation.confirmed_species),
            "Вид модели": species_label(prediction.prediction.predicted_species) if prediction else "Нет прогноза",
            "Редакция прогноза": prediction.sample_version if prediction else None,
            "Прогноз для текущей редакции": bool(prediction and prediction.sample_version == sample.version),
            **{FEATURE_LABELS[name]: getattr(observation.measurements, name) for name in FEATURE_COLUMNS},
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.download_button(
        "Скачать выбранные записи в CSV", data=service.export_csv(filtered),
        file_name="iris-journal.csv", mime="text/csv", key="journal_export",
    )
    st.divider()
    by_id = {sample.id: sample for sample in filtered}
    selected_id = st.selectbox(
        "Открыть образец", list(by_id), index=None,
        format_func=lambda sample_id: by_id[sample_id].observation.sample_code,
        placeholder="Выберите запись для просмотра или правки", key="selected_sample",
    )
    if not selected_id:
        st.session_state.pop("editing_snapshot", None)
        return
    # Снимок удерживается между rerun: автоматическая подмена его новой редакцией
    # отключила бы защиту от перезаписи чужой правки по устаревшей форме.
    snapshot = st.session_state.get("editing_snapshot")
    if snapshot is None or snapshot.id != selected_id:
        snapshot = by_id[selected_id]
        st.session_state["editing_snapshot"] = snapshot
    if st.button("Обновить запись из журнала", key="reload_sample"):
        st.session_state.pop("editing_snapshot", None)
        st.rerun()
    # Прогноз строится по записи из базы, а не по несохранённым полям формы правки.
    sample_predictions(models, by_id[selected_id], local_time)
    with st.expander("Измерения и определение", expanded=True):
        sample_form(service, snapshot)
    with st.expander("История изменений"):
        history = service.repository.history(selected_id)
        history_rows = []
        for revision in history:
            history_rows.append({
                "Версия": revision["version"],
                "Сохранено (Екатеринбург)": local_time(revision["saved_at"]),
                "Номер": revision["sample_code"],
                "Дата наблюдения": revision["observed_on"],
                "Место": revision["location"],
                **{FEATURE_LABELS[name]: revision[name] for name in FEATURE_COLUMNS},
                "Предположение": species_label(Species(revision["hypothesized_species"])) if revision["hypothesized_species"] else "Не определён",
                "Обоснование": revision["hypothesis_basis"],
                "Подтверждённый вид": species_label(Species(revision["confirmed_species"])) if revision["confirmed_species"] else "Не определён",
                "Основание подтверждения": revision["confirmation_basis"],
                "Наблюдения": revision["notes"],
            })
        st.dataframe(pd.DataFrame(history_rows), hide_index=True, width="stretch")


def summary_page(service: JournalService, samples: list[Sample]) -> None:
    """Показывает состав коллекции, статистику измерений и согласованность актуальных прогнозов."""
    st.header("Сводка коллекции")
    confirmed = [sample for sample in samples if sample.observation.confirmed_species is not None]
    columns = st.columns(3)
    columns[0].metric("Всего образцов", len(samples))
    columns[1].metric("Подтверждено человеком", len(confirmed))
    columns[2].metric("Ожидают подтверждения", len(samples) - len(confirmed))
    if not samples:
        st.info("Сводка появится после добавления образцов.")
        return
    predictions = service.repository.latest_predictions()
    # Сравнение с подтверждением учитывает только ту же редакцию: прежний
    # прогноз мог быть рассчитан по другим измерениям и не входит в эти показатели.
    current = [sample for sample in samples if sample.id in predictions and predictions[sample.id].sample_version == sample.version]
    checked = [sample for sample in current if sample.observation.confirmed_species is not None]
    matches = sum(predictions[sample.id].prediction.predicted_species == sample.observation.confirmed_species for sample in checked)
    columns = st.columns(3)
    columns[0].metric("С прогнозом текущей редакции", len(current))
    columns[1].metric("Прогноз требует обновления", sum(sample.id in predictions for sample in samples) - len(current))
    columns[2].metric("Совпало с подтверждением", f"{matches} из {len(checked)}")
    st.caption("Сопоставление в журнале относится к вашей коллекции; оно не заменяет независимую проверку модели.")
    st.subheader("Подтверждённые определения")
    counts = pd.DataFrame([
        {"Вид": species.label, "Образцов": sum(sample.observation.confirmed_species == species for sample in confirmed)}
        for species in Species
    ])
    st.dataframe(counts, hide_index=True, width="stretch")
    st.subheader("Измерения коллекции")
    measurement_rows = [
        {FEATURE_LABELS[name]: getattr(sample.observation.measurements, name) for name in FEATURE_COLUMNS}
        for sample in samples
    ]
    frame = pd.DataFrame(measurement_rows)
    statistics = frame.agg(["min", "mean", "max"]).T.rename(
        columns={"min": "Минимум", "mean": "Среднее", "max": "Максимум"},
    )
    st.dataframe(statistics.round(2), width="stretch")


def model_page(service: ModelService) -> None:
    """Показывает активную модель, метрики и версии и запускает базовое обучение по кнопке."""
    st.header("Модель определения вида")
    st.write("Обучите базовую модель на сохранённых измерениях Iris. Результат обучения сохраняется между запусками.")
    st.caption("Базовое обучение использует логистическую регрессию и четыре измерения цветка. Другие алгоритмы доступны в разделе «Сравнение моделей».")
    try:
        active = service.store.active_info()
    except ModelError as error:
        active = None
        st.warning(str(error))
    if active is None:
        st.info("Активной модели пока нет.")
    else:
        st.subheader("Активная модель")
        st.caption(f"Алгоритм: {ALGORITHM_LABELS.get(active.algorithm, active.algorithm)}")
        columns = st.columns(3)
        columns[0].metric("Верных определений при проверке", f"{active.accuracy:.1%}")
        columns[1].metric("Образцов для обучения", len(active.training_ids))
        columns[2].metric("Образцов для проверки", len(active.evaluation_ids))
        st.caption(f"Обучена: {local_time(active.created_at)} · Модель: {active.model_id[:12]}")
        st.caption("Показатель рассчитан на образцах, которые не участвовали в обучении этой модели.")
        if active.evaluation is not None:
            with st.expander("Подробная оценка активной модели"):
                render_evaluation(active)
    if st.button("Обучить базовую модель", type="primary", key="train_baseline"):
        try:
            with st.spinner("Обучение и проверка модели…"):
                trained = service.train()
        except (ModelError, ValidationError) as error:
            st.error(str(error))
        except FileNotFoundError:
            st.error("Набор Iris не найден. Восстановите файл данных и повторите обучение.")
        except Exception:
            logger.exception("Не удалось обучить модель из интерфейса")
            st.error("Не удалось завершить обучение. Повторите попытку.")
        else:
            st.session_state["flash"] = f"Модель обучена и сохранена. Верных определений при проверке: {trained.accuracy:.1%}."
            st.rerun()
    st.divider()
    st.caption("Предположения и подтверждённые виды из пользовательского журнала не включаются в обучение автоматически.")
    versions = service.store.list_models()
    if versions:
        with st.expander("История обучения"):
            st.dataframe(pd.DataFrame([
                {
                    "Модель": info.model_id[:12],
                    "Алгоритм": ALGORITHM_LABELS.get(info.algorithm, info.algorithm),
                    "Обучена (Екатеринбург)": local_time(info.created_at),
                    "Верных определений": f"{info.accuracy:.1%}",
                    "Обучающих образцов": len(info.training_ids),
                    "Проверочных образцов": len(info.evaluation_ids),
                    "Активна": bool(active and info.model_id == active.model_id),
                }
                for info in versions
            ]), hide_index=True, width="stretch")


def main() -> None:
    """Строит приложение Streamlit: настройки, сервисы, навигацию и выбранный раздел.

    Пути берутся из окружения; сообщения успешных операций переживают st.rerun
    через состояние сеанса. Недоступная база останавливает построение страницы.
    """
    st.set_page_config(page_title="Ирис · Определение вида", page_icon="🌿", layout="wide")
    settings = Settings.from_environment()
    st.title("Ирис")
    st.caption("Определение вида по измерениям и журнал коллекции")
    try:
        service = get_service(str(settings.database_path), str(settings.log_dir))
        samples = service.repository.list_samples()
    except (sqlite3.Error, OSError, RuntimeError):
        logger.exception("Не удалось открыть журнал")
        st.error("Не удалось открыть журнал. Проверьте доступность папки приложения.")
        st.stop()
    with st.sidebar:
        st.subheader("Коллекция ирисов")
        page = st.radio("Раздел", ["Определить вид", "Новый образец", "Журнал", "Сводка", "Модель", "Сравнение моделей"], key="page")
        st.divider()
        st.caption(f"Образцов в журнале: {len(samples)}")
        st.caption("Поддерживаемые виды: Iris setosa, Iris versicolor, Iris virginica.")
    if message := st.session_state.pop("flash", None):
        st.success(message)
    models = ModelService(ModelStore(settings.artifacts_dir), settings.dataset_path, service.repository)
    if page == "Определить вид":
        classification_page(models, local_time)
    elif page == "Новый образец":
        st.header("Новый образец")
        sample_form(service)
    elif page == "Журнал":
        journal_page(service, models, samples)
    elif page == "Сводка":
        summary_page(service, samples)
    elif page == "Модель":
        model_page(models)
    else:
        methods_page(models, local_time)
