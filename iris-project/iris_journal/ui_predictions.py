"""Определение новых ирисов, пакетная обработка и сохранённые прогнозы."""

from datetime import datetime, timedelta, timezone
import logging
import sqlite3
from typing import Callable

import pandas as pd
import streamlit as st

from .batch import BatchValidationError, export_predictions, template_csv
from .domain import ConflictError, FEATURE_COLUMNS, FEATURE_LABELS, Measurements, Observation, Prediction, Sample, Species, ValidationError
from .model_service import ModelService
from .models import ModelError
from .training import ALGORITHM_LABELS


logger = logging.getLogger(__name__)


def model_gate(service: ModelService):
    """Возвращает сведения готового конвейера или предлагает подготовить модель по кнопке."""
    try:
        return service.store.load_active().info
    except ModelError as error:
        st.info(str(error))
    if st.button("Подготовить модель", type="primary", key="prepare_model"):
        try:
            with st.spinner("Подготовка модели на Iris…"):
                service.prepare()
        except Exception:
            logger.exception("Не удалось подготовить модель из интерфейса")
            st.error("Не удалось подготовить модель. Проверьте наличие набора Iris и доступность папки приложения.")
        else:
            st.session_state["flash"] = "Модель подготовлена. Можно определять новые ирисы."
            st.rerun()
    return None


def prediction_card(service: ModelService, prediction: Prediction, format_time: Callable[[str], str]) -> None:
    """Показывает вид, оценки и происхождение прогноза вместе с использованными измерениями."""
    st.success(f"Предполагаемый вид: {prediction.predicted_species.label}")
    st.caption(
        f"{ALGORITHM_LABELS.get(prediction.algorithm, prediction.algorithm)} · "
        f"Модель {prediction.model_id[:12]} · {format_time(prediction.created_at)}"
    )
    st.dataframe(pd.DataFrame([
        {"Вид": species.label, "Оценка модели": f"{prediction.probabilities[species]:.1%}"}
        for species in Species
    ]), hide_index=True, width="stretch")
    st.caption("Оценки относятся к трём поддерживаемым видам. Для подтверждённого определения нужно независимое основание.")
    with st.expander("Измерения, использованные для этого результата"):
        st.dataframe(pd.DataFrame([
            {"Признак": FEATURE_LABELS[name], "Значение, см": getattr(prediction.measurements, name)}
            for name in FEATURE_COLUMNS
        ]), hide_index=True, width="stretch")
    for warning in service.measurement_warnings(prediction.measurements):
        st.warning(warning)


def single_tab(service: ModelService, active_id: str, format_time: Callable[[str], str]) -> None:
    """Определяет один цветок и предлагает сохранить его вместе с полученным прогнозом.

    Результат хранится в сеансе отдельно от редактируемых полей формы.
    После успешной записи повторная кнопка сохранения убирается.
    """
    st.write("Введите четыре измерения одного цветка в сантиметрах.")
    with st.form("single_prediction_form"):
        columns = st.columns(2)
        values = {
            name: columns[index % 2].number_input(
                FEATURE_LABELS[name], min_value=0.01, value=None, step=0.01,
                format="%.2f", key=f"single:{name}",
            )
            for index, name in enumerate(FEATURE_COLUMNS)
        }
        submitted = st.form_submit_button("Определить вид", type="primary")
    if submitted:
        # Ошибка нового ввода должна убрать прежнюю карточку, чтобы пользователь
        # не принял результат предыдущих измерений за ответ на текущие.
        st.session_state.pop("single_result", None)
        st.session_state.pop("single_saved", None)
        try:
            st.session_state["single_result"] = service.predict_measurements(Measurements(**values))
        except (ValidationError, ModelError) as error:
            st.error(str(error))
        except Exception:
            logger.exception("Ошибка определения нового образца")
            st.error("Не удалось определить вид. Повторите попытку.")
    prediction = st.session_state.get("single_result")
    if prediction is None:
        return
    prediction_card(service, prediction, format_time)
    if prediction.model_id != active_id:
        st.info("Этот результат получен предыдущей моделью. Для результата активной модели повторите определение.")
    saved = st.session_state.get("single_saved")
    if saved is not None:
        sample = service.repository.get(saved.sample_id)
        st.info(f"Образец «{sample.observation.sample_code}» и прогноз сохранены. Они доступны в журнале.")
        return
    with st.form("save_single_prediction"):
        st.subheader("Сохранить образец в журнал")
        columns = st.columns(2)
        code = columns[0].text_input("Номер образца", max_chars=100, key="single_save_code", placeholder="Например, IR-001")
        observed_on = columns[1].date_input(
            "Дата наблюдения", value=datetime.now(timezone(timedelta(hours=5))).date(),
            format="DD.MM.YYYY", key="single_save_date",
        )
        location = st.text_input("Место наблюдения", max_chars=5000, key="single_save_location")
        notes = st.text_area("Дополнительные наблюдения", max_chars=5000, key="single_save_notes")
        save = st.form_submit_button("Сохранить образец и прогноз", type="primary")
    if save:
        try:
            observation = Observation(
                # Сохраняем измерения самого прогноза, даже если поля верхней
                # формы были отредактированы после последнего определения.
                sample_code=code, observed_on=observed_on, measurements=prediction.measurements,
                location=location.strip(), notes=notes.strip(),
            )
            st.session_state["single_saved"] = service.save_new_prediction(observation, prediction)
        except (ValidationError, ConflictError) as error:
            st.error(str(error))
        except (sqlite3.Error, OSError):
            logger.exception("Не удалось сохранить новый образец с прогнозом")
            st.error("Не удалось сохранить результат. Повторите попытку.")
        else:
            st.session_state["flash"] = "Образец и прогноз сохранены в журнале."
            st.rerun()


def batch_tab(service: ModelService) -> None:
    """Проверяет загруженный CSV, показывает результаты и предлагает экспорт или общее сохранение.

    Смена содержимого файла сбрасывает прежние результаты и отметку сохранения.
    Прогнозы появляются до записи в журнал; запись пакета атомарна.
    """
    st.write("Загрузите CSV с измерениями. Сначала просмотрите результаты, затем сохраните их в журнал или скачайте таблицу.")
    st.caption("Обязательные столбцы: " + ", ".join(FEATURE_COLUMNS))
    st.caption("Дополнительно: sample_code, observed_on, location, notes. Поддерживаются UTF-8 и Windows-1251, разделители , ; и табуляция, десятичная точка или запятая. До 5000 строк и 2 МБ.")
    st.download_button("Скачать шаблон CSV", template_csv(), file_name="iris-input.csv", mime="text/csv", key="batch_template")
    uploaded = st.file_uploader("CSV с новыми образцами", type=["csv"], key="batch_upload")
    payload = uploaded.getvalue() if uploaded is not None else None
    if st.session_state.get("batch_payload") != payload:
        # Кнопки Streamlit перезапускают страницу. Связываем состояние с байтами
        # файла, чтобы смена загрузки не оставила прогнозы и отметку записи старого CSV.
        st.session_state["batch_payload"] = payload
        st.session_state.pop("batch_result", None)
        st.session_state.pop("batch_saved", None)
    if st.button("Проверить CSV и определить виды", type="primary", disabled=payload is None, key="classify_csv"):
        st.session_state.pop("batch_result", None)
        st.session_state.pop("batch_saved", None)
        try:
            with st.spinner("Проверка и определение видов…"):
                st.session_state["batch_result"] = service.classify_csv(payload)
        except BatchValidationError as error:
            st.error(str(error))
            st.dataframe(pd.DataFrame([
                {"Строка CSV": issue.line, "Ошибка": issue.message} for issue in error.issues
            ]), hide_index=True, width="stretch")
        except (ValidationError, ModelError) as error:
            st.error(str(error))
        except Exception:
            logger.exception("Ошибка пакетного определения из интерфейса")
            st.error("Не удалось обработать файл. Повторите попытку.")
    result = st.session_state.get("batch_result")
    if result is None:
        return
    warning_count = sum(bool(item.warnings) for item in result.rows)
    st.caption(f"Образцов: {len(result.rows)} · Требуют проверки измерений: {warning_count} · Модель: {result.model_id[:12]}")
    frame = pd.DataFrame([
        {
            "Номер": item.observation.sample_code, "Предполагаемый вид": item.prediction.predicted_species.label,
            **{species.label: item.prediction.probabilities[species] for species in Species},
            **{FEATURE_LABELS[name]: getattr(item.observation.measurements, name) for name in FEATURE_COLUMNS},
            "Замечания": " ".join(item.warnings),
        }
        for item in result.rows
    ])
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button(
        "Скачать результаты в CSV", export_predictions(result), file_name="iris-predictions.csv",
        mime="text/csv", key="batch_export",
    )
    if st.session_state.get("batch_saved") is not None:
        st.success("Все образцы и прогнозы сохранены в журнале.")
    elif st.button("Добавить все образцы и прогнозы в журнал", key="save_batch"):
        try:
            st.session_state["batch_saved"] = service.save_batch(result)
        except (ValidationError, ConflictError) as error:
            st.error(str(error))
        except (sqlite3.Error, OSError):
            logger.exception("Не удалось сохранить пакет образцов")
            st.error("Пакет не сохранён. Повторите попытку.")
        else:
            st.session_state["flash"] = f"Сохранено образцов с прогнозами: {len(result.rows)}."
            st.rerun()


def classification_page(service: ModelService, format_time: Callable[[str], str]) -> None:
    """Проверяет доступность модели и показывает вкладки одного цветка и CSV."""
    st.header("Определить вид ириса")
    info = model_gate(service)
    if info is None:
        return
    st.caption(f"Активная модель: {ALGORITHM_LABELS.get(info.algorithm, info.algorithm)} · {info.model_id[:12]}")
    single, batch = st.tabs(["Один цветок", "CSV с образцами"])
    with single:
        single_tab(service, info.model_id, format_time)
    with batch:
        batch_tab(service)


def sample_predictions(service: ModelService, sample: Sample, format_time: Callable[[str], str]) -> None:
    """Показывает сохранённые прогнозы образца и позволяет определить прочитанную запись заново.

    Сопоставление с человеческим подтверждением выполняется только для совпавшей
    редакции; прежний результат остаётся в истории и получает предупреждение.
    """
    st.subheader("Прогноз модели")
    st.caption(f"Определение выполняется по сохранённым измерениям редакции {sample.version}.")
    if st.button("Определить вид этого образца", key=f"predict_sample:{sample.id}"):
        try:
            service.predict_sample(sample.id)
        except (ModelError, ValidationError) as error:
            st.error(str(error))
        except (sqlite3.Error, OSError):
            logger.exception("Не удалось сохранить прогноз существующего образца")
            st.error("Не удалось сохранить прогноз. Повторите попытку.")
        except Exception:
            logger.exception("Не удалось определить сохранённый образец")
            st.error("Не удалось определить вид образца. Проверьте модель в разделе «Модель».")
        else:
            st.session_state["flash"] = "Прогноз сохранён для текущих измерений образца."
            st.rerun()
    predictions = service.repository.list_predictions(sample.id)
    if not predictions:
        st.info("Для этого образца ещё нет сохранённых прогнозов.")
        return
    latest = predictions[0]
    prediction_card(service, latest.prediction, format_time)
    if latest.sample_version != sample.version:
        st.warning(f"Прогноз относится к редакции {latest.sample_version}; текущая редакция — {sample.version}. Выполните новое определение.")
    else:
        confirmed = sample.observation.confirmed_species
        hypothesis = sample.observation.hypothesized_species
        if confirmed is not None:
            if confirmed == latest.prediction.predicted_species:
                st.info("Прогноз совпадает с подтверждённым человеком видом.")
            else:
                st.warning(f"Прогноз отличается от подтверждённого человеком вида: {confirmed.label}.")
        elif hypothesis is not None:
            st.info(f"Предположение человека: {hypothesis.label}. Подтверждённый вид пока не указан.")
    with st.expander("История прогнозов"):
        st.dataframe(pd.DataFrame([
            {
                "Время (Екатеринбург)": format_time(item.prediction.created_at),
                "Редакция образца": item.sample_version, "Текущая редакция": item.sample_version == sample.version,
                "Вид модели": item.prediction.predicted_species.label,
                "Алгоритм": ALGORITHM_LABELS.get(item.prediction.algorithm, item.prediction.algorithm),
                "Модель": item.prediction.model_id[:12],
                **{species.label: item.prediction.probabilities[species] for species in Species},
            }
            for item in predictions
        ]), hide_index=True, width="stretch")
