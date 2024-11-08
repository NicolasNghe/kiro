"""Module to fetch the data and prepare it."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from challenge.data_processing.data_imputation import create_patient_result

ROOT_PATH = Path(__file__).parent.parent.parent / "dataset"
DB_NAME = "senior_ds_case_study_data.db"

pd.set_option("display.max_columns", 500)


def fetch_table(cursor: sqlite3.Cursor, table_name: str) -> pd.DataFrame:
    """Fetches SQLite3 table.

    Args:
        cursor: Cursor the database.
        table_name: name of the table to fetch.

    Returns:
        Queried table.
    """
    query_result = cursor.execute(f"SELECT * FROM {table_name};")
    col_names = list(map(lambda x: x[0], cursor.description))
    results = pd.DataFrame(query_result.fetchall(), columns=col_names)
    return results


def eliminate_artefacts(patient_measures: pd.DataFrame) -> pd.DataFrame:
    """Eliminates artefacts based on heuristics.
    Reference values are obtained through internet research and upper bound are
    made up.

    Args:
        patient_measures: dataframe with patient lab measurements.

    Returns:
        dataframe with patient lab measurements with aberrant values clipped.
    """
    patient_measures["glomerular_filtration_rate"] = np.clip(
        patient_measures.glomerular_filtration_rate,
        0,
        150,
    )
    # normal rate seem to be around 60 mL/min, above 90 mL/min is already worrying
    patient_measures["fasting_blood_glucose"] = np.clip(
        patient_measures.fasting_blood_glucose,
        0,
        7,
    )
    # because 2g/L seem already elevated and the 90% of the values are below ~6.6 g/L

    patient_measures["cholesterol"] = np.clip(
        patient_measures.cholesterol,
        0,
        25,
    )
    # 2g/L is elevated and 90% of the values are below ~24g/L

    patient_measures["alanine_aminotransferase"] = np.clip(
        patient_measures.alanine_aminotransferase, 0, 50
    )
    # 35 UI/L is the upper normal range and 90% of the values are below ~50 UI/L

    patient_measures["aspartate_aminotransferase"] = np.clip(
        patient_measures.aspartate_aminotransferase,
        0,
        50,
    )
    # 35 UI/L is the upper normal range and 95% of the values are below ~47 UI/L
    patient_measures["creatinine"] = np.clip(
        patient_measures.creatinine,
        0,
        30,
    )
    # upper normal range at 14 mg/L and 95% of the values are below ~15mg/L
    patient_measures["hemoglobin"] = np.clip(
        patient_measures.hemoglobin,
        0,
        10,
    )
    # upper normal range around 1.8 g/L, values too high seem unlikely

    patient_measures["leukocytes"] = np.clip(
        patient_measures.leukocytes,
        0,
        20,
    )
    # normal range for leukocytes around 4 to 10x10⁹/L, 95% of the values are below ~10x10⁹/L

    patient_measures["erythrocytes"] = np.clip(
        patient_measures.erythrocytes,
        0,
        20,
    )
    # normal range for leukocytes around 4 to 6x10¹²/L, max values seem to high

    patient_measures["triglycerids"] = np.clip(
        patient_measures.triglycerids,
        0,
        10,
    )
    # normal range below 1.5g/L and above 5g/L is already deemed dangerous
    patient_measures["CRP"] = np.clip(
        patient_measures.CRP,
        0,
        200,
    )
    # normal range below 6 mg/L, between 50 to 200 mg/L it is considered as an infection

    patient_measures["hba1c"] = np.clip(
        patient_measures.hba1c,
        0,
        10,
    )
    # diabetes from 6.5% and above
    return patient_measures


def create_unique_table() -> pd.DataFrame:
    """Fetches relevant tables from the database and clean the data.

    Returns:
        A unique pandas DataFrame with relevant information.
    """
    database = sqlite3.connect(ROOT_PATH / DB_NAME)
    cursor = database.cursor()

    # clean lab_results
    lab_results = fetch_table(cursor, table_name="lab_test_results")
    lab_results["report_date_utc"] = lab_results.report_date_utc.apply(pd.Timestamp)

    # remove artefacts
    lab_results = eliminate_artefacts(lab_results)
    missing_values = (
        lab_results.drop(["patient_id", "report_date_utc"], axis=1).isna().all(axis=1)
    )
    lab_results = lab_results[~missing_values]

    patient_results = pd.DataFrame(
        [
            create_patient_result(patient)
            for _, patient in lab_results.groupby("patient_id")
        ]
    )

    # prepare patient_info
    patient_info = fetch_table(cursor, table_name="patient_info")
    patient_info["birth_date"] = patient_info.birth_date.apply(pd.Timestamp)

    # merge all tables
    patient_summary = patient_results.merge(
        patient_info, on="patient_id", how="left", suffixes=("", "_")
    )
    patient_summary["age"] = patient_summary.apply(
        lambda x: (x.report_date_utc - x.birth_date).days, axis=1
    )
    patient_summary = patient_summary.drop(["birth_date", "report_date_utc"], axis=1)

    return patient_summary
