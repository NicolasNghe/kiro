"""Module for data imputation."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

SEED = 0


def select_valid_value(measure: pd.Series) -> Any:
    """Selects the first non-NaN value in a pandas Series.

    Args:
        measure: pandas Series with NaN values.
    Returns:
        first non-NaN value if there are, otherwise a NaN
    """
    candidates = pd.notna(measure)
    value = np.nan
    if candidates.any():
        value = measure[candidates].iloc[0]
    return value


def create_patient_result(patient_measures: pd.DataFrame) -> dict[str, Any]:
    """Creates a unique patient result from the different lab results.
    The unique result is created using the most recent measurements.

    Args:
        patient_measures: DataFrame containing all the lab results for a given
            patient.
    Returns:
        a dictionary containing a unique set of results.
    """
    patient_measures = patient_measures.sort_values("report_date_utc", ascending=False)
    return {
        col: select_valid_value(patient_measures[col])
        for col in patient_measures.columns
    }


def impute_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imputes missing data.

    Args:
        train_data: training data used to fit the imputer.
        test_data: test data

    Returns:
        tuple with the training data and the test data filled.
    """
    imp_ = IterativeImputer(max_iter=100, random_state=SEED)
    imp_train_data = pd.DataFrame(
        imp_.fit_transform(train_data), columns=train_data.columns
    )
    imp_test_data = pd.DataFrame(imp_.transform(test_data), columns=test_data.columns)
    return imp_train_data, imp_test_data
