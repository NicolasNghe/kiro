"""Main module for training and evaluation."""

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from numpy.typing import NDArray
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from challenge.data_processing.data_imputation import impute_data
from challenge.data_processing.prepare_raw_data import create_unique_table

# two stratification strategies possible:
# - stratify by the outcome diabetes, this one show incredible good performances as if it was over fitting
# - stratify by the 3 other pathologies but this strategy show very poor performances
STRATIFICATION_VARIABLES = ["diabetes"]
DIABETES_THRESHOLD = 1  # 2 comorbidities creates a database too imbalanced and return really poor results
SEED = 42
np.random.seed(SEED)


def train_test_split(
    table: pd.DataFrame,
    stratification_variables: list[str] = STRATIFICATION_VARIABLES,
    groups: str = "patient_id",
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Splits the dataset into a train and a test dataset.
    The datasets are stratified using the `stratification_variables`.

    Args:
        table: pandas DataFrame to split into a train DataFrame and a test
            DataFrame
        stratification_variables: variables on which to stratify the datasets
        groups: variables to consider for preventing leakage.

    Returns:
        train_idx: indices of the rows to include in the train dataset.
        test_idx: indices of the rows to include in the test dataset.
    """
    sgkf = StratifiedGroupKFold(shuffle=True, random_state=42)
    _splits = sgkf.split(
        table,
        y=table[stratification_variables].apply(
            lambda x: ", ".join(str(_) for _ in x), axis=1
        ),
        groups=table[groups],
    )
    train_idx, test_idx = next(_splits)
    return train_idx, test_idx


def normalize_data(
    train_feat: pd.DataFrame, train_label: pd.DataFrame, test_feat: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalizes the train and test data using a RobustScaler.

    Args:
        train_feat: train features.
        train_label: train label.
        test_feat: test features.

    Returns:
        norm_train_feat: normalized train features.
        norm_test_feat: normalized test features.
    """
    scaler = RobustScaler()
    norm_train_feat = scaler.fit_transform(train_feat, train_label)
    norm_test_feat = scaler.transform(test_feat)
    return norm_train_feat, norm_test_feat


def split_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits features and labels into train and test.

    Args:
        data: dataframe with the features and the labels..
            - labels are contained in the column `diabetes`

    Returns:
        tuple with the train set (features and labels) and test set (features and
            labels).
    """
    train_idx, test_idx = train_test_split(data)

    Y = data["diabetes"]
    X = data.drop(["diabetes", "patient_id"] + STRATIFICATION_VARIABLES, axis=1)

    x_train = X.iloc[train_idx]
    y_train = Y.iloc[train_idx]

    x_test = X.iloc[test_idx]
    y_test = Y.iloc[test_idx]
    return x_train, y_train, x_test, y_test


def eval_single_model(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    model: Any,
) -> dict[str, Any]:
    """Trains and evaluates a single model.

    Args:
        x_train: train features
        y_train: labels in the train dataset
        x_test: test features
        y_test: labels in the test dataset
        model: model to test

    Returns:
        dictionary containing a trained model
    """

    class_weight = {
        0: (len(y_train) - y_train.sum()) / len(y_train),
        1: y_train.sum() / len(y_train),
    }
    try:
        model_ = model(class_weight=class_weight, random_state=SEED)
    except TypeError:
        model_ = model(random_state=SEED)
    model_.fit(x_train, y_train)
    predictions = model_.predict(x_test)
    f1_score_ = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    logger.info(
        f"{model_.__class__.__name__}:\n"
        f" f1-score: {f1_score_}\n recall: {recall}\n precision: {precision}"
    )
    return {model_.__class__.__name__: f1_score_, "model": model}


def eval_models(
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    scale_x: bool = True,
    *,
    impute: bool = False,
    handle_nan: bool = False,
) -> list[dict[str, Any]]:
    """Evaluates models.

    Args:
        y_train: labels in the train dataset
        y_test: labels in the test dataset
        x_train: train features
        x_test: test features
        scale_x: scale the features
        impute:
            - True: impute missing data,
            - False: leaves missing data,
        handle_nan:
            - True run models capable of handling NaN values,
            - False run models not capable of handling NaN values,
    Returns:
        list containing dictionaries with trained models and their performance
    """
    if impute:
        x_train, x_test = impute_data(x_train, x_test)
    if handle_nan:
        models = [
            HistGradientBoostingClassifier,
            BaggingClassifier,
            RandomForestClassifier,
            DecisionTreeClassifier,
            LGBMClassifier,
        ]
    else:
        models = [
            LogisticRegression,
            RandomForestClassifier,
            AdaBoostClassifier,
            GradientBoostingClassifier,
            SGDClassifier,
            SVC,
        ]
    if scale_x:
        x_train, x_test = normalize_data(x_train, y_train, x_test)

    model_performances = []
    for model in models:
        model_performances.append(
            eval_single_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                model=model,
            )
        )
    return model_performances


def create_label(patient_data: pd.DataFrame, strategy: str) -> pd.Series:
    """Creates diabetes labels based on the strategy.

    Args:
        patient_data: patient features.
        strategy: chosen strategy to create the labels.

    Returns:
        labels
    """
    match strategy:
        case "comorbidities":
            # hypothesis: diabetes when there is at least a certain number of comorbidities
            labels = patient_data.apply(
                lambda x: sum([x.hepatitis, x.high_blood_pressure, x.hyperlipidemia])
                >= DIABETES_THRESHOLD,
                axis=1,
            )
        case "measures":
            # hypothesis: diabetes when some measurements are too high
            labels = patient_data.apply(
                lambda x: x.fasting_blood_glucose >= 1.26 or x.hba1c > 6.5, axis=1
            )
        case _:
            raise NotImplementedError
    return labels


def main() -> None:
    """Main script to prepare the data and run different model trainings and
    evaluations"""
    patient_data = create_unique_table()

    # Hypothesis 1: pathologies as labels
    patient_data["diabetes"] = create_label(patient_data, "comorbidities")
    x_train, y_train, x_test, y_test = split_data(patient_data)

    x_train = x_train.drop(
        ["hyperlipidemia", "high_blood_pressure", "hepatitis"], axis=1
    )
    x_test = x_test.drop(["hyperlipidemia", "high_blood_pressure", "hepatitis"], axis=1)

    for handle_nan, impute_ in zip([True, False], [False, True]):
        logger.info(
            "Running model evaluation for hypothesis 1:\n"
            f"data imputation: {impute_}\n"
            f"classifier handling NaNs: {handle_nan}"
        )
        _ = eval_models(
            y_train=y_train,
            y_test=y_test,
            x_train=x_train,
            x_test=x_test,
            impute=impute_,
            handle_nan=handle_nan,
        )
        # without imputation: results are terrible, f1 scores are lower than 0.2
        # with imputation: results are terrible in comparison to without imputation

    # Hypothesis 2: uses fasting blood glucose > 1.26 g/L or HbA1c > 6.5% as diabetes
    no_info = patient_data.apply(
        lambda x: np.isnan(x.fasting_blood_glucose) and np.isnan(x.hba1c), axis=1
    )
    patient_data.loc[no_info, "diabetes"] = None
    patient_data["diabetes"] = patient_data["diabetes"] = create_label(
        patient_data, "measures"
    )
    # work on data with labels only
    patient_data_ = patient_data.dropna(subset="diabetes")

    x_train_, y_train_, x_test_, y_test_ = split_data(
        patient_data_.drop(["fasting_blood_glucose", "hba1c"], axis=1)
    )
    for handle_nan, impute_ in zip([True, False], [False, True]):
        logger.info(
            "Running model evaluation for hypothesis 2:\n"
            f"data imputation: {impute_} "
            f"using classifier capable of handling NaNs: {handle_nan}"
        )
        _ = eval_models(
            y_train=y_train_,
            y_test=y_test_,
            x_train=x_train_,
            x_test=x_test_,
            impute=impute_,
            handle_nan=handle_nan,
        )
    # without imputation: results are poor (~0.2 to 0.3) if we drop `fasting_blood_glucose` and `hba1c`
    # but if these columns are not dropped, the predictions are obvious
    # with imputation: results are equivalent

    # if we drop all the NaN the performances are good
    logger.info("\nHypothesis 2: dropping all the rows containing NaNs\n")
    patient_data_ = patient_data.dropna(subset="diabetes").dropna()

    x_train_, y_train_, x_test_, y_test_ = split_data(
        patient_data_.drop(["fasting_blood_glucose", "hba1c"], axis=1)
    )
    for handle_nan, impute_ in zip([True, False], [False, True]):
        logger.info(
            "Running model evaluation for hypothesis 2: "
            f"data imputation: {impute_} "
            f"using classifier capable of handling NaNs: {handle_nan}"
        )
        _ = eval_models(
            y_train=y_train_,
            y_test=y_test_,
            x_train=x_train_,
            x_test=x_test_,
            impute=impute_,
            handle_nan=handle_nan,
        )


if __name__ == "__main__":
    main()
