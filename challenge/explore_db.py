# %%
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

ROOT_PATH = Path(__file__).parent.parent / "dataset"
DB_NAME = "senior_ds_case_study_data.db"
DIABETES_THRESHOLD = 2  # need at least one except for high blood pressure
# diabetes => hepatitis / hepatitis => liver damage => diabetes
# diabetes => liver damage => high blood pressure (less important)
# diabetes => hyperlipidemia
pd.set_option("display.max_columns", 500)

# %%
# Data preparation
# ----------------

database = sqlite3.connect(ROOT_PATH / DB_NAME)
cursor = database.cursor()

# %%
# query lab results
lab_result_query = cursor.execute("SELECT * FROM lab_test_results")
res_col_names = list(map(lambda x: x[0], cursor.description))
lab_results = pd.DataFrame(lab_result_query.fetchall(), columns=res_col_names)
lab_results["report_date_utc"] = lab_results.report_date_utc.apply(pd.Timestamp)

print(lab_results.info())

missing_values = (
    lab_results.drop(["patient_id", "report_date_utc"], axis=1).isna().all(axis=1)
)
lab_results = lab_results[~missing_values]
# many missing data: initial thought too many missing data to be able to impute
# values for missing data

# %%
# query patient information
patient_query = cursor.execute("SELECT * FROM patient_info;")
patient_col_names = list(map(lambda x: x[0], cursor.description))
patient_info = pd.DataFrame(patient_query.fetchall(), columns=patient_col_names)
patient_info["birth_date"] = patient_info.birth_date.apply(pd.Timestamp)

# %%
print(patient_info.info())
# No missing information on patients

# %%
# work on aggregation
full_df = lab_results.merge(
    patient_info, on="patient_id", how="left", suffixes=("", "_")
)
full_df["age"] = full_df.apply(
    lambda x: (x.report_date_utc - x.birth_date).days, axis=1
)
full_df = full_df.drop(["birth_date", "report_date_utc"], axis=1)

# %%
# Data visualization
# -------------------

# visualize collinearity of data and diabetes
# visualize collinearity of data between each other (to drop any feature)
# handle artifacts from visual plot
full_df["glomerular_filtration_rate"] = np.clip(
    full_df.glomerular_filtration_rate,
    0,
    int(full_df.glomerular_filtration_rate.quantile(0.9)),
)
# normal rate seem to be around 60 mL/min, above 90 mL/min is already worrying
full_df["fasting_blood_glucose"] = np.clip(
    full_df.fasting_blood_glucose,
    0,
    7,
)
# because 2g/L seem already elevated and the 90% of the values are below ~6.6 g/L

full_df["cholesterol"] = np.clip(
    full_df.cholesterol,
    0,
    25,
)
# 2g/L is elevated and 90% of the values are below ~24g/L

full_df["alanine_aminotransferase"] = np.clip(full_df.alanine_aminotransferase, 0, 50)
# 35 UI/L is the upper normal range and 90% of the values are below ~50 UI/L

full_df["aspartate_aminotransferase"] = np.clip(
    full_df.aspartate_aminotransferase,
    0,
    50,
)
# 35 UI/L is the upper normal range and 95% of the values are below ~47 UI/L
full_df["creatinine"] = np.clip(
    full_df.creatinine,
    0,
    30,
)
# upper normal range at 14 mg/L and 95% of the values are below ~15mg/L
full_df["hemoglobin"] = np.clip(
    full_df.hemoglobin,
    0,
    10,
)
# upper normal range around 1.8 g/L, values too high seem unlikely

full_df["leukocytes"] = np.clip(
    full_df.leukocytes,
    0,
    20,
)
# normal range for leukocytes around 4 to 10x10⁹/L, 95% of the values are below ~10x10⁹/L

full_df["erythrocytes"] = np.clip(
    full_df.erythrocytes,
    0,
    20,
)
# normal range for leukocytes around 4 to 6x10¹²/L, max values seem to high

full_df["triglycerids"] = np.clip(
    full_df.triglycerids,
    0,
    10,
)
# normal range below 1.5g/L and above 5g/L is already deemed dangerous
full_df["CRP"] = np.clip(
    full_df.CRP,
    0,
    200,
)
# normal range below 6 mg/L, between 50 to 200 mg/L it is considered as an infection

full_df["hba1c"] = np.clip(
    full_df.hba1c,
    0,
    10,
)
# diabetes from 6.5% and above

# %%
# Visualization with clusters
# ---------------------------
clusterer = KMeans(n_clusters=2)
patient_data_without_na = full_df.dropna()
clusterer.fit(patient_data_without_na.drop(["sex", "age", "patient_id"], axis=1))
patient_data_without_na["clusters"] = clusterer.labels_
sns.pairplot(
    patient_data_without_na.drop(["sex", "age", "patient_id"], axis=1),
    hue="clusters",
    diag_kind=None,
)

# It is difficult to say if the clusters correspond to patients with diabetes
# DBSCAN was tested but contains a single cluster
# %%
# Hypothesis 1:
# -------------
# hypothesis diabetes when there is at least a 2 of comorbidities
# rationale is, we do not have missing data on the comorbidities so why not test it
full_df["diabetes"] = full_df.apply(
    lambda x: sum([x.hepatitis, x.high_blood_pressure, x.hyperlipidemia])
    >= DIABETES_THRESHOLD,
    axis=1,
)
# sub_col_names = res_col_names[2:] + ["diabetes"]
sub_col_names = full_df.drop(["patient_id"], axis=1).columns
sns.pairplot(full_df[sub_col_names], hue="diabetes", diag_kind=None)

# %%

sns.pairplot(full_df[sub_col_names].dropna(), hue="diabetes", diag_kind=None)
# There does not seem to be any correlation between the features

# %%
# Hypothesis 2
# ------------
# fasting_blood_glucose >= 1.26 or hba1c >6.5

full_df["diabetes"] = full_df.apply(
    lambda x: x.fasting_blood_glucose >= 1.26 or x.hba1c > 6.5, axis=1
)
sns.pairplot(full_df[sub_col_names], hue="diabetes", diag_kind=None)
# %%

sns.pairplot(full_df[sub_col_names].dropna(), hue="diabetes", diag_kind=None)
# Except when on fasting_blood_glucose and hba1c, it is impossible to tell
# diabetes from other elements.
