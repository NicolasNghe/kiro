# Kiro challenge

## Requirements

- Python 3.10
- poetry 1.8.4

## Installation


* Clone repository from Gitlab:

```shell
git clone git@github.com:NicolasNghe/kiro.git
```

* move to the project's directory:

```shell
cd kiro
```

### Installation with Poetry
* create a Poetry virtual environment for the project:

```shell
poetry install
```

* activate the virtual environment:

```sh
source .venv/bin/activate
```

## Presentation

The code is contained in the folder `challenge`.  

#### Notebook exploration  
The script `explore_db.py` is a notebook to explore the database.  
It contains the initial data exploration to try to understand the data contained in the database.  

This initial step helped to see the presence of artefacts, missing values and explore 2 different hypotheses to represent diabetes:  
- presence of comorbidities as a proxy for diabetes
- fasting_blood_glucose >= 1.26 g/L or hba1c >6.5% for diabetes

### Main code
The main code is located in the subfolder `data_processing/` and `evaluation/`.  

To run the main code, run the command

```sh
python challenge/evaluation/evaluation.py
```

It contains 3 modules:  
- `data_processing/prepare_raw_dataset.py`: fetches the data and make an initial processing
- `data_processing/data_imputation.py`: utilitary functions to further process the data
- `evaluation/evaluation.py`: main module to run which attempts to train different models and explore the two strategies

### Results

Neither hypothesis allowed to build a satisfactory model. The performances are quite low.  
The **f1-score is below 0.3** no matter the models used, with or without data imputation.  
However, when only the rows (patient records) with all the measurements are kept (no missing value),
the performances became good with an **f1-score ~0.7/0.8, precision ~0.7, recall ~0.8 to 1**.  

It seems that diabetes can be inferred from the other variables.  

<u>Further investigations</u>:
1. feature importance analysis to select the most important features (and ask the patient to do those tests?).  
2. a) try to improve feature imputation for those features.  
   b) alternatively train a model on the important features only.   
3. fine tune a selected model. 