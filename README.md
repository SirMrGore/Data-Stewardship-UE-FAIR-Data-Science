# BEST Office TOF Sensor Activity

[![DOI](https://zenodo.org/badge/966793228.svg)](https://doi.org/10.5281/zenodo.15228894)

This repository contains the code and model outputs for a project that predicts human activity near a door based on time-of-flight (TOF) sensor data. The project was completed as part of the *Introduction to Research Data Management* course at TU Wien (Summer Semester 2025).

---

## Project Overview

A TOF sensor was mounted at 150 cm height and 60 cm from a door, recording distance readings. Values below 700 mm were filtered out, and the remaining data was aggregated into 10-minute intervals to create an `Activity_Count` label.

A machine learning model was trained to predict this activity using time-based features (`Hour`, `Minute`, `DayOfWeek`, and `Weekend`) derived from the timestamp.

---

## Contents

- `TofActivity.ipynb`: Full data loading, preprocessing, training and evaluation
- `output_model.pkl`: Trained gradient boosted regressor (scikit-learn, Poisson loss)
- `test_predictions_plot.png`: Visual comparison of predicted vs actual activity
- `codemeta.json`: Metadata describing model provenance and dependencies
- `README.md`: This file

---

## Model Details

- Type: `HistGradientBoostingRegressor`
- Loss: Poisson
- Features: Hour, Minute, DayOfWeek, Weekend
- Target: Activity_Count (aggregated in 10-min windows)
- Evaluation: MAE, MSE, R² on held-out test set

---

## Dataset Access (via DBRepo)

The input dataset and its splits are stored in TU Wien's DBRepo:

- **Full dataset**: [https://test.dbrepo.tuwien.ac.at/pid/6d1b5273-20b2-4187-a879-4eea0addf996](https://test.dbrepo.tuwien.ac.at/pid/6d1b5273-20b2-4187-a879-4eea0addf996)  
- **Training set**: [https://test.dbrepo.tuwien.ac.at/pid/15053a73-72c9-447f-911d-994db9fae658](https://test.dbrepo.tuwien.ac.at/pid/15053a73-72c9-447f-911d-994db9fae658)  
- **Validation set**: [https://test.dbrepo.tuwien.ac.at/pid/57c3d0e5-1167-4b12-8bad-c7f3825fa045](https://test.dbrepo.tuwien.ac.at/pid/57c3d0e5-1167-4b12-8bad-c7f3825fa045)  
- **Test set**: [https://test.dbrepo.tuwien.ac.at/pid/79df1043-496c-4502-9340-d96d884b29d9](https://test.dbrepo.tuwien.ac.at/pid/79df1043-496c-4502-9340-d96d884b29d9)  

[DBRepo Repository Link](https://test.dbrepo.tuwien.ac.at/pid/1ee874cd-08b0-4252-9dff-71cb8010e474)

---

## 📤 Output Model and Results (via TUWRD)

The trained model and output visualizations are available on TUWRD:

- **Model file**: `output_model.pkl`  
- **Evaluation plot**: `test_predictions_plot.png`  
- **Metadata**: Includes CodeMeta and FAIR4ML-compliant fields

[TUWRD Entry]() *(TODO)*

---

## Dependencies

- Python 3.12+
- `scikit-learn >=1.4`
- `pandas >=2.0`
- `matplotlib >=3.7`

Install dependencies via:
```bash
pip install -r requirements.txt
```