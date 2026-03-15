# Build an ML Pipeline for Short-Term Rental Prices in NYC

This project implements an end-to-end ML pipeline to estimate rental prices for short-term rental properties in New York City. The pipeline automates data ingestion, validation, model training, evaluation, and deployment using MLflow, Hydra, and Weights & Biases.

The goal is to create a reproducible and modular pipeline capable of retraining the model every week as new rental data becomes available.

## Project Links

- [**GitHub Repository:**](https://github.com/BelenEsteve/build-ml-pipeline-for-short-term-rental-prices)
- [**Weights & Biases Project:**](https://wandb.ai/belen-esteve-co-odin-vision/nyc_airbnb?nw=nwuserbelenesteveco)

## Table of Contents

- [Project Overview](#project-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Environment Setup](#environment-setup)
- [Running the Pipeline](#running-the-pipeline)
- [Pipeline Steps](#pipeline-steps)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Model Selection](#model-selection)
- [Model Testing](#model-testing)
- [Release the Pipeline](#release-the-pipeline)
- [Troubleshooting](#troubleshooting)

## Project Overview

A property management company rents rooms and apartments on several short-term rental platforms. To stay competitive, they need a system that predicts typical nightly prices based on the characteristics of similar listings.

New data arrives weekly, so the system must support automated retraining through a reproducible ML pipeline.

This project demonstrates an MLOps workflow including:

- Data ingestion
- Data validation
- Feature engineering
- Model training
- Experiment tracking
- Model evaluation
- Reproducible deployment

The pipeline is built using:

- **MLflow** → pipeline orchestration
- **Hydra** → configuration management
- **Weights & Biases** → experiment tracking and artifact storage
- **Scikit-learn** → machine learning model

## Pipeline Architecture

The pipeline consists of the following steps:
```
download → eda → basic_cleaning → data_check → train_val_test_split → train_random_forest → test_regression_model
```

- Each step is implemented as an MLflow component.
- Artifacts are tracked and versioned using Weights & Biases.

## Environment Setup

### Supported Operating Systems

- Ubuntu 22.04 / 24.04
- macOS
- Windows (via WSL recommended)

### Create the Development Environment

Install dependencies with Conda:
```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

### Configure Weights & Biases

Login using your API key:
```bash
wandb login
```

You can obtain your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize).

## Running the Pipeline

Run the entire pipeline:
```bash
mlflow run .
```

Run only specific steps:
```bash
mlflow run . -P steps=download
```

Run multiple steps:
```bash
mlflow run . -P steps=download,basic_cleaning
```

Override configuration parameters with Hydra:
```bash
mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="etl.min_price=50 modeling.random_forest.n_estimators=100"
```

## Pipeline Steps

### 1. Data Download

Downloads a sample of the dataset and stores it as a W&B artifact (`sample.csv`).
```bash
mlflow run . -P steps=download
```

### 2. Exploratory Data Analysis (EDA)

An exploratory notebook analyzes the dataset to identify:

- Missing values
- Incorrect data types
- Outliers in the `price` column

Example findings:
- `last_review` must be converted to datetime
- Extreme price outliers must be removed

### 3. Data Cleaning

The `basic_cleaning` step removes price outliers, converts date columns, and outputs cleaned data as `clean_sample.csv`.

### 4. Data Validation

Tests are implemented to ensure the dataset remains consistent. Example checks:

- Row count within expected range
- Price values within acceptable bounds

### 5. Train / Validation / Test Split

The pipeline uses the reusable component `train_val_test_split`, which produces:

- `trainval_data.csv`
- `test_data.csv`

### 6. Model Training

A Random Forest Regressor is trained to predict rental prices. Features include:

- Categorical variables
- Numerical variables
- NLP features extracted from listing titles using TF-IDF

Metrics tracked:
- R² score
- Mean Absolute Error (MAE)

The trained model is exported as the artifact `random_forest_export`.

## Hyperparameter Optimization

Hyperparameters are optimized using Hydra multi-run:
```bash
mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.max_tfidf_features=10,15,30 modeling.random_forest.max_features=0.1,0.33,0.5,0.75,1 -m"
```

This generates multiple runs tracked automatically in Weights & Biases.

## Model Selection

The best model is selected based on lowest MAE. Using the W&B dashboard:

1. Switch to **Table view**
2. Add columns: ID, Job Type, `mae`, `r2`
3. Sort by MAE ascending

The best model artifact is then tagged `prod`.

## Model Testing

The production model is evaluated on the test set:
```bash
mlflow run . -P steps=test_regression_model
```

Parameters used:
- `mlflow_model=random_forest_export:prod`
- `test_dataset=test_data.csv:latest`

### Visualizing the Pipeline

The pipeline graph can be viewed in **Weights & Biases → Artifacts → Graph View**, showing how datasets and models depend on each other.

## Release the Pipeline

Once the best hyperparameters are identified:

1. Update `config.yaml` with optimal values.
2. Create a GitHub release (e.g. `v1.0.0`).

### Training on New Data

To retrain the model using a new dataset:
```bash
mlflow run https://github.com/<YOUR_USERNAME>/<REPOSITORY>.git \
  -v 1.0.0 \
  -P hydra_options="etl.sample='sample2.csv'"
```

## Troubleshooting

### Reset MLflow Environments

If an environment becomes corrupted, list all MLflow environments:
```bash
conda info --envs | grep mlflow | cut -f1 -d" "
```

Then remove them:
```bash
for e in $(conda info --envs | grep mlflow | cut -f1 -d" "); do conda uninstall --name $e --all -y; done
```

## License

See [LICENSE.txt](LICENSE.txt).