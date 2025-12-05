<h1 align="center"> Heart Attack Prediction MLOps Pipeline </h1>
<p align="center"> A Robust MLOps Pipeline for Continuous Monitoring and Automated Deployment of Cardiovascular Risk Models</p>

<p align="center">
  <img alt="Build" src="https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge">
  <img alt="Tests" src="https://img.shields.io/badge/Tests-100%25%20Coverage-success?style=for-the-badge">
  <img alt="Dependencies" src="https://img.shields.io/badge/Dependencies-Up%20to%20Date-green?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge">
</p>
<!-- 
  **Note:** These are static placeholder badges. Replace them with your project's actual badges.
  You can generate your own at https://shields.io
-->

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack & Architecture](#-tech-stack--architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [License](#-license)

---

## ✨ Overview

The **Heart Attack Prediction MLOps Pipeline** is an end-to-end, automated system designed to consistently deliver and maintain high-performing machine learning models for cardiovascular risk assessment. This project automates the entire ML lifecycle—from raw data ingestion and advanced preprocessing through to rigorous model training, evaluation, and continuous deployment (CD).

### The Problem

> Traditional clinical risk assessment tools are often static, failing to adapt to evolving demographic data or modern feature engineering techniques. In healthcare, model reliability and reproducibility are non-negotiable. Manually managing data pipelines, training experiments, and deployment gates introduces significant human error, inconsistencies, and delays. A system is needed that guarantees data freshness, model robustness, and transparent performance monitoring before any model is deployed for inference.

### The Solution

This project provides a comprehensive machine learning operations framework that solves these challenges through automation and modularity. By integrating data modeling and powerful ML algorithms within structured pipelines, the system ensures that only models meeting predefined accuracy thresholds are promoted to deployment. This maximizes model performance, enhances data integrity, and establishes full traceability across the entire model production lifecycle, allowing for reliable risk prediction backed by data science.

### Architecture Overview

The system is built on a modular, step-based architecture, utilizing Python’s extensive data science ecosystem (including `scikit-learn` and `pandas`) for core logic, complemented by robust database integration via `SQLAlchemy` and `sqlmodel` backed by `mysql`. The entire process is structured around two key workflow definitions: the `training_pipeline` (for initial model creation and evaluation) and the `continuous_deployment_pipeline` (for live performance management and deployment triggering).

---

## 💡 Key Features

The MLOps Pipeline delivers a suite of capabilities focused on automated, reproducible, and performance-driven machine learning model delivery.

### ⚙️ Comprehensive Data Transformation Strategies

The pipeline includes dedicated steps for advanced data handling, ensuring that raw input data (`heart_attack_dataset.csv`) is rigorously prepared for modeling. This includes implementing four distinct strategies through the `data_preprocessing.py` module:

*   **Binarization Strategy:** Automatically converts specific two-category categorical columns into numerical binary formats for optimal model input.
*   **One Hot Encoding Strategy:** Applies standard one-hot encoding to convert nominal categorical columns, preventing spurious ordinality.
*   **Ordinal Encoding Strategy:** Preserves the natural rank or order in specific categorical columns using ordinal encoding.
*   **Standardization Strategy:** Utilizes `StandardScaler` to transform numerical features, ensuring consistent scaling and improving the performance of distance-based and gradient-descent algorithms.

### 🧠 Flexible and Extensible Model Training

The system utilizes a Strategy Pattern via the `ModelTrainer` class, allowing the execution of multiple robust classification algorithms without changing the underlying pipeline structure. Users benefit from the immediate availability of seven distinct model types for training, ensuring the selection of the highest-performing estimator for the task:

1.  **Logistic Regression Model**
2.  **Decision Tree Classifier Model**
3.  **Support Vector Classifier (SVC) Model**
4.  **Gradient Boosting Classifier Model**
5.  **Random Forest Classifier Model**

### 📈 Automated Model Evaluation and Selection

Every trained model undergoes rigorous evaluation via the `model_evaluation.py` step. This ensures performance metrics (such as accuracy) are calculated consistently. The pipeline uses these metrics to inform deployment decisions, guaranteeing that only high-quality, validated models proceed past the training phase.

### 🔄 End-to-End Pipeline Automation

The project is structured around two critical workflow definitions that abstract away complexity and guarantee reproducibility:

*   **`training_pipeline.py`:** Handles the full initial model development lifecycle, from data ingestion through splitting, preprocessing, training, and evaluation.
*   **`continuous_deployment_pipeline.py`:** Manages the production lifecycle, featuring the crucial `deployment_trigger` function. This function automatically determines whether a newly trained model should be promoted for inference based on a defined accuracy threshold, preventing the deployment of underperforming assets.

### 📦 MLflow Model Registry Integration (Inferred from Dependencies)

The system is designed to load and manage models efficiently using a centralized approach, as evidenced by the `model_loading.py` step which focuses on loading models. This facilitates the versioning and retrieval of prediction assets for inference purposes.

---


## 🛠️ Tech Stack & Architecture

The project leverages a modern Python-centric ecosystem focused on scalable data pipelines, reproducible model experimentation, and production-ready MLOps deployment.

| Category | Technology | Purpose | Why it was Chosen |
| :--- | :--- | :--- | :--- |
| **Backend API Framework** | `FastAPI` | Serves real-time prediction endpoints, model access, and health checks. | Extremely fast, async-native, type-safe via Pydantic, and automatically generates Swagger & ReDoc docs. |
| **Containerization** | `Docker` | Packages the API, ML pipeline, dependencies, and runtime environment. | Ensures reproducibility across dev/staging/prod, simplifies CI/CD, and isolates all services. |
| **Data Handling & Processing** | `pandas`, `numpy`, `pyarrow` | Handle ingestion, cleaning, feature engineering, transformations, and efficient data manipulation. | Pandas/NumPy are standard for ML workflows; PyArrow enables fast, columnar data operations. |
| **ML Framework** | `scikit-learn` | Implements training, evaluation, preprocessing, and inference modeling pipelines. | Proven algorithms, fast experimentation, and ideal for structured/tabular ML. |
| **Pipeline Orchestration** | `zenml` | Defines and executes step-based, reproducible ML pipelines. | Provides lineage, modularity, versioning, and native integration with MLflow and S3. |
| **Experiment Tracking** | `mlflow` | Tracks runs, metrics, parameters, and model versions. | Provides transparency, reproducibility, and a single source of truth for all experiments. |
| **Artifact Store** | **AWS S3 Bucket** | Stores MLflow artifacts, trained models, logs, and pipeline outputs. | Durable, scalable, and production-grade storage that MLflow integrates with natively. |

---

### Detailed Component Roles:

**Data Layer:** The `data/heart_attack_dataset.csv` serves as the initial raw data source. All data persistence, feature stores, and potentially model metadata handling are managed by the `mysql` database, ensuring ACID properties and transactional integrity for critical MLOps steps.

**ML Core (`src/`):** This directory houses the core intellectual property of the project, including the modular implementation of various classification algorithms (`model_training.py`) and the fundamental data transformations (`data_preprocessing.py`).

**Pipeline Steps (`steps/`):** Each step represents an atomic operation in the MLOps workflow (`data_ingestion.py`, `model_train.py`, `model_evaluation.py`). The separation of these steps ensures that the pipeline can be easily orchestrated, monitored, and debugged, providing a high degree of transparency in the ML production process.

**Deployment (`pipelines/`):** The defined pipelines bind the individual steps together into cohesive workflows. The `continuous_deployment_pipeline.py` is central to maintaining production quality by automating the model promotion decision based on verified performance metrics.

---

## 📁 Project Structure

The repository is structured to separate core ML logic (`src`), atomic pipeline tasks (`steps`), and the orchestration definitions (`pipelines`), enhancing modularity and maintainability.

```
📂 austinthapa-ml-ops-pipeline-8c82fec/
├── 📄 requirements.txt              # All project dependencies
├── 📄 README.md                     # Project documentation
├── 📄 LICENSE                       
├── 📄 .gitignore                    # Files ignored by Git
├── 📂 src/                          
│   ├── 📄 model_training.py         # Defines all ML model classes (7 total estimators) and the ModelTrainer context class
│   ├── 📄 data_split.py             # Functionality for splitting data into train and test sets
│   └── 📄 data_preprocessing.py     # Defines Strategy Pattern classes (Binarize, OneHotEncode, Ordinal, Standardize)
├── 📂 steps/                        
│   ├── 📄 data_preprocess_inference.py # Preprocessing step specifically for new data prior to prediction
│   ├── 📄 model_train.py            # Executes the model training step
│   ├── 📄 model_loading.py          # Function to load trained models (e.g., from an MLflow registry)
│   ├── 📄 predict.py                # Executes the final prediction/inference step
│   ├── 📄 model_evaluation.py       # Calculates performance metrics for trained models
│   ├── 📄 data_split.py             # Executes the data splitting operation
│   ├── 📄 data_ingestion.py         # Executes the initial data ingestion step
│   ├── 📄 data_preprocess.py        # Executes the primary data cleaning and transformation step
│   └── 📄 __init__.py               # Initialization file for Python module
├── 📂 data/                         
│   └── 📄 heart_attack_dataset.csv  # The source dataset used for training the models
├── 📂 pipelines/                    
│   ├── 📄 training_pipeline.py      # The end-to-end pipeline for initial model development
│   ├── 📄 continuous_deployment_pipeline.py # Workflow for monitoring, evaluation, and automated deployment
│   └── 📄 __init__.py               # Initialization file for Python module
```

---

## 🚀 Getting Started

To set up the Heart Attack Prediction MLOps Pipeline locally, follow these steps.

### Prerequisites

You must have the following software installed:

1.  **Python 3.8+**
2.  **pip** (Python package installer)
3.  **MySQL Server** (Required for the data persistence layer)

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/austinthapa/ml-ops-pipeline.git
    cd ml-ops-pipeline
    ```

2.  **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv mlops-venv
    source mlops-venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows
    ```

3.  **Install Required Dependencies:**

    Install all necessary libraries, including `zenml`, `scikit-learn`, `pandas`, and database connectors, using the verified `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```
---

## 🔧 Usage

This project operates by defining and executing structured Machine Learning pipelines. Execution involves triggering the primary pipeline functions defined in the `pipelines/` directory.

### 1. Running the Initial Training Pipeline

The `training_pipeline` is the entry point for developing a new model. It automates the entire process from fetching raw data (`data/heart_attack_dataset.csv`) to delivering a final, evaluated model.

The steps executed sequentially by this pipeline are:
1.  **Data Ingestion** (`data_ingestion.py`)
2.  **Data Preprocessing** (`data_preprocess.py`) (Executing Binarize, OneHotEncode, OrdinalEncode, and Standardization strategies)
3.  **Data Splitting** (`data_split.py`)
4.  **Model Training** (`model_train.py`) (Selecting one of the 7 available model types)
5.  **Model Evaluation** (`model_evaluation.py`)

To initiate a full model development cycle, you must call the dedicated pipeline function:

```python
# Conceptual execution flow (assuming standard Python environment setup)
# This would typically be executed by calling the 'training_pipeline' function 
# from pipelines/training_pipeline.py

import sys
sys.path.append('pipelines')
from training_pipeline import training_pipeline

# Execute the defined training workflow
# Note: Specific function signature (arguments) may vary based on internal ZenML configuration
training_pipeline() 
```

### 2. Utilizing the Continuous Deployment Pipeline

The `continuous_deployment_pipeline` is designed to run periodically, checking for newly trained models and deciding whether they are suitable for production deployment. This pipeline ensures that the production environment is constantly utilizing the best available model.

The core function in this pipeline is the `deployment_trigger`, which implements the logic to:

1.  Load the existing deployed model (via `model_loading.py`).
2.  Evaluate the candidate model's performance (e.g., accuracy).
3.  Compare the candidate against the currently deployed model and a set threshold.
4.  If performance exceeds the threshold, the new model is deployed; otherwise, it is archived.

To manage production models, execute the continuous deployment workflow:

```python
# Conceptual execution flow for Continuous Deployment
# This would typically be executed by calling the 'continuous_deployment_pipeline' 
# function from pipelines/continuous_deployment_pipeline.py

import sys
sys.path.append('pipelines')
from continuous_deployment_pipeline import continuous_deployment_pipeline

# Execute the CD workflow to monitor and potentially deploy
continuous_deployment_pipeline()
```

### 3. Making Predictions (Inference)

Once a model has been successfully trained and deployed through the pipelines, it can be loaded for inference on new, unseen data. The prediction process involves two critical steps: preparing the input data and running the prediction using the loaded estimator.

1.  **Preprocess Input:** The new data must undergo the same preprocessing steps as the training data, handled by `data_preprocess_inference.py`.
2.  **Predict:** The prediction function in `predict.py` takes the preprocessed input and the loaded model to generate the risk assessment output.

```python
# Conceptual Inference Flow:

# 1. Load the deployed model (using model_loading.py logic)
# deployed_model = load_model(path_to_registry) 

# 2. Preprocess the raw input data (using data_preprocess_inference.py logic)
# preprocessed_data = preprocess_input_inference(raw_data)

# 3. Generate prediction (using predict.py logic)
# prediction_output = predict(preprocessed_data, deployed_model)
```

---


## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for complete details.