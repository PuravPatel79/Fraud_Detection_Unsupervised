# Unsupervised Financial Transaction Fraud Detection System

## Project Overview

This project implements an unsupervised learning system to detect potentially fraudulent or anomalous behavior in financial transactions. It utilizes the PaySim dataset, a synthetic dataset that mimics mobile money transactions with known fraudulent instances used for evaluation purposes only (not during unsupervised model training). The primary models explored are Isolation Forest and Variational Autoencoders (VAEs), chosen for their suitability in uncovering rare patterns and outliers in high-dimensional data.

The system is developed in Python and involves data preprocessing, feature engineering, model training, hyperparameter tuning, and comprehensive evaluation including the generation of synthetic "normal" transaction data by the VAE.

## File Structure

├── main.py                     # Trains models (Isolation Forest, multiple VAEs) and saves artifacts.
├── eval_visualize.py           # Loads trained models, performs evaluations, generates all plots and synthetic data.
├── preprocessor.joblib         # Saved scikit-learn preprocessor.
├── isolation_forest.joblib     # Saved trained Isolation Forest model.
├── vae_model_.keras            # Saved trained VAE models (e.g., vae_model_ld32_ep30_bs512_lr001.keras).
├── synthetic_normal_transactions_{PRIMARY_VAE_ID_FOR_DETAILED_PLOTS}.csv # Example CSV of VAE-generated synthetic data.
├── PS_20174392719_1491204439457_log.csv # The PaySim dataset (not included in submission, but required to run).
├── requirements.txt            # Lists all Python package dependencies.
├── roc_curves_combined.png     # Output: Combined ROC curves.
├── precision_recall_curves_combined.png # Output: Combined PR curves.
├── tsne_data_dist_with_anom_synth.jpg # Output: t-SNE visualization.
├── anomaly_score_scatter_if_vs_vae_{PRIMARY_VAE_ID_FOR_DETAILED_PLOTS}*.png # Output: Anomaly score scatter plot.
└── README.md                   # This file.
## Setup and Installation

### Environment Setup

1.  **Clone the Repository (if applicable) or Download Project Files.**
    ```bash
    # git clone <repository_url>
    # cd <project_directory>
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    * Using `venv`:
      ```bash
      python -m venv .venv
      # On Windows
      # .venv\Scripts\activate
      # On macOS/Linux
      # source .venv/bin/activate
      ```

3.  **Install Required Libraries:**
    The necessary libraries are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    **Key Libraries:**
    * `pandas`
    * `numpy`
    * `scikit-learn` (Ensure version >= 1.0 for full `ColumnTransformer` compatibility, e.g., 1.6.1 as used in debugging)
    * `tensorflow`
    * `matplotlib`
    * `seaborn`
    * `joblib` (usually included with scikit-learn)

    In your activated virtual environment, run:
    ```bash
    pip freeze > requirements.txt
    ```

4.  **Dataset:**
    * Download the PaySim dataset (`PS_20174392719_1491204439457_log.csv`).
    * Place it in the root directory of the project, or update the `DATASET_PATH` variable in the Python scripts if you place it elsewhere. The dataset can be found at: [https://www.kaggle.com/datasets/ealaxi/paysim1/data](https://www.kaggle.com/datasets/ealaxi/paysim1/data)

## Running the Code to Reproduce Results

The project is structured in two main phases:

### Phase 1: Model Training and Artifact Generation

This phase involves preprocessing the data, training the Isolation Forest and multiple Variational Autoencoder (VAE) configurations, and saving the trained preprocessor and models.

1.  **Configure `main.py` (Optional):**
    * Open `main.py`.
    * Adjust `N_SAMPLE_ROWS` if you want to train on a different sample size (e.g., 600000 as used in recent runs, or set to `None` to use the full dataset, though this will be very time-consuming).
    * Review the `VAE_CONFIGS_TO_TRAIN_AND_EVALUATE` list. This script will train a VAE for each configuration defined here and save it with a corresponding `id`. The current script trains multiple specified configurations.

2.  **Run `main.py`:**
    Execute the script from the project's root directory:
    ```bash
    python main.py
    ```
    * **Expected Output:** The script will print progress messages for data loading, preprocessing, Isolation Forest training, and each VAE configuration training.
    * **Generated Artifacts:**
        * `preprocessor.joblib`
        * `isolation_forest.joblib`
        * Multiple VAE model files, e.g., `vae_model_1.keras`, `vae_model_2.keras`, etc., based on the `id`s in `VAE_CONFIGS_TO_TRAIN_AND_EVALUATE`.

### Phase 2: Evaluation and Visualization

This phase loads the artifacts generated in Phase 1, performs evaluations, and generates all plots.

1.  **Configure `eval_visualize.py` (or your consolidated script name like `master_evaluation_and_visualization.py`):**
    * Open the script (e.g., `master_evaluation_and_visualization.py`).
    * **Crucially, ensure the `VAE_CONFIGS_TO_EVALUATE` list matches the VAE models you trained and want to compare.** The `id` for each entry must correspond to the suffix of a saved `.keras` model file (e.g., if `id` is `'1'`, it will look for `vae_model_1.keras`).
    * Set `PRIMARY_VAE_ID_FOR_DETAILED_PLOTS` to the `id` of the VAE model you want to use for the more detailed visualizations (like the IF-vs-VAE scatter plot, t-SNE including synthetic data derived from this VAE, and potentially score distributions if re-enabled). This ID must be one of those present in `VAE_CONFIGS_TO_EVALUATE` or at least a model file with that ID must exist.
    * Adjust `N_SAMPLE_ROWS_DATASET` if needed (should generally match what `main.py` used for consistency in deriving the test set).
    * You can also adjust visualization parameters like `DR_METHOD_FOR_EMBEDDING` (t-SNE or PCA).

2.  **Run `eval_visualize.py`:**
    Execute the script from the project's root directory:
    ```bash
    python eval_visualize.py
    ```
    * **Expected Output:** The script will print progress messages for loading data and models, evaluation metrics (ROC AUC, PR AUC, classification reports for each model configuration), and confirmations for saved plots and synthetic data.
    * **Generated Artifacts (Plots & Synthetic Data):**
        * `roc_curves_combined.png`
        * `precision_recall_curves_combined.png`
        * `tsne_data_dist_with_anom_synth.png`
        * `anomaly_score_scatter_if_vs_PRIMARY_VAE_NAME.png`
        * `synthetic_normal_transactions_generated_by_PRIMARY_VAE_ID.csv`

## Reproducing Specific Results from the Paper

To reproduce the key results (e.g., for your best performing VAE configuration):
1.  Ensure `main.py` is run with the `N_SAMPLE_ROWS` set to the value used for the reported results (e.g., 60000 or 600000). The `VAE_CONFIGS_TO_TRAIN_AND_EVALUATE` list in `main.py` should include the hyperparameter set corresponding to your best model.
2.  In `eval_visualize.py`:
    * Make sure `N_SAMPLE_ROWS_DATASET` matches that of `main.py`.
    * Ensure the `VAE_CONFIGS_TO_EVALUATE` list includes the entry for your best model.
    * Set `PRIMARY_VAE_ID_FOR_DETAILED_PLOTS` to the `id` of your best model.
3.  Running `main.py` followed by `eval_visualize.py` should then regenerate the plots and metrics presented in the paper. The console output of `eval_visualize.py` will contain the numerical AUC values and classification reports.

## Notes

* Training VAEs, especially multiple configurations or on large datasets, can be computationally intensive and time-consuming.
* Ensure your Python environment and library versions are consistent to guarantee reproducibility. The provided `requirements.txt` should facilitate this.
* The `RANDOM_STATE` variable is used across scripts to ensure consistent data sampling and model initializations where applicable, aiding reproducibility.