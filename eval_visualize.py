import os
# Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import tensorflow as tf
from tensorflow.keras import layers, Model

# --- Configuration ---
DATASET_PATH = 'PS_20174392719_1491204439457_log.csv'
PREPROCESSOR_PATH = 'preprocessor.joblib'
ISO_FOREST_PATH = 'isolation_forest.joblib'

# VAE Model configurations to evaluate for ROC/PR curves
# Update 'id' to match the suffix of the saved Keras model files (1, 2, 3 & 4)

VAE_CONFIGS_TO_EVALUATE = [
    # Example from user's successful plot:
    {'id': '1', 'name': 'VAE 1 (LD 4, B 256)', 'style': 'g'},
    {'id': '2', 'name': 'VAE 2 (LD 4, B 512)', 'style': 'r'},
    {'id': '3', 'name': 'VAE 3 (LD 32, B 256)', 'style': 'c'},
    {'id': '4', 'name': 'VAE 4 (LD 32, B 512)', 'style': 'm'},
]

# ID of the VAE model to use for detailed plots (t-SNE, scatter) and synthetic data generation
# Choose an ID that exists in your vae_model_{id}.keras files
PRIMARY_VAE_ID_FOR_DETAILED_PLOTS = '3' # Use the ID of the best VAE here
PRIMARY_VAE_MODEL_PATH = f'vae_model_{PRIMARY_VAE_ID_FOR_DETAILED_PLOTS}.keras'
SYNTHETIC_DATA_FILENAME = f'synthetic_normal_transactions_{PRIMARY_VAE_ID_FOR_DETAILED_PLOTS}.csv'


N_SAMPLE_ROWS_DATASET = 600000
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42

N_POINTS_REAL_FOR_VIS = 10000
N_POINTS_SYNTHETIC_FOR_VIS = 2000 # Total synthetic samples to generate for t-SNE
N_POINTS_SCATTER_PLOT_PER_CATEGORY = 500
ANOMALOUS_SYNTHETIC_PERCENTILE_THRESHOLD = 95

DR_METHOD_FOR_EMBEDDING = 'TSNE'

# --- Custom VAE Layers and Model Definition (MUST MATCH main.py) ---
class Sampling(layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        batch, dim = tf.shape(mu)[0], tf.shape(mu)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * log_var) * epsilon
    def get_config(self): return super().get_config()
    @classmethod
    def from_config(cls, config): return cls(**config)

class VAE(Model):
    def __init__(self, input_dim, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.input_dim_val = input_dim
        self.latent_dim_val = latent_dim
        self.encoder_dense1 = layers.Dense(64, activation='relu', name="encoder_dense1")
        self.encoder_dense2 = layers.Dense(32, activation='relu', name="encoder_dense2")
        self.mu_layer = layers.Dense(self.latent_dim_val, name='z_mean')
        self.logvar_layer = layers.Dense(self.latent_dim_val, name='z_log_var')
        self.sampling = Sampling(name="sampling_layer")
        self.decoder_dense1 = layers.Dense(32, activation='relu', name="decoder_dense1")
        self.decoder_dense2 = layers.Dense(64, activation='relu', name="decoder_dense2")
        self.output_layer = layers.Dense(self.input_dim_val, activation='linear', name="decoder_output")
    
    def call(self, inputs):
        x = self.encoder_dense1(inputs)
        x = self.encoder_dense2(x)
        mu, log_var = self.mu_layer(x), self.logvar_layer(x)
        z = self.sampling((mu, log_var))
        x_dec = self.decoder_dense1(z)
        x_dec = self.decoder_dense2(x_dec)
        return self.output_layer(x_dec)
    
    def get_config(self):
        config = super().get_config()
        config.update({"input_dim": self.input_dim_val, "latent_dim": self.latent_dim_val})
        return config
    
    @classmethod
    def from_config(cls, config): return cls(**config)

# --- Feature definitions (MUST MATCH main.py) ---
numeric_features = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
    'errorBalanceOrg', 'is_large_amount', 'is_merchant_receiver',
    'amountToBalanceOrgRatio', 'hourOfDay'
]
categorical_feats = ['type']
features_for_model_input_order = numeric_features + categorical_feats

# --- Helper Functions ---
def load_and_prep_real_test_data(preprocessor_obj):
    print(f"\nLoading and preparing real test data from: {DATASET_PATH}")
    df_full = pd.read_csv(DATASET_PATH)
    if N_SAMPLE_ROWS_DATASET is not None and N_SAMPLE_ROWS_DATASET < len(df_full):
        df_sample = df_full.sample(n=N_SAMPLE_ROWS_DATASET, random_state=RANDOM_STATE)
    else:
        df_sample = df_full.copy()
    
    train_df_for_threshold, test_df_real_orig = train_test_split(
        df_sample, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )
    
    test_df_real = test_df_real_orig.copy()
    large_amount_threshold = train_df_for_threshold['amount'].quantile(0.95)
    
    test_df_real['errorBalanceOrg'] = test_df_real['oldbalanceOrg'] - test_df_real['amount'] - test_df_real['newbalanceOrig']
    test_df_real['is_large_amount'] = (test_df_real['amount'] > large_amount_threshold).astype(int)
    test_df_real['is_merchant_receiver'] = test_df_real['nameDest'].str.startswith('M').astype(int)
    test_df_real['amountToBalanceOrgRatio'] = test_df_real['amount'] / (test_df_real['oldbalanceOrg'] + 1e-6)
    test_df_real['hourOfDay'] = test_df_real['step'] % 24
    test_df_real.replace([np.inf, -np.inf], 0, inplace=True)
    
    X_real_test_features = test_df_real[features_for_model_input_order]
    X_real_test_processed = preprocessor_obj.transform(X_real_test_features)
    y_real_labels = test_df_real['isFraud'].astype(int).values
    
    print(f"Real test data prepared. Processed shape: {X_real_test_processed.shape}, Fraud count: {np.sum(y_real_labels)}")
    return X_real_test_processed, y_real_labels

def generate_and_preprocess_synthetic_data(vae_generator_model, preprocessor_obj, num_samples_to_generate):
    print(f"\nGenerating {num_samples_to_generate} synthetic samples using VAE: {vae_generator_model.name}")
    if hasattr(vae_generator_model, 'latent_dim_val'):
        vae_latent_dim = vae_generator_model.latent_dim_val
    else:
        vae_latent_dim = vae_generator_model.get_layer('z_mean').output_shape[-1]
    
    decoder_input_tensor = layers.Input(shape=(vae_latent_dim,), name=f"decoder_input_z_gen_{vae_generator_model.name}")
    temp_layer = vae_generator_model.get_layer('decoder_dense1')(decoder_input_tensor)
    temp_layer = vae_generator_model.get_layer('decoder_dense2')(temp_layer)
    decoder_output_tensor = vae_generator_model.get_layer('decoder_output')(temp_layer)
    decoder = Model(decoder_input_tensor, decoder_output_tensor, name=f"vae_decoder_for_gen_{vae_generator_model.name}")
    
    z_samples = np.random.normal(size=(num_samples_to_generate, vae_latent_dim))
    synthetic_scaled_for_inverse = decoder.predict(z_samples, batch_size=256)

    numeric_transformer = preprocessor_obj.named_transformers_['num']
    categorical_transformer = preprocessor_obj.named_transformers_['cat']
    num_numeric_cols = len(numeric_features)
    
    synthetic_numeric_original = numeric_transformer.inverse_transform(synthetic_scaled_for_inverse[:, :num_numeric_cols])
    synthetic_categorical_original = categorical_transformer.inverse_transform(synthetic_scaled_for_inverse[:, num_numeric_cols:])
    
    df_synthetic_numeric = pd.DataFrame(synthetic_numeric_original, columns=numeric_features)
    df_synthetic_categorical = pd.DataFrame(synthetic_categorical_original, columns=categorical_feats)
    df_synthetic = pd.concat([df_synthetic_numeric, df_synthetic_categorical], axis=1)
    df_synthetic = df_synthetic[features_for_model_input_order] 
    
    df_synthetic.to_csv(SYNTHETIC_DATA_FILENAME, index=False)
    print(f"Synthetic data generated and saved to {SYNTHETIC_DATA_FILENAME}")
    
    X_synthetic_processed = preprocessor_obj.transform(df_synthetic)
    print(f"Processed synthetic data shape: {X_synthetic_processed.shape}")
    return X_synthetic_processed

# --- Plotting Functions ---
def plot_combined_roc_curves(roc_plot_results_list):
    plt.figure(figsize=(10, 8))
    for res in roc_plot_results_list: plt.plot(res['fpr'], res['tpr'], res.get('style', '.-'), label=res['name'])
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curves'); plt.legend(loc='lower right'); plt.grid(True); plt.tight_layout()
    plt.savefig('roc_curves_combined.png'); print("\nCombined ROC curve plot saved.")

def plot_combined_pr_curves(pr_plot_results_list, y_true_labels):
    plt.figure(figsize=(10, 8))
    for res in pr_plot_results_list: plt.plot(res['rec'], res['prec'], res.get('style', '.-'), label=res['name'])
    baseline_pr = y_true_labels.mean()
    plt.plot([0, 1], [baseline_pr, baseline_pr], 'k--', label=f'No Skill (Baseline={baseline_pr:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Combined Precision-Recall Curves')
    plt.legend(loc='upper right'); plt.grid(True); plt.tight_layout()
    plt.savefig('precision_recall_curves_combined.png'); print("Combined PR curve plot saved.")

def plot_score_distributions_comparison(scores_collection_map):
    print("\n--- Plotting Anomaly Score Distributions Comparison ---")
    num_models_to_plot = len(scores_collection_map)
    if num_models_to_plot == 0: return
    fig, axes = plt.subplots(1, num_models_to_plot, figsize=(8 * num_models_to_plot, 6), sharey=True)
    if num_models_to_plot == 1: axes = [axes]
    for ax, (model_name, data_scores) in zip(axes, scores_collection_map.items()):
        if 'real_normal' in data_scores: sns.histplot(data_scores['real_normal'], color="skyblue", label='Real Normal', kde=True, stat="density", common_norm=False, ax=ax)
        if 'synthetic_normal' in data_scores: sns.histplot(data_scores['synthetic_normal'], color="lightgreen", label='Synthetic Normal', kde=True, stat="density", common_norm=False, alpha=0.7, ax=ax)
        if 'real_fraud' in data_scores: sns.histplot(data_scores['real_fraud'], color="salmon", label='Real Fraud', kde=True, stat="density", common_norm=False, ax=ax)
        ax.set_title(f'Scores ({model_name.replace("_", " ").title()})'); ax.set_xlabel('Anomaly Score'); ax.legend(); ax.grid(True)
    axes[0].set_ylabel('Density'); fig.suptitle('Comparison of Anomaly Score Distributions', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig('score_distributions_comparison.png')
    print("Comparison score distribution plot saved.")

def plot_data_embedding_with_anomalous_synthetic(X_rn_2d, X_rf_2d, X_sn_bulk_2d, X_sn_anom_2d, dr_method):
    print(f"\nGenerating {dr_method} scatter plot with anomalous synthetic data...")
    plt.figure(figsize=(13, 10))
    plt.scatter(X_rn_2d[:, 0], X_rn_2d[:, 1], label=f'Real Normal (n={X_rn_2d.shape[0]})', alpha=0.3, s=20, c='blue')
    if X_sn_bulk_2d is not None and X_sn_bulk_2d.shape[0] > 0:
        plt.scatter(X_sn_bulk_2d[:, 0], X_sn_bulk_2d[:, 1], label=f'Synthetic Normal (Bulk, n={X_sn_bulk_2d.shape[0]})', alpha=0.4, s=20, c='green', marker='x')
    if X_sn_anom_2d is not None and X_sn_anom_2d.shape[0] > 0:
        plt.scatter(X_sn_anom_2d[:, 0], X_sn_anom_2d[:, 1], label=f'Synthetic Anomalous (Top {100-ANOMALOUS_SYNTHETIC_PERCENTILE_THRESHOLD}%, n={X_sn_anom_2d.shape[0]})', alpha=0.6, s=30, c='purple', marker='^')
    plt.scatter(X_rf_2d[:, 0], X_rf_2d[:, 1], label=f'Real Fraud (n={X_rf_2d.shape[0]})', alpha=0.7, s=50, c='red', marker='o', edgecolors='k')
    plt.title(f'{dr_method} Visualization: Real, Synthetic Normal, Synthetic Anomalous, and Real Fraud')
    plt.xlabel(f'{dr_method} Component 1'); plt.ylabel(f'{dr_method} Component 2')
    plt.legend(loc='best'); plt.grid(True); plt.tight_layout()
    plt.savefig(f'{dr_method.lower()}_data_dist_with_anom_synth.png')
    print(f"{dr_method} plot with anomalous synthetic data saved.")

def plot_anomaly_score_scatter_comparison(if_scores_map, vae_scores_map, primary_vae_display_name="Primary VAE"):
    print("\nGenerating anomaly score scatter plot (IF vs Primary VAE)...")
    plt.figure(figsize=(14, 10))
    def get_plot_samples(scores1, scores2, n_max):
        if scores1 is None or scores2 is None or len(scores1) == 0: return np.array([]), np.array([])
        if len(scores1) > n_max:
            indices = np.random.choice(len(scores1), n_max, replace=False); return scores1[indices], scores2[indices]
        return scores1, scores2

    if_rn, vae_rn = get_plot_samples(if_scores_map.get('real_normal'), vae_scores_map.get('real_normal'), N_POINTS_SCATTER_PLOT_PER_CATEGORY)
    plt.scatter(if_rn, vae_rn, label=f'Real Normal (n={len(if_rn)})', alpha=0.2, s=20, c='blue')
    if if_scores_map.get('synthetic_normal_bulk') is not None:
        if_sn_bulk, vae_sn_bulk = get_plot_samples(if_scores_map['synthetic_normal_bulk'], vae_scores_map['synthetic_normal_bulk'], N_POINTS_SCATTER_PLOT_PER_CATEGORY)
        plt.scatter(if_sn_bulk, vae_sn_bulk, label=f'Synthetic Normal (Bulk, n={len(if_sn_bulk)})', alpha=0.4, s=20, c='green', marker='x')
    if if_scores_map.get('synthetic_anomalous') is not None:
        if_sa, vae_sa = get_plot_samples(if_scores_map['synthetic_anomalous'], vae_scores_map['synthetic_anomalous'], N_POINTS_SCATTER_PLOT_PER_CATEGORY)
        plt.scatter(if_sa, vae_sa, label=f'Synthetic Anomalous (Top {100-ANOMALOUS_SYNTHETIC_PERCENTILE_THRESHOLD}%, n={len(if_sa)})', alpha=0.6, s=30, c='purple', marker='^')
    if_rf, vae_rf = get_plot_samples(if_scores_map.get('real_fraud'), vae_scores_map.get('real_fraud'), N_POINTS_SCATTER_PLOT_PER_CATEGORY)
    plt.scatter(if_rf, vae_rf, label=f'Real Fraud (n={len(if_rf)})', alpha=0.7, s=50, c='red', marker='o', edgecolors='k')
    plt.xlabel('Isolation Forest Anomaly Score'); plt.ylabel(f'{primary_vae_display_name} Anomaly Score')
    plt.title(f'Anomaly Score Comparison: IF vs. {primary_vae_display_name}')
    plt.legend(loc='upper right'); plt.grid(True); plt.tight_layout()
    plt.savefig(f'anomaly_score_scatter_if_vs_{primary_vae_display_name.replace(" ","_").lower()}.png')
    print(f"Anomaly score scatter plot saved for IF vs {primary_vae_display_name}.")

# --- Main Execution ---
def main():
    print(f"TensorFlow Version: {tf.__version__}")
    # 1. Load Preprocessor
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Preprocessor loaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not load preprocessor from {PREPROCESSOR_PATH}. Error: {e}"); return

    # 2. Load and Prepare Real Test Data
    X_real_test_processed, y_real_labels = load_and_prep_real_test_data(preprocessor)
    if X_real_test_processed is None: return
    X_real_normal_processed = X_real_test_processed[y_real_labels == 0]
    X_real_fraud_processed = X_real_test_processed[y_real_labels == 1]

    # 3. Load Primary VAE and Generate/Process Synthetic Data
    X_synthetic_processed = None; primary_vae_model = None
    primary_vae_display_name = "Primary VAE" 
    try:
        primary_vae_model = tf.keras.models.load_model(PRIMARY_VAE_MODEL_PATH, custom_objects={'VAE': VAE, 'Sampling': Sampling}, compile=False)
        primary_vae_display_name = next((config['name'] for config in VAE_CONFIGS_TO_EVALUATE if config['id'] == PRIMARY_VAE_ID_FOR_DETAILED_PLOTS), PRIMARY_VAE_ID_FOR_DETAILED_PLOTS)
        print(f"Primary VAE model ({PRIMARY_VAE_MODEL_PATH} as '{primary_vae_display_name}') loaded.")
        X_synthetic_processed = generate_and_preprocess_synthetic_data(primary_vae_model, preprocessor, num_samples_to_generate=N_POINTS_SYNTHETIC_FOR_VIS)
    except Exception as e: print(f"Warning: Could not load primary VAE or process synthetic data. Error: {e}")

    # 4. Load Isolation Forest
    iso_forest = None
    try: iso_forest = joblib.load(ISO_FOREST_PATH); print("Isolation Forest model loaded.")
    except Exception as e: print(f"Warning: Could not load Isolation Forest. Error: {e}")

    # --- ROC/PR CURVES & CLASSIFICATION REPORTS ---
    roc_plot_data, pr_plot_data = [], []
    batch_s = 256
    if iso_forest:
        iso_scores = -iso_forest.decision_function(X_real_test_processed)
        fpr, tpr, _ = roc_curve(y_real_labels, iso_scores); roc_auc_val = auc(fpr, tpr)
        roc_plot_data.append({'name': f'Isolation Forest (AUC={roc_auc_val:.3f})', 'fpr': fpr, 'tpr': tpr, 'style': 'b-'})
        prec, rec, _ = precision_recall_curve(y_real_labels, iso_scores); pr_auc_val = auc(rec, prec)
        pr_plot_data.append({'name': f'Isolation Forest (PR AUC={pr_auc_val:.3f})', 'rec': rec, 'prec': prec, 'style': 'b-'})
        print(f"\nIF - ROC AUC: {roc_auc_val:.3f}, PR AUC: {pr_auc_val:.3f}")
        print(classification_report(y_real_labels, (iso_scores >= np.percentile(iso_scores, 95)).astype(int), zero_division=0, digits=3))
    for config in VAE_CONFIGS_TO_EVALUATE:
        model_path = f"vae_model_{config['id']}.keras"
        try:
            vae_eval = tf.keras.models.load_model(model_path, custom_objects={'VAE': VAE, 'Sampling': Sampling}, compile=False)
            vae_errors = np.mean(np.square(X_real_test_processed - vae_eval.predict(X_real_test_processed, batch_size=batch_s)), axis=1)
            fpr, tpr, _ = roc_curve(y_real_labels, vae_errors); roc_auc_val = auc(fpr, tpr)
            roc_plot_data.append({'name': f"{config['name']} (AUC={roc_auc_val:.3f})", 'fpr': fpr, 'tpr': tpr, 'style': config['style']})
            prec, rec, _ = precision_recall_curve(y_real_labels, vae_errors); pr_auc_val = auc(rec, prec)
            pr_plot_data.append({'name': f"{config['name']} (PR AUC={pr_auc_val:.3f})", 'rec': rec, 'prec': prec, 'style': config['style']})
            print(f"\n{config['name']} - ROC AUC: {roc_auc_val:.3f}, PR AUC: {pr_auc_val:.3f}")
            print(classification_report(y_real_labels, (vae_errors >= np.percentile(vae_errors, 95)).astype(int), zero_division=0, digits=3))
        except Exception as e: print(f"Could not evaluate VAE {config['id']}: {e}")
    if roc_plot_data: plot_combined_roc_curves(roc_plot_data)
    if pr_plot_data: plot_combined_pr_curves(pr_plot_data, y_real_labels)

    # --- DATA EMBEDDING PLOT (t-SNE) with ANOMALOUS SYNTHETIC ---
    if X_synthetic_processed is not None and primary_vae_model and X_real_normal_processed.shape[0]>0 and X_real_fraud_processed.shape[0]>0:
        # Get VAE scores for all synthetic data to identify anomalous ones
        vae_scores_all_synthetic_primary = np.mean(np.square(X_synthetic_processed - primary_vae_model.predict(X_synthetic_processed, batch_size=batch_s)), axis=1)
        synth_anom_thresh = np.percentile(vae_scores_all_synthetic_primary, ANOMALOUS_SYNTHETIC_PERCENTILE_THRESHOLD)
        anom_mask = vae_scores_all_synthetic_primary >= synth_anom_thresh
        
        X_synthetic_normal_bulk_processed = X_synthetic_processed[~anom_mask]
        X_synthetic_anomalous_processed = X_synthetic_processed[anom_mask]

        # Subsample for t-SNE visualization
        plot_X_rn = X_real_normal_processed[np.random.choice(X_real_normal_processed.shape[0], min(N_POINTS_REAL_FOR_VIS, X_real_normal_processed.shape[0]), replace=False)]
        plot_X_rf = X_real_fraud_processed
        plot_X_sn_bulk = X_synthetic_normal_bulk_processed[np.random.choice(X_synthetic_normal_bulk_processed.shape[0], min(N_POINTS_SYNTHETIC_FOR_VIS // 2, X_synthetic_normal_bulk_processed.shape[0]), replace=False)] if X_synthetic_normal_bulk_processed.shape[0] > 0 else np.array([])
        plot_X_sn_anom = X_synthetic_anomalous_processed[np.random.choice(X_synthetic_anomalous_processed.shape[0], min(N_POINTS_SYNTHETIC_FOR_VIS // 2, X_synthetic_anomalous_processed.shape[0]), replace=False)] if X_synthetic_anomalous_processed.shape[0] > 0 else np.array([])

        data_to_embed = np.vstack([arr for arr in [plot_X_rn, plot_X_rf, plot_X_sn_bulk, plot_X_sn_anom] if arr.shape[0] > 0])
        
        if data_to_embed.shape[0] > 1:
            print(f"\nShape of combined data for embedding with {DR_METHOD_FOR_EMBEDDING}: {data_to_embed.shape}")
            embedded_data_2d = None
            if DR_METHOD_FOR_EMBEDDING == 'TSNE':
                reducer = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, data_to_embed.shape[0]-1), max_iter=300, verbose=0); embedded_data_2d = reducer.fit_transform(data_to_embed)
            
            if embedded_data_2d is not None:
                idx1 = plot_X_rn.shape[0]; idx2 = idx1 + plot_X_rf.shape[0]; idx3 = idx2 + plot_X_sn_bulk.shape[0]
                plot_data_embedding_with_anomalous_synthetic(embedded_data_2d[:idx1], embedded_data_2d[idx1:idx2], embedded_data_2d[idx2:idx3], embedded_data_2d[idx3:], DR_METHOD_FOR_EMBEDDING)
    else: print("Skipping data embedding plot due to missing synthetic or real data components, or primary VAE.")

    # --- ANOMALY SCORE SCATTER PLOT (IF vs. Primary VAE) ---
    if iso_forest and primary_vae_model and X_synthetic_processed is not None:
        if_scores_all_synthetic = -iso_forest.decision_function(X_synthetic_processed) 
        
        if 'vae_scores_all_synthetic_primary' not in locals() and X_synthetic_processed is not None:
            vae_scores_all_synthetic_primary = np.mean(np.square(X_synthetic_processed - primary_vae_model.predict(X_synthetic_processed, batch_size=batch_s)), axis=1)
            synth_anom_thresh = np.percentile(vae_scores_all_synthetic_primary, ANOMALOUS_SYNTHETIC_PERCENTILE_THRESHOLD)
            anom_mask = vae_scores_all_synthetic_primary >= synth_anom_thresh


        if_scores_scatter_map = {
            'real_normal': -iso_forest.decision_function(X_real_normal_processed),
            'real_fraud': -iso_forest.decision_function(X_real_fraud_processed),
        }
        vae_scores_scatter_map = {
            'real_normal': np.mean(np.square(X_real_normal_processed - primary_vae_model.predict(X_real_normal_processed, batch_size=batch_s)), axis=1),
            'real_fraud': np.mean(np.square(X_real_fraud_processed - primary_vae_model.predict(X_real_fraud_processed, batch_size=batch_s)), axis=1),
        }
        if X_synthetic_processed is not None:
            if_scores_scatter_map['synthetic_normal_bulk'] = if_scores_all_synthetic[~anom_mask]
            if_scores_scatter_map['synthetic_anomalous'] = if_scores_all_synthetic[anom_mask]
            vae_scores_scatter_map['synthetic_normal_bulk'] = vae_scores_all_synthetic_primary[~anom_mask]
            vae_scores_scatter_map['synthetic_anomalous'] = vae_scores_all_synthetic_primary[anom_mask]

        plot_anomaly_score_scatter_comparison(if_scores_scatter_map, vae_scores_scatter_map, primary_vae_display_name)
    else: print("Skipping anomaly score scatter plot due to missing models or synthetic data.")

    print("\nMaster evaluation and visualization script execution complete.")

if __name__ == '__main__':
    main()