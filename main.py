import os
# Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

# --- Configuration ---
DATASET_PATH = 'PS_20174392719_1491204439457_log.csv'
N_SAMPLE_ROWS = 60000  # Number of rows to sample, set to None for a full dataset
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42

# --- Custom VAE Layers and Model Definition ---
class Sampling(layers.Layer):
    """Reparameterization trick: Z = mu + exp(0.5 * log_var) * epsilon."""
    def call(self, inputs):
        mu, log_var = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * log_var) * epsilon

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class VAE(Model):
    """Variational Autoencoder."""
    def __init__(self, input_dim, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.input_dim_val = input_dim
        self.latent_dim_val = latent_dim

        # Encoder
        self.encoder_dense1 = layers.Dense(64, activation='relu', name="encoder_dense1")
        self.encoder_dense2 = layers.Dense(32, activation='relu', name="encoder_dense2")
        self.mu_layer = layers.Dense(self.latent_dim_val, name='z_mean')
        self.logvar_layer = layers.Dense(self.latent_dim_val, name='z_log_var')
        self.sampling = Sampling(name="sampling_layer")

        # Decoder
        self.decoder_dense1 = layers.Dense(32, activation='relu', name="decoder_dense1")
        self.decoder_dense2 = layers.Dense(64, activation='relu', name="decoder_dense2")
        self.output_layer = layers.Dense(self.input_dim_val, activation='linear', name="decoder_output")

    def call(self, inputs):
        x = self.encoder_dense1(inputs)
        x = self.encoder_dense2(x)
        mu = self.mu_layer(x)
        log_var = self.logvar_layer(x)
        z = self.sampling((mu, log_var))
        x_dec = self.decoder_dense1(z)
        x_dec = self.decoder_dense2(x_dec)
        reconstruction = self.output_layer(x_dec)
        
        # VAE loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(inputs - reconstruction), axis=1)
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        )
        total_loss = reconstruction_loss + kl_loss
        self.add_loss(total_loss)
        return reconstruction

    def get_config(self):
        config = super().get_config()
        # Storing the actual dimensions used by this instance
        config.update({"input_dim": self.input_dim_val, "latent_dim": self.latent_dim_val})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def main():
    # 1. Load Data
    print(f"Loading dataset from: {DATASET_PATH}")
    df_full = pd.read_csv(DATASET_PATH)
    if N_SAMPLE_ROWS is not None and N_SAMPLE_ROWS < len(df_full):
        print(f"Sampling {N_SAMPLE_ROWS} rows...")
        df_sample = df_full.sample(n=N_SAMPLE_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        print("Using full dataset (or N_SAMPLE_ROWS >= dataset size).")
        df_sample = df_full.copy().reset_index(drop=True)
    print(f"Sampled dataset shape: {df_sample.shape}")

    # 2. Split Data
    print(f"Splitting data into train and test sets (test_size={TEST_SET_SIZE})...")
    train_df, test_df = train_test_split(df_sample, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE)
    print(f"Train_df shape: {train_df.shape}, Test_df shape: {test_df.shape}")

    # 3. Feature Engineering
    print("Starting Feature Engineering...")
    large_amount_threshold = train_df['amount'].quantile(0.95)
    print(f"Large amount threshold (95th percentile of training amount): {large_amount_threshold:.2f}")

    for df_iter in [train_df, test_df]:
        df_iter['errorBalanceOrg'] = df_iter['oldbalanceOrg'] - df_iter['amount'] - df_iter['newbalanceOrig']
        df_iter['is_large_amount'] = (df_iter['amount'] > large_amount_threshold).astype(int)
        df_iter['is_merchant_receiver'] = df_iter['nameDest'].str.startswith('M').astype(int)
        df_iter['amountToBalanceOrgRatio'] = df_iter['amount'] / (df_iter['oldbalanceOrg'] + 1e-6)
        df_iter['hourOfDay'] = df_iter['step'] % 24
        df_iter.replace([np.inf, -np.inf], 0, inplace=True)
    print("Engineered features added to train_df and test_df.")

    # 4. Data Analysis (Fraud Distribution) - Optional extensive printout
    print("\n--- Data Split Analysis ---")
    for df_name, df_content in [('train_df', train_df), ('test_df', test_df)]:
        print(f"\n--- Analysis of {df_name} ---")
        if 'isFraud' in df_content.columns: # Check if 'isFraud' column exists
            fraud_counts = df_content['isFraud'].value_counts()
            fraud_percentage = df_content['isFraud'].value_counts(normalize=True) * 100
            print(f"Shape of {df_name}: {df_content.shape}")
            print(f"Fraud counts in {df_name}:\n{fraud_counts}")
            print(f"\nFraud percentages in {df_name}:\n{fraud_percentage}")
            if 1 in fraud_counts: print(f"Number of actual fraud cases in {df_name}: {fraud_counts[1]}")
            else: print(f"No actual fraud cases found in {df_name}.")
        else:
            print(f"'isFraud' column not found in {df_name} for analysis printout.")
    print("--- END Data Split Analysis ---\n")

    # 5. Preprocessing (Done once for all models)
    print("Setting up and fitting preprocessing pipeline...")
    numeric_features  = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        'errorBalanceOrg', 'is_large_amount', 'is_merchant_receiver',
        'amountToBalanceOrgRatio', 'hourOfDay'
    ]
    categorical_feats = ['type']
    features_for_model = numeric_features + categorical_feats
    
    # Sanity check features before creating ColumnTransformer
    for feature in features_for_model:
        if feature not in train_df.columns:
            print(f"FATAL ERROR: Feature '{feature}' expected for preprocessing not found in train_df. Columns are: {train_df.columns.tolist()}")
            return
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_feats)
        ],
        remainder='drop'
    )
    X_train_features_df = train_df[features_for_model]
    X_test_features_df = test_df[features_for_model]

    X_train = preprocessor.fit_transform(X_train_features_df)
    X_test  = preprocessor.transform(X_test_features_df)
    joblib.dump(preprocessor, 'preprocessor.joblib')
    print(f"Preprocessor saved. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 6. Train Isolation Forest (Done once)
    print("\nTraining Isolation Forest...")
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=RANDOM_STATE, n_jobs=-1)
    iso_forest.fit(X_train)
    joblib.dump(iso_forest, 'isolation_forest.joblib')
    print("Isolation Forest trained and saved.")

    # 7. VAE Hyperparameter Training Loop
    print("\n--- Starting VAE Hyperparameter Training Loop ---")
    
    vae_configs = [
        # Different configs, we tested out many different configs and these gave us the best results till now.
        {'id': '1', 'latent_dim': 4, 'epochs': 30, 'batch_size': 256, 'learning_rate': 0.001},
        {'id': '2', 'latent_dim': 4, 'epochs': 30, 'batch_size': 512, 'learning_rate': 0.003},
        {'id': '3', 'latent_dim': 32, 'epochs': 50, 'batch_size': 256, 'learning_rate': 0.002},
        {'id': '4', 'latent_dim': 32, 'epochs': 30, 'batch_size': 512, 'learning_rate': 0.002},
            ]

    input_dim_for_vae = X_train.shape[1]

    for config in vae_configs:
        print(f"\nTraining VAE with config: {config}")
        
        current_latent_dim = config['latent_dim']
        current_epochs = config['epochs']
        current_batch_size = config['batch_size']
        current_lr = config['learning_rate']
        model_id = config['id']

        # Instantiate VAE with current configuration
        vae = VAE(input_dim=input_dim_for_vae, latent_dim=current_latent_dim)
        vae.compile(optimizer=Adam(learning_rate=current_lr))
        
        print(f"VAE params - input_dim: {input_dim_for_vae}, latent_dim: {current_latent_dim}, lr: {current_lr}, epochs: {current_epochs}, batch: {current_batch_size}")
        
        history = vae.fit(
            X_train, X_train,
            epochs=current_epochs,
            batch_size=current_batch_size,
            validation_data=(X_test, X_test),
            shuffle=True,
            verbose=1
        )
        
        model_save_path = f'vae_model_{model_id}.keras'
        try:
            vae.save(model_save_path)
            print(f"VAE with config {model_id} trained and saved to {model_save_path}")
        except Exception as e:
            print(f"ERROR saving VAE model {model_id}: {e}")


    print("\n--- VAE Hyperparameter Training Loop Complete ---")
    print("\nmain.py execution complete.")

if __name__ == '__main__':
    main()