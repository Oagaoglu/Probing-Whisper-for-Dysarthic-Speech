import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Setup logging
logging.basicConfig(filename='training_experimental.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Load data
def load_data(representation_file, acoustic_file):
    # Load acoustic features
    acoustic_df = pd.read_csv(acoustic_file)

    # Load representation features from the specified block
    rep_df = pd.read_csv(representation_file)

    # Merge the two DataFrames on 'speaker_id' and 'file_id'
    merged_df = pd.merge(rep_df, acoustic_df, on=['speaker_id', 'file_id'])

    return merged_df

# Preprocess data
def preprocess_data(data, acoustic_features):
    # Ensure no NaN or Inf values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    return data

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=100, patience=20):
    best_loss = float('inf')
    patience_counter = 0
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.float(), targets.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation loss
        val_loss = running_loss / len(train_loader)
        if np.isnan(val_loss) or np.isinf(val_loss):
            logging.error("Encountered NaN or Inf in loss values, stopping training.")
            break
        val_losses.append(val_loss)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info("Early stopping triggered")
            break
    
    return best_loss

# Save results to CSV file
def save_results(results, csv_filename):
    df = pd.DataFrame([results])
    df.to_csv(csv_filename, mode='a', header=not os.path.exists(csv_filename), index=False)

# Main function for training on each block separately for each feature
def main():
    representation_files = [
        './data_folder/block_2/experimental_features.csv',
        './data_folder/block_3/experimental_features.csv',
        './data_folder/block_4/experimental_features.csv',
        './data_folder/block_5/experimental_features.csv',
        './data_folder/block_6/experimental_features.csv',
        './data_folder/block_7/experimental_features.csv',
    ]
    acoustic_file = './data_folder/experimental_acoustic_features.csv'  # Use cleaned file
    results_file = 'results_experimental.csv'
    
    # List of acoustic features to predict
    acoustic_features = [
        'mean_F2_bandwidth', 'mean_harmonic_difference_H1_A3', '50th_percentile_pitch_semitone',
        'mean_harmonic_difference_H1_H2', 'mean_spectral_slope_500_1500_voiced', 'cv_mfcc3',
        'cv_spectral_flux', 'maximal_intensity', 'cv_HNR', 'cv_F2', 'cv_F3_bandwidth',
        'loudness', 'logHNR'
    ]

    # Ensure results file is empty initially
    if os.path.exists(results_file):
        os.remove(results_file)
    
    for block_path in representation_files:
        block = os.path.basename(os.path.dirname(block_path))  # Get the block name from the directory
        logging.info(f"Processing {block}...")
        for feature in acoustic_features:
            logging.info(f"Predicting {feature} from {block}...")
            # Load and preprocess data
            data = load_data(block_path, acoustic_file)
            #data = preprocess_data(data, acoustic_features)

            # Split data into inputs (X) and outputs (y)
            X = data.drop(columns=['speaker_id', 'file_id'] + acoustic_features)
            y = data[[feature]]

            # Standardize the data
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X = scaler_X.fit_transform(X)
            y = scaler_y.fit_transform(y)

            # Split into training and testing sets (no validation set for simplicity)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Convert to PyTorch tensors
            train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
            test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Define model, loss function, and optimizer
            input_size = X.shape[1]
            output_size = y.shape[1]
            model = MLP(input_size, output_size)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Train the model
            best_val_loss = train_model(model, train_loader, criterion, optimizer, num_epochs=100, patience=10)

            # Evaluate the model
            model.eval()
            with torch.no_grad():
                test_loss = 0.0
                for inputs, targets in test_loader:
                    inputs, targets = inputs.float(), targets.float()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                test_loss /= len(test_loader)
                logging.info(f"Test Loss for {feature} from {block}: {test_loss:.4f}")

            # Save results immediately
            result = {
                'block': block,
                'feature': feature,
                'validation_loss': best_val_loss,
                'test_loss': test_loss
            }
            save_results(result, results_file)

if __name__ == "__main__":
    main()
