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
logging.basicConfig(filename=f'training.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Representation files based on severity level
representation_files = {
    'Normal': [f'./data_folder/block_{i}/Normal.csv' for i in range(2, 26)],
    'Medium': [f'./data_folder/block_{i}/Medium.csv' for i in range(2, 26)],
    'Low': [f'./data_folder/block_{i}/Low.csv' for i in range(2, 26)],
    'Very low': [f'./data_folder/block_{i}/Very low.csv' for i in range(2, 26)]
}

# Load data
def load_data(representation_file, acoustic_file, sample_size=None):
    # Load acoustic features
    acoustic_df = pd.read_csv(acoustic_file).drop(["maximal_intensity"], axis=1)

    # Load representation features from the specified block
    rep_df = pd.read_csv(representation_file)

    # Ensure 'speaker_id' and 'file_id' are treated as floats
    rep_df['file_id'] = rep_df['file_id'].astype(float)
    acoustic_df['file_id'] = acoustic_df['file_id'].astype(float)

    # Merge the two DataFrames on 'speaker_id' and 'file_id'
    merged_df = pd.merge(rep_df, acoustic_df, on=['speaker_id', 'file_id'])

    # If sample_size is provided, sample the data
    if sample_size is not None:
        merged_df = merged_df.sample(n=sample_size, random_state=42)

    return merged_df

# Preprocess data
def preprocess_data(data):
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
def train_model(model, train_loader, criterion, optimizer, num_epochs=300, patience=30):
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
    severities = ["Low", "Medium", "Very low", "Normal"]  # Change this to 'Normal', 'Medium', 'Low', or 'Very low' as needed
    for severity_level in severities:
        # Adjust the label_file based on the severity level
        
        if severity_level == 'Normal':
            label_file = 'control'
        else:
            label_file = 'experimental'

        representation_files_list = representation_files[severity_level]
        acoustic_file = f'./data_folder/{label_file}_acoustic_features.csv'  # Use cleaned file
        experimental_acoustic_file = './data_folder/block_1/Low.csv'  # Experimental acoustic file for sample size

        # Determine the number of samples in the experimental group
        experimental_acoustic_df = pd.read_csv(experimental_acoustic_file)
        sample_size = len(experimental_acoustic_df)
        #sample_size = experimental_acoustic_file

        results_file = f'results_SampleSize_{severity_level}.csv'
        
        # List of acoustic features to predict
        acoustic_features = [
            'mean_F2_bandwidth', 'mean_harmonic_difference_H1_A3', '50th_percentile_pitch_semitone',
            'mean_harmonic_difference_H1_H2', 'mean_spectral_slope_500_1500_voiced', 'cv_mfcc3',
            'cv_spectral_flux', 'cv_HNR', 'cv_F2', 'cv_F3_bandwidth',
            'loudness', 'logHNR'
        ]

        # Ensure results file is empty initially
        if os.path.exists(results_file):
            os.remove(results_file)
        
        for block_path in representation_files_list:
            block = os.path.basename(os.path.dirname(block_path))  # Get the block name from the directory
            logging.info(f"Processing {block}...")
            for feature in acoustic_features:
                logging.info(f"Predicting {feature} from {block}...")
                # Load and preprocess data
                data = load_data(block_path, acoustic_file, sample_size=sample_size)
                data = preprocess_data(data)
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
                best_val_loss = train_model(model, train_loader, criterion, optimizer, num_epochs=200, patience=10)

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
