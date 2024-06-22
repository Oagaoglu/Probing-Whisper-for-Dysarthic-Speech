import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(filename='svm_training.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Example dictionary for speaker IDs and corresponding severity labels
severity_dict = {
    'FC01': 'Normal',
    'FC02': 'Normal',
    'FC03': 'Normal',
    'MC01': 'Normal',
    'MC02': 'Normal',
    'MC03': 'Normal',
    'MC04': 'Normal',
    'F04': 'Very low',
    'M03': 'Very low',
    'F03': 'Low',
    'M05': 'Low',
    'F01': 'Medium',
    'M01': 'Medium',
    'M02': 'Medium',
    'M04': 'Medium'
}

def create_dataset_from_dict(severity_dict, block_number):
    combined_df = pd.DataFrame()
    for speaker_id, severity_label in severity_dict.items():
        file_path = f'./data_folder/block_{block_number}/{severity_label}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['severity_level'] = severity_label
            df = df[df['speaker_id'] == speaker_id]
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            logging.warning(f'File {file_path} does not exist.')
    return combined_df

def equalize_classes(df, target_column):
    min_count = df[target_column].value_counts().min()
    balanced_df = df.groupby(target_column).apply(lambda x: x.sample(min_count)).reset_index(drop=True)
    return balanced_df

def train_svm_on_block(block_number):
    logging.info(f'Creating dataset from severity dictionary for block {block_number}...')
    combined_df = create_dataset_from_dict(severity_dict, block_number)

    if combined_df.empty:
        logging.warning(f'No data found for block {block_number}. Skipping this block.')
        return

    # Equalize the classes
    logging.info(f'Equalizing class distribution for block {block_number}...')
    combined_df = equalize_classes(combined_df, 'severity_level')

    # Prepare features and labels
    X = combined_df.drop(columns=['severity_level', 'speaker_id', 'file_id'])
    y = combined_df['severity_level']

    # Split the data into training and testing sets
    logging.info(f'Splitting data into training and testing sets for block {block_number}...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Log the label distribution in the splits
    logging.info(f'Training set label distribution for block {block_number}:\n{y_train.value_counts()}')
    logging.info(f'Testing set label distribution for block {block_number}:\n{y_test.value_counts()}')

    # Standardize the data
    logging.info(f'Standardizing the data for block {block_number}...')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the SVM classifier with RBF kernel and C=5
    logging.info(f'Training the SVM classifier for block {block_number}...')
    svm_clf = SVC(kernel='rbf', C=5, random_state=42)
    svm_clf.fit(X_train_scaled, y_train)

    # Save the trained model and scaler
    model_filename = f'svm_classifier_block_{block_number}.joblib'
    scaler_filename = f'scaler_block_{block_number}.joblib'
    logging.info(f'Saving the trained model and scaler for block {block_number} as {model_filename} and {scaler_filename}...')
    joblib.dump(svm_clf, model_filename)
    joblib.dump(scaler, scaler_filename)

    # Evaluate the model
    logging.info(f'Evaluating the model for block {block_number}...')
    y_pred = svm_clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info(f'Accuracy for block {block_number}: {accuracy}')
    logging.info(f'Classification Report for block {block_number}:\n{report}')
    logging.info(f'Confusion Matrix for block {block_number}:\n{conf_matrix}')

    # Save the confusion matrix as an image
    logging.info(f'Saving the confusion matrix for block {block_number}...')
    plt.figure(figsize=(10, 7))
    severity_levels = sorted(list(set(severity_dict.values())))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=severity_levels, yticklabels=severity_levels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for Block {block_number}')
    
    # Ensure the plots directory exists
    plots_dir = './plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_block_{block_number}.png'))

    # Print results to console as well
    print(f'Accuracy for block {block_number}: {accuracy}')
    print(f'Classification Report for block {block_number}:\n{report}')
    print(f'Confusion Matrix for block {block_number}:\n{conf_matrix}')

def main():
    logging.info('Starting the SVM training script for all blocks...')
    for block_number in range(23, 24):
        train_svm_on_block(block_number)
    logging.info('SVM training script finished.')

if __name__ == "__main__":
    main()
