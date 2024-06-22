import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import logging

# Set up logging
logging.basicConfig(filename='visualization.log', level=logging.INFO, format='%(asctime)s %(message)s')

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

def load_data_from_block(block_number):
    combined_df = pd.DataFrame()
    for severity_label in severity_dict.values():
        file_path = f'./data_folder/block_{block_number}/{severity_label}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['severity_level'] = severity_label
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            logging.warning(f'File {file_path} does not exist.')
    return combined_df

def visualize_representations(block_number):
    logging.info(f'Loading data for block {block_number}...')
    data_df = load_data_from_block(block_number)

    if data_df.empty:
        logging.warning(f'No data found for block {block_number}. Skipping visualization.')
        return

    # Prepare features and labels
    X = data_df.drop(columns=['severity_level', 'speaker_id', 'file_id'])
    y = data_df['severity_level']

    # Apply PCA to reduce dimensions to 2D
    logging.info(f'Applying PCA to reduce dimensions for block {block_number}...')
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['severity_level'] = y.values

    # Calculate the mean points for each severity level
    mean_points = pca_df.groupby('severity_level')[['PC1', 'PC2']].mean().reset_index()

    # Plot the PCA results
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='severity_level', palette=['green', 'blue', 'orange', 'red'], s=100, alpha=0.2)
    plt.title(f'PCA of Representations for Encoding Layer {block_number - 1}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    # Colors and markers for mean points
    colors = {'Normal': 'green', 'Very low': 'blue', 'Low': 'orange', 'Medium': 'red'}
    markers = {'Normal': 'o', 'Very low': 's', 'Low': 'D', 'Medium': '^'}

    # Plot mean points
    for severity in ['Normal', 'Very low', 'Low', 'Medium']:
        mean_point = mean_points[mean_points['severity_level'] == severity]
        plt.scatter(mean_point['PC1'], mean_point['PC2'], color=colors[severity], marker=markers[severity], s=200, edgecolor='black', label=f'Mean {severity}')

    # Adjust legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Ensure the plots directory exists
    plots_dir = './plots/feature_plots'
    os.makedirs(plots_dir, exist_ok=True)

    plt.savefig(f'./plots/feature_plots/pca_visualization_encoding_layer_{block_number - 1}.png')

def main():
    logging.info('Starting the visualization script for all blocks...')
    for block_number in range(24, 25):
        visualize_representations(block_number)
    logging.info('Visualization script finished.')

if __name__ == "__main__":
    main()
