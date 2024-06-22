import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
def load_data(file_path):
    return pd.read_csv(file_path)

# File paths for "Normal" and "Medium" severity levels
file_paths = {
    'Normal': './results_Normal.csv',
    'Medium': './results_Medium.csv'
}

# Load datasets
data_frames = {level: load_data(path) for level, path in file_paths.items()}

# Extract block number from the 'block' column and adjust for plotting
for level in data_frames.keys():
    data_frames[level]['block'] = data_frames[level]['block'].str.extract('(\d+)').astype(int) - 1

# Ensure the plots directory exists
plots_dir = './plots/feature_plots'
os.makedirs(plots_dir, exist_ok=True)

# Extract unique features
features = data_frames['Normal']['feature'].unique()

# Human-readable feature names
feature_names = {
    'mean_F2_bandwidth': 'Mean F2 Bandwidth',
    'mean_harmonic_difference_H1_A3': 'Mean Harmonic Difference H1-A3',
    '50th_percentile_pitch_semitone': '50th Percentile Pitch (Semitone)',
    'mean_harmonic_difference_H1_H2': 'Mean Harmonic Difference H1-H2',
    'mean_spectral_slope_500_1500_voiced': 'Mean Spectral Slope 500-1500 Voiced',
    'cv_mfcc3': 'CV MFCC3',
    'cv_spectral_flux': 'CV Spectral Flux',
    'cv_HNR': 'CV HNR',
    'cv_F2': 'CV F2',
    'cv_F3_bandwidth': 'CV F3 Bandwidth',
    'loudness': 'Loudness',
    'logHNR': 'Log HNR'
}

# Plot each feature for "Normal" and "Medium"
for feature in features:
    plt.figure(figsize=(14, 10))
    
    for level in file_paths.keys():
        df = data_frames[level]
        feature_data = df[df['feature'] == feature]
        plt.plot(feature_data['block'], feature_data['test_loss'], marker='o', label=level)
    
    plt.xlabel('Encoding Layer')
    plt.ylabel('Test Loss')
    plt.title(f'Test Loss of {feature_names.get(feature, feature)} for Normal and Medium Severity Levels')
    plt.legend(title="Severity")
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(plots_dir, f'{feature}_comparison.png'))
    plt.close()

print("Plots saved successfully in the plots/feature_plots folder.")
