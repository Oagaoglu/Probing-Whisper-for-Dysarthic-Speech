import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
for severity in ["Normal", "Low", "Very low", "Medium"]:
    results_df = pd.read_csv(f'./results_{severity}.csv')

    # Ensure block is treated as a numeric value and decrement by 1 for correct plotting
    results_df['layer'] = results_df['block'].str.extract('(\d+)').astype(int) - 1

    # Human-readable feature labels
    feature_labels = {
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

    # Map human-readable labels to the features
    results_df['feature'] = results_df['feature'].map(feature_labels)

    # Extract unique features and layers
    features = results_df['feature'].unique()
    layers = sorted(results_df['layer'].unique())

    # Normalize losses to percentage of the relative performance
    for feature in features:
        feature_data = results_df[results_df['feature'] == feature]
        min_loss = feature_data['test_loss'].min()
        results_df.loc[results_df['feature'] == feature, 'test_loss'] = (
            (results_df[results_df['feature'] == feature]['test_loss'] - min_loss) / min_loss) * 100

    # Calculate the average loss per layer
    avg_loss_per_layer = results_df.groupby('layer')['test_loss'].mean().reset_index()

    # Set up the plot
    plt.figure(figsize=(14, 10))

    # Create a seaborn lineplot with confidence intervals
    sns.lineplot(x='layer', y='test_loss', hue='feature', data=results_df, markers=True, ci='sd', palette='tab10', linewidth=2.5)

    # Add the average loss line
    plt.plot(avg_loss_per_layer['layer'], avg_loss_per_layer['test_loss'], marker='o', linestyle='--', color='black', label='Average Loss', linewidth=3)

    plt.xlabel('Encoding Layer')
    plt.ylabel('Relative Performance (%)')
    plt.title(f'Test Loss of Different Features Across Layers ({severity})')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Save the plot
    plt.savefig(f'./plots/feature_plots/{severity}.png', dpi=300)
    plt.show()
