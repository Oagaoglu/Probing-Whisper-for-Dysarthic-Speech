import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths for the result CSVs
result_files = {
    'Normal': './results_Normal.csv',
    'Medium': './results_Medium.csv',
    'Low': './results_Low.csv',
    'Very low': './results_Very low.csv',
    'Random Normal': './results_random_Normal.csv',
    'Random Medium': './results_random_Medium.csv',
    'Random Low': './results_random_Low.csv',
    'Random Very low': './results_random_Very low.csv'
}

def calculate_min_loss_per_feature(result_file, has_blocks=True):
    """
    Calculate the minimum loss per feature for a given result file.
    
    Parameters:
    - result_file (str): Path to the result CSV file.
    - has_blocks (bool): Whether the file contains block information.
    
    Returns:
    - pd.DataFrame: DataFrame containing the minimum loss per feature.
    """
    df = pd.read_csv(result_file)
    
    if has_blocks:
        min_loss_per_feature = df.groupby('feature')['test_loss'].min()
    else:
        min_loss_per_feature = df.groupby('feature')['test_loss'].min()
    
    return min_loss_per_feature

def plot_side_by_side_min_losses(min_losses):
    """
    Plot the minimum losses for each severity level and the random baseline side-by-side.
    
    Parameters:
    - min_losses (dict): Dictionary containing the minimum losses per feature for each severity level and the random baseline.
    """
    for severity in ['Normal', 'Medium', 'Low', 'Very low']:
        random_severity = f'Random {severity}'
        features = min_losses[severity].index

        plt.figure(figsize=(14, 10))
        
        bar_width = 0.4
        r1 = range(len(features))
        r2 = [x + bar_width for x in r1]

        plt.bar(r1, min_losses[severity], width=bar_width, label=severity, color='blue', edgecolor='black')
        plt.bar(r2, min_losses[random_severity], width=bar_width, label=random_severity, color='orange', edgecolor='black')

        plt.xlabel('Feature')
        plt.ylabel('Minimum Test Loss')
        plt.title(f'Minimum Test Loss for {severity} Compared to Random Baseline')
        plt.xticks([r + bar_width / 2 for r in range(len(features))], features, rotation=90)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Ensure the plots directory exists
        plots_dir = './plots'
        os.makedirs(plots_dir, exist_ok=True)

        plt.savefig(os.path.join(plots_dir, f'minimum_test_loss_comparison_{severity.lower()}.png'))
        plt.show()

def main():
    min_losses = {}
    
    for severity, result_file in result_files.items():
        has_blocks = 'Random' not in severity
        min_losses[severity] = calculate_min_loss_per_feature(result_file, has_blocks)
    
    plot_side_by_side_min_losses(min_losses)

if __name__ == "__main__":
    main()
