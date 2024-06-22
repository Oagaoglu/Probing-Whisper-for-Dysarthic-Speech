import os
import re
import matplotlib.pyplot as plt

def parse_log_file(log_file):
    accuracies = {}
    with open(log_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = re.search(r'Accuracy for block (\d+): ([0-9.]+)', line)
            if match:
                block_number = int(match.group(1)) - 1  # Decrement block number by 1
                accuracy = float(match.group(2))
                accuracies[block_number] = accuracy
    return accuracies

def plot_accuracies(accuracies):
    sorted_blocks = sorted(accuracies.keys())
    accuracy_values = [accuracies[block] for block in sorted_blocks]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_blocks, accuracy_values, marker='o', linestyle='-', color='b', label='SVM Accuracy')
    plt.xlabel('Encoding Layer')
    plt.ylabel('Accuracy')
    plt.title('SVM Classification Accuracy Across Layers')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Ensure the plots directory exists
    plots_dir = './plots'
    os.makedirs(plots_dir, exist_ok=True)

    plt.savefig(os.path.join(plots_dir, 'svm_accuracies_across_layers.png'))
    plt.show()

def main():
    log_file = 'svm_training.log'
    if not os.path.exists(log_file):
        print(f"Log file {log_file} does not exist.")
        return

    accuracies = parse_log_file(log_file)
    if not accuracies:
        print("No accuracies found in the log file.")
        return

    plot_accuracies(accuracies)

if __name__ == "__main__":
    main()
