import os
import pandas as pd
import numpy as np
from encoding_extractor import load_model_and_hooks, process_audio
from wav_loader import list_wav_files
from f_extractor import extract_features

# Mapping of speaker IDs to severity levels
severity_mapping = {
    'FC01': 'Normal', 'FC02': 'Normal', 'FC03': 'Normal',
    'MC01': 'Normal', 'MC02': 'Normal', 'MC03': 'Normal', 'MC04': 'Normal',
    'F04': 'Very low', 'M03': 'Very low',
    'F03': 'Low', 'M05': 'Low',
    'F01': 'Medium', 'M01': 'Medium', 'M02': 'Medium', 'M04': 'Medium'
}

def save_to_csv(data, csv_filename):
    """Appends block data to a CSV file."""
    df = pd.DataFrame([data])  # Ensure data is encapsulated as a single row
    df.to_csv(csv_filename, mode='a', header=False, index=False)

def process_and_save_files(group_files, group_name, model, output_dir):
    total_files = len(group_files)
    print(total_files)
    for index, file_path in enumerate(group_files):
        if (index + 1) % 10 == 0 or index + 1 == total_files:
            remaining_files = total_files - index - 1
            print(f"Processed {index + 1}/{total_files} files. {remaining_files} files remaining.")
        file_id = os.path.basename(file_path).replace('.wav', '')
        # Correct speaker ID extraction: assuming the speaker ID directory is two levels up
        speaker_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
        severity = severity_mapping.get(speaker_id, 'Unknown')

        hidden_states, num_frames = process_audio(model, file_path)

        if hidden_states is None or num_frames is None:
            print(f"Skipping file {file_path} due to processing error.")
            continue

        for i, block_output in enumerate(hidden_states):
            # Calculate the time-average of the valid frames
            valid_data = block_output[:, :num_frames, :].mean(dim=1).cpu().numpy()  # Average over time

            block_dir = os.path.join(output_dir, f"block_{i + 1}")
            os.makedirs(block_dir, exist_ok=True)

            csv_filename = os.path.join(block_dir, f"{severity}.csv")
            if not os.path.exists(csv_filename):
                # Initialize CSV with headers if it doesn't exist
                headers = ['speaker_id', 'file_id'] + [f'feature_{j}' for j in range(valid_data.shape[1])]
                pd.DataFrame([], columns=headers).to_csv(csv_filename, mode='w', index=False)

            # Prepare data for saving
            valid_data = valid_data.flatten()
            data_to_save = [speaker_id, file_id] + valid_data.tolist()
            save_to_csv(data_to_save, csv_filename)

def extract_and_save_features(wav_files, group_name, output_dir):
    """Extracts features from WAV files and saves them to CSV."""
    for file_path in wav_files:
        file_id = os.path.basename(file_path).replace('.wav', '')
        # Correct speaker ID extraction: assuming the speaker ID directory is two levels up
        speaker_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
        severity = severity_mapping.get(speaker_id, 'Unknown')

        # Extract features from WAV file
        extracted_features = extract_features(file_path, None)

        # Prepare data for saving
        data_to_save = [speaker_id, file_id] + list(extracted_features.values())
        block_dir = os.path.join(output_dir, severity)
        os.makedirs(block_dir, exist_ok=True)
        csv_filename = os.path.join(block_dir, f"{group_name}_acoustic_features.csv")
        save_to_csv(data_to_save, csv_filename)

def main():
    base_directory = './TORGO/tudelft.net/staff-umbrella/SpeechLabData/TORGO'
    output_directory = './data_folder'
    model = load_model_and_hooks('medium')
    wav_files = list_wav_files(base_directory)
    print(len(wav_files))
    # Process control group files
    print("Processing control group files...")
    process_and_save_files(wav_files['control'], 'control', model, output_directory)

    # Process experimental group files
    #print("Processing experimental group files...")
    #process_and_save_files(wav_files['experimental'], 'experimental', model, output_directory)

    # Uncomment to extract and save features
    # extract_and_save_features(wav_files['control'], 'control', output_directory)
    # extract_and_save_features(wav_files['experimental'], 'experimental', output_directory)

if __name__ == "__main__":
    main()
