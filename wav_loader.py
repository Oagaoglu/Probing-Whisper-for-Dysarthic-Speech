import os
import pandas as pd

def list_wav_files(base_directory):
    """
    Walks through the directory structure starting from the base_directory,
    specifically looks for directories named like 'Session#' and within them,
    identifies WAV files within 'wav_arrayMic' subdirectories, and categorizes them
    into 'control' and 'experimental' based on folder naming conventions.

    Parameters:
    - base_directory (str): The root directory to start the search from.

    Returns:
    - dict: A dictionary with keys 'control' and 'experimental', each containing a list of unprocessed WAV file paths.
    """
    # Hardcoded CSV file paths to check for already processed files
    csv_files = ['./data_folder/block_1/Medium.csv','./data_folder/block_1/Very low.csv', './data_folder/block_1/Low.csv']

    # Check processed files
    processed_files = check_processed_files(csv_files)

    audio_files = {'control': [], 'experimental': []}

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_directory):
        # Look for session directories specifically named like 'Session1', 'Session2', etc.
        if root.split(os.sep)[-1].startswith('Session'):
            # Check for the presence of 'wav_arrayMic' directory within the session folder
            if 'wav_arrayMic' in dirs:
                wav_dir_path = os.path.join(root, 'wav_arrayMic')
                # Determine the group type based on the presence of 'C' in the directory name
                is_control = 'C' in os.path.basename(os.path.dirname(root))
                group_type = 'control' if is_control else 'experimental'
                
                # List all WAV files in the 'wav_arrayMic' directory
                for file in os.listdir(wav_dir_path):
                    if file.endswith('.wav'):
                        speaker_id = os.path.basename(os.path.dirname(root))
                        file_id = os.path.splitext(file)[0]
                        
                        if (speaker_id, file_id) not in processed_files:
                            audio_files[group_type].append(os.path.join(wav_dir_path, file))
                        else: 
                            print(speaker_id,file_id)
                            print("-")

    return audio_files

def check_processed_files(csv_files):
    """
    Reads the CSV files to get the list of already processed files.

    Parameters:
    - csv_files (list): List of CSV file paths.

    Returns:
    - set: A set of tuples containing (speaker_id, file_id) of processed files.
    """
    processed_files = set()
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(csv_file)
            df = pd.read_csv(csv_file, dtype={'speaker_id': str, 'file_id': str})
            if 'speaker_id' in df.columns and 'file_id' in df.columns:
                processed_files.update(zip(df['speaker_id'], df['file_id']))
    return processed_files

def main():
    base_directory = './TORGO/tudelft.net/staff-umbrella/SpeechLabData/TORGO/'  # Update this path as necessary
    audio_files = list_wav_files(base_directory)
    print(audio_files)  # Optional: Print the output for verification during development

if __name__ == "__main__":
    main()
