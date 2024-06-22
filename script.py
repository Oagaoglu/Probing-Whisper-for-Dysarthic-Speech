import os
import pandas as pd

# Define the path to your speech files
directory_path = './TORGO/tudelft.net/staff-umbrella/SpeechLabData/TORGO'

# Define the mapping of severity levels to speaker IDs
severity_levels = {
    "Normal": ["FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"],
    "Very Low": ["F04", "M03"],
    "Low": ["F03", "M05"],
    "Medium": ["F01", "M01", "M02", "M04"]
}

# Initialize a dictionary to hold the counts
severity_counts = {key: 0 for key in severity_levels.keys()}

# Iterate over all files in the directory and its subdirectories
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith(".wav"):  # assuming the files are in .wav format
            # Get the speaker ID from the directory path
            speaker_id = root.split(os.sep)[-3]
            # Check which severity level the file belongs to
            for severity, speakers in severity_levels.items():
                if speaker_id in speakers:
                    severity_counts[severity] += 1
                    break

# Convert the counts to a DataFrame for better visualization (optional)
df_counts = pd.DataFrame(list(severity_counts.items()), columns=['Severity Level', 'Number of Speech Files'])

print(df_counts)
