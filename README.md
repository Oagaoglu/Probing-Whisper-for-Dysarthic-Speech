# Probing Whisper for Dysarthric Speech

This repository contains the code and resources for the research project titled "How Does OpenAI's Whisper Interpret Dysarthric Speech? Probing Acoustic Features". The project aims to analyze how the Whisper model processes dysarthric speech by probing its internal acoustic feature representations.

## Repository Structure

- `baseline_comparison.py`: Script for comparing baseline models.
- `compare.py`: Script for comparing different probing models.
- `E_probe.py`: Script for probing encoder layers.
- `encoding_extractor.py`: Script for extracting encodings from the Whisper model.
- `f_extractor.py`: Script for extracting features from audio files.
- `f_utils.py`: Utility functions for feature extraction and processing.
- `main.py`: Main script for running experiments.
- `plot.py`: Script for generating plots.
- `probe.py`: Script for probing specific features.
- `random_probe.py`: Script for probing with random vectors as a baseline.
- `script.py`: Miscellaneous script for various tasks.
- `svm.py`: Script for training and evaluating SVM classifiers.
- `svm_plot.py`: Script for generating SVM-related plots.
- `visualization_whisper.py`: Script for visualizing Whisper model representations.
- `wav_loader.py`: Script for loading WAV files.

## Plots and Results

- `plots/`: Directory containing plots generated during the analysis.
- `results/`: Directory containing results from various experiments.

## Data

The dataset used in this project is the TORGO database, which consists of high-quality audio recordings from individuals with various degrees and types of dysarthria, as well as age- and gender-matched controls.

## Running the Code

To reproduce the results in the research paper run main.py and extract the internal representations and features to be probed from speech files. Then run probe.py and svm.py files to analyze the representations via probes and severity classification accuracy.

## Contact

For any questions or issues, please contact [Orhan Agaoglu](mailto:oagaoglu@student.tudelft.nl).
