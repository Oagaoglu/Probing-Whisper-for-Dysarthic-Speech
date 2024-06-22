import numpy as np
import pandas as pd
import opensmile

def extract_features(wav_file, output_file):
    # Initialize OpenSmile for feature extraction
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    # Process the WAV file and extract features
    y = smile.process_file(wav_file)

    # Define the features to keep from the provided list
    features_to_keep = [
        'F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
        'spectralFlux_sma3_amean',
        'spectralFlux_sma3_stddevNorm',
        'HNRdBACF_sma3nz_amean',
        'HNRdBACF_sma3nz_stddevNorm',
        'F2frequency_sma3nz_amean',
        'F2frequency_sma3nz_stddevNorm',
        'F2bandwidth_sma3nz_amean',
        'F2bandwidth_sma3nz_stddevNorm',
        'F3bandwidth_sma3nz_amean',
        'F3bandwidth_sma3nz_stddevNorm',
        'logRelF0-H1-A3_sma3nz_amean',
        'logRelF0-H1-H2_sma3nz_amean',
        'slopeV500-1500_sma3nz_amean',
        'loudness_sma3_amean',
        'HNRdBACF_sma3nz_amean',  # logHNR correction
        'pcm_RMSenergy_sma_maxPos'  # Assuming this is maximal intensity
    ]

    # Filter the features to keep only the desired ones
    filtered_features = {key: y[key].values[0] for key in features_to_keep if key in y}

    # Calculate additional features
    if 'spectralFlux_sma3_amean' in filtered_features and 'spectralFlux_sma3_stddevNorm' in filtered_features:
        filtered_features['cv_spectralFlux'] = filtered_features['spectralFlux_sma3_stddevNorm'] / filtered_features['spectralFlux_sma3_amean']

    if 'HNRdBACF_sma3nz_amean' in filtered_features and 'HNRdBACF_sma3nz_stddevNorm' in filtered_features:
        filtered_features['cv_HNR'] = filtered_features['HNRdBACF_sma3nz_stddevNorm'] / filtered_features['HNRdBACF_sma3nz_amean']

    if 'F2frequency_sma3nz_amean' in filtered_features and 'F2frequency_sma3nz_stddevNorm' in filtered_features:
        filtered_features['cv_F2'] = filtered_features['F2frequency_sma3nz_stddevNorm'] / filtered_features['F2frequency_sma3nz_amean']

    if 'F3bandwidth_sma3nz_amean' in filtered_features and 'F3bandwidth_sma3nz_stddevNorm' in filtered_features:
        filtered_features['cv_F3bandwidth'] = filtered_features['F3bandwidth_sma3nz_stddevNorm'] / filtered_features['F3bandwidth_sma3nz_amean']

    if 'mfcc3_sma3_amean' in y and 'mfcc3_sma3_stddevNorm' in y:
        filtered_features['cv_mfcc3'] = y['mfcc3_sma3_stddevNorm'].values[0] / y['mfcc3_sma3_amean'].values[0]

    # Prepare the data for CSV
    final_features = {
        'mean_F2_bandwidth': filtered_features.get('F2bandwidth_sma3nz_amean', np.nan),
        'mean_harmonic_difference_H1_A3': filtered_features.get('logRelF0-H1-A3_sma3nz_amean', np.nan),
        '50th_percentile_pitch_semitone': filtered_features.get('F0semitoneFrom27.5Hz_sma3nz_percentile50.0', np.nan),
        'mean_harmonic_difference_H1_H2': filtered_features.get('logRelF0-H1-H2_sma3nz_amean', np.nan),
        'mean_spectral_slope_500_1500_voiced': filtered_features.get('slopeV500-1500_sma3nz_amean', np.nan),
        'cv_mfcc3': filtered_features.get('cv_mfcc3', np.nan),
        'cv_spectral_flux': filtered_features.get('cv_spectralFlux', np.nan),
        'maximal_intensity': filtered_features.get('pcm_RMSenergy_sma_maxPos', np.nan),
        'cv_HNR': filtered_features.get('cv_HNR', np.nan),
        'cv_F2': filtered_features.get('cv_F2', np.nan),
        'cv_F3_bandwidth': filtered_features.get('cv_F3bandwidth', np.nan),
        'loudness': filtered_features.get('loudness_sma3_amean', np.nan),
        'logHNR': filtered_features.get('HNRdBACF_sma3nz_amean', np.nan)  # logHNR correction
    }

    # Create a DataFrame and save to CSV
    df = pd.DataFrame([final_features])
    df.to_csv(output_file, index=False)

    #print(f"Feature names and values saved to: {output_file}")
    return final_features

if __name__ == "__main__":
    # Test the feature extraction
    wav_file = "./0001.wav"  # Replace with your WAV file path
    output_file = 'extracted_features.csv'
    extracted_features = extract_features(wav_file, output_file)
    print("Extracted features:")
    for feature, value in extracted_features.items():
        print(f"{feature}: {value}")
