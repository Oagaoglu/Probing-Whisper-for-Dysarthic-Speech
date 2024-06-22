import torch
import soundfile as sf
from scipy.signal import resample
import os
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperFeatureExtractor
import math

def load_model_and_hooks(model_size='medium'):
    # Load the Whisper model
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{model_size}")
    model.eval()
    model.encoder_only = True
    return model

def adjust_model_config(model, num_frames, model_size='medium'):
    # Adjust the model's configuration based on the number of frames
    config = WhisperConfig.from_pretrained(f"openai/whisper-{model_size}", max_source_positions=num_frames)
    state_dict = model.state_dict()
    
    # Ensure the positional embeddings are correctly sized
    current_positional_emb_size = state_dict["model.encoder.embed_positions.weight"].shape[0]
    if num_frames > current_positional_emb_size:
        padding_size = num_frames - current_positional_emb_size
        state_dict["model.encoder.embed_positions.weight"] = torch.cat(
            [state_dict["model.encoder.embed_positions.weight"],
             torch.zeros((padding_size, state_dict["model.encoder.embed_positions.weight"].shape[1]))],
            dim=0
        )
    else:
        state_dict["model.encoder.embed_positions.weight"] = state_dict["model.encoder.embed_positions.weight"][:num_frames, :]
    
    # Load these weights back into a Whisper model configured for this new sequence length
    new_model = WhisperForConditionalGeneration(config)
    new_model.load_state_dict(state_dict)
    new_model.eval()
    new_model.encoder_only = True
    return new_model

def process_audio(model, audio_path, model_size='medium'):
    target_sr = 16000
    max_retries = 2
    audio, sr = sf.read(audio_path)
    if sr != target_sr:
        # Resample audio to 16000 Hz
        num_samples = int(len(audio) * float(target_sr) / sr)
        audio = resample(audio, num_samples)
        sr = target_sr

    duration = len(audio) / sr
    num_frames = math.ceil(duration * 50)  # Initial number of frames

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt+1}: Processing file {audio_path} with num_frames={num_frames}")
            model = adjust_model_config(model, num_frames, model_size)
            
            decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

            # Initialize feature extraction
            feature_extractor = WhisperFeatureExtractor(chunk_length=duration)
            inputs = feature_extractor(audio.flatten(), return_tensors="pt", sampling_rate=sr)

            # Process the entire audio
            with torch.no_grad():
                outputs = model(inputs.input_features, decoder_input_ids=decoder_input_ids, output_hidden_states=True)
            
            hidden_states = outputs.encoder_hidden_states
            last_layer_hidden_states = hidden_states[-1]

            return hidden_states, num_frames
        
        except Exception as e:
            print(f"Error processing file {audio_path} with num_frames={num_frames}: {e}")
            # Adjust the num_frames based on the error message
            if "The size of tensor a" in str(e) and "must match the size of tensor b" in str(e):
                # Extract the expected length from the error message
                expected_length = int(str(e).split('tensor a (')[1].split(')')[0])
                num_frames = expected_length  # Adjust num_frames based on the expected length

    print(f"Failed to process file {audio_path} after {max_retries} attempts.")
    return None, None
