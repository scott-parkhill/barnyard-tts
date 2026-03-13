#!/usr/bin/env python3

import argparse
import scipy.io.wavfile
import os
import sys
import torch
import yaml
from omegaconf import OmegaConf
import io

import synthesis.inference as inference
from synthesis import utils
from matcha.models.matcha_tts import MatchaTTS
from vocos import Vocos


def batch_synthesize_from_file(args, model, vocoder, device):
    """
    Synthesize audio for each word in a text file.

    Args:
        args: The command-line arguments
        model: Loaded TTS model
        vocoder: Loaded vocoder model
        device: PyTorch device (CPU or CUDA)

    Returns:
        0 on success, 1 on failure
    """
    try:
        # Check if input file exists
        if not os.path.exists(args.input_file):
            print(f"Error: Input file {args.input_file} not found")
            return 1

        # Create output directory if it doesn't exist
        output_dir = args.output
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Read words from the file
        print(f"Reading words from {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as file:
            words = [line.strip() for line in file if line.strip()]

        print(f"Found {len(words)} words to synthesize")

        # Process each word
        for i, word in enumerate(words):
            print(f"Processing word {i + 1}/{len(words)}: '{word}'")

            # Create a safe filename
            safe_filename = ''.join(c if c.isalnum() else '_' for c in word)
            output_path = os.path.join(output_dir, f"{safe_filename}.wav")

            # Process text
            processed_text = inference.process_text(word, device)

            # Set up speaker and language tensors if needed
            speaker_tensor = None
            language_tensor = None

            # Check if model is multi-speaker
            if hasattr(model, 'spk_emb'):
                speaker_tensor = torch.tensor([args.speaker_id], device=device)

            # Check if model is multilingual
            if hasattr(model, 'lang_emb'):
                language_tensor = torch.tensor([args.language_id], device=device)

            # Synthesize speech
            output = inference.synthesise(
                processed_text,
                model,
                temperature=args.temperature,
                length_scale=args.length_scale,
                spks=speaker_tensor,
                lang=language_tensor
            )

            # Convert to waveform
            waveform = inference.to_waveform(output['mel'], None, vocoder)

            # Save audio
            audio_data = waveform.squeeze().numpy()
            # Normalize if needed
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))
            # Convert to int16
            audio_data = (audio_data * 32767).astype('int16')
            scipy.io.wavfile.write(output_path, 22050, audio_data)
            print(f"Saved audio to {output_path}")

        print(f"Successfully synthesized {len(words)} words")
        return 0

    except Exception as e:
        print(f"Error during batch synthesis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def check_cuda_availability():
    """Check if CUDA is available and print diagnostic information."""
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    device_count = torch.cuda.device_count() if cuda_available else 0
    current_device = torch.cuda.current_device() if cuda_available and device_count > 0 else "N/A"
    device_name = torch.cuda.get_device_name(current_device) if cuda_available and device_count > 0 else "N/A"

    print("\n=== CUDA Diagnostic Information ===")
    print(f"CUDA is available: {cuda_available}")
    print(f"CUDA version: {cuda_version}")
    print(f"Number of CUDA devices: {device_count}")
    print(f"Current CUDA device: {current_device}")
    print(f"CUDA device name: {device_name}")
    print("==============================\n")

    return cuda_available







tts_model_ro="../models/multilingual_fnet_last.ckpt"
tts_model_syllabics="../models/moosecree_syllabics.ckpt"

vocoder="../models/vocos_last.ckpt"
vocos_config="../models/vocos-matcha.yaml"

device = torch.device("cpu")

model_ro = MatchaTTS.load_from_checkpoint(tts_model_ro, map_location=device)
model_ro.eval()

model_syllabics = MatchaTTS.load_from_checkpoint(tts_model_syllabics, map_location=device)
model_syllabics.eval()

vocoder = utils.load_vocoder(
                vocos_config,
                vocoder,
                device,
                data_type=None  # Remove the data_type parameter or use torch.float32
            )
vocoder.eval()

# Temporary copy paste from main() just to get the API running.
def synthesize(text: str,
               model_id: int,
               language_id: int,
               speaker_id: int) -> io.BytesIO:

    if model_id == 1:
        model = model_ro
    if model_id == 2:
        model = model_syllabics

    # Process single text
    print(f"Processing text: '{text}'")
    processed_text = inference.process_text(text, device)

    # Set up speaker and language tensors if needed
    speaker_tensor = None
    language_tensor = None

    # Check if model is multi-speaker
    if hasattr(model, 'spk_emb'):
        speaker_tensor = torch.tensor([speaker_id], device=device)
        print(f"Using speaker ID: {speaker_id}")

    # Check if model is multilingual
    if hasattr(model, 'lang_emb'):
        language_tensor = torch.tensor([language_id], device=device)
        print(f"Using language ID: {language_id}")

    # Synthesize speech
    print("Synthesizing speech...")
    temperature = 0.667
    length_scale = 1.0
    output = inference.synthesise(
        processed_text,
        model,
        temperature=temperature,
        length_scale=length_scale,
        spks=speaker_tensor,
        lang=language_tensor
    )

    # Convert to waveform
    print("Converting to waveform...")
    waveform = inference.to_waveform(output['mel'], None, vocoder)

    audio_data = waveform.squeeze().numpy()
    # Normalize if needed
    if audio_data.max() > 1.0 or audio_data.min() < -1.0:
        audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))
    # Convert to int16
    audio_data = (audio_data * 32767).astype('int16')

    # Write to buffer.
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, 22050, audio_data)
    buffer.seek(0)

    # Print synthesis stats if you have these utility functions
    # If you don't have these functions, comment out or remove these lines
    try:
        rtf = utils.compute_rtf(output)
        inference_time = utils.compute_time_spent(output)
        print(f"Synthesis time: {inference_time:.2f} seconds")
        print(f"Real-time factor: {rtf:.2f}x")
    except (AttributeError, KeyError):
        # Skip if these utility functions aren't available
        pass

    print("Successfully returning waveform as bytes.")
    return buffer









def main():
    
    parser = argparse.ArgumentParser(description="Text-to-Speech Synthesis with MatchaTTS")

    # Changed from required=True to required=False for text
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--tts-model", type=str, required=True, help="Path to the TTS model checkpoint")
    parser.add_argument("--vocoder", type=str, required=True, help="Path to the vocoder checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output WAV file path or directory for batch mode")

    # Optional arguments
    parser.add_argument("--vocos-config", type=str, help="Path to Vocos configuration file")
    parser.add_argument("--temperature", type=float, default=0.667, help="Temperature for generation randomness")
    parser.add_argument("--length-scale", type=float, default=1.0, help="Length scale for audio duration")
    parser.add_argument("--speaker-id", type=int, default=2, help="Speaker ID for multi-speaker models")
    parser.add_argument("--language-id", type=int, default=2, help="Language ID for multilingual models")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if CUDA is available")
    parser.add_argument("--input-file", type=str, help="Text file containing words to synthesize (one per line)")
    parser.add_argument("--batch", action="store_true", help="Process a batch of words from input file")

    args = parser.parse_args()

    # Add validation to ensure either text or batch mode is specified
    if not args.text and not (args.batch and args.input_file):
        parser.error("Either --text or both --batch and --input-file must be provided")

    # Check CUDA availability
    cuda_available = check_cuda_availability()
    device = torch.device("cpu") if args.cpu or not cuda_available else torch.device("cuda")

    try:
        # Load TTS model
        print(f"Loading TTS model: {args.tts_model}")
        model = MatchaTTS.load_from_checkpoint(args.tts_model, map_location=device)
        model.eval()
        print("Model loaded successfully!")

        # Load vocoder
        print(f"Loading vocoder: {args.vocoder}")
        if args.vocos_config:
            from synthesis import utils

            # Use the project's existing utility function to load the vocoder
            vocoder = utils.load_vocoder(
                args.vocos_config,
                args.vocoder,
                device,
                data_type=None  # Remove the data_type parameter or use torch.float32
            )
        else:
            vocoder = torch.load(args.vocoder, map_location=device)
        vocoder.eval()
        print("Vocoder loaded successfully!")

        # Check if we're in batch mode
        if args.batch and args.input_file:
            return batch_synthesize_from_file(args, model, vocoder, device)
        else:
            # Process single text
            print(f"Processing text: '{args.text}'")
            processed_text = inference.process_text(args.text, device)

            # Set up speaker and language tensors if needed
            speaker_tensor = None
            language_tensor = None

            # Check if model is multi-speaker
            if hasattr(model, 'spk_emb'):
                speaker_tensor = torch.tensor([args.speaker_id], device=device)
                print(f"Using speaker ID: {args.speaker_id}")

            # Check if model is multilingual
            if hasattr(model, 'lang_emb'):
                language_tensor = torch.tensor([args.language_id], device=device)
                print(f"Using language ID: {args.language_id}")

            # Synthesize speech
            print("Synthesizing speech...")
            output = inference.synthesise(
                processed_text,
                model,
                temperature=args.temperature,
                length_scale=args.length_scale,
                spks=speaker_tensor,
                lang=language_tensor
            )

            # Convert to waveform
            print("Converting to waveform...")
            waveform = inference.to_waveform(output['mel'], None, vocoder)

            audio_data = waveform.squeeze().numpy()
            # Normalize if needed
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))
            # Convert to int16
            audio_data = (audio_data * 32767).astype('int16')
            scipy.io.wavfile.write(args.output, 22050, audio_data)
            print(f"Audio saved to {args.output}")

            # Print synthesis stats if you have these utility functions
            # If you don't have these functions, comment out or remove these lines
            try:
                rtf = utils.compute_rtf(output)
                inference_time = utils.compute_time_spent(output)
                print(f"Synthesis time: {inference_time:.2f} seconds")
                print(f"Real-time factor: {rtf:.2f}x")
            except (AttributeError, KeyError):
                # Skip if these utility functions aren't available
                pass

    except Exception as e:
        print(f"Error during synthesis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
