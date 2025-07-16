import nemo.collections.asr as nemo_asr
import librosa
import soundfile as sf
import numpy as np
import os 
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import argparse

def parakeet(input_audio):

	asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

	audio, orig_sr = librosa.load(input_audio, sr=None)
	audio_16k = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)

	output_path = input_audio.replace('.wav', '_16k.wav')

	sf.write(output_path, audio_16k, 16000, subtype='PCM_16')
			
	output = asr_model.transcribe(output_path)

	os.remove(output_path)  # Clean up the temporary file

	return output[0].text

def whisper(input_audio):

	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

	model_id = "openai/whisper-large-v3"

	model = AutoModelForSpeechSeq2Seq.from_pretrained(
		model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
	)
	model.to(device)

	processor = AutoProcessor.from_pretrained(model_id)

	pipe = pipeline(
		"automatic-speech-recognition",
		model=model,
		tokenizer=processor.tokenizer,
		feature_extractor=processor.feature_extractor,
		torch_dtype=torch_dtype,
		device=device,
	)

	dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
	sample = dataset[0]["audio"]

	result = pipe(input_audio)
	print(result["text"])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file using either Parakeet or Whisper."
    )
    
    subparsers = parser.add_subparsers(dest='model_choice', required=True, help='Choose the ASR model to use')

    parser_parakeet = subparsers.add_parser('parakeet', help='Use the Parakeet model for transcription.')
    parser_parakeet.add_argument('-a', '--input_audio', type=str, help='Path to the input audio file.')

    parser_whisper = subparsers.add_parser('whisper', help='Use the Whisper model for transcription.')
    parser_whisper.add_argument('-a', '--input_audio', type=str, help='Path to the input audio file.')

    args = parser.parse_args()

    audio_text = ""
    if args.model_choice == 'parakeet':
        audio_text = parakeet(args.input_audio)
    elif args.model_choice == 'whisper':
        audio_text = whisper(args.input_audio)

    print("\n--- Speech Recognition Result ---")
    print(audio_text)
