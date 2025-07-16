import sys
import os
import torch
from inference.base_tts_infer import BaseTTSInfer
from utils.ckpt_utils import load_ckpt, get_last_checkpoint
from utils.hparams import hparams
from modules.ProDiff.model.ProDiff import GaussianDiffusion
from usr.diff.net import DiffNet
import os
import argparse
import numpy as np
from functools import partial
import random
import pandas as pd
from utils.hparams import set_hparams
from utils.hparams import hparams as hp
from utils.audio import save_wav
from utils.zip import compress_string_to_bits, decompress_bits_to_string


class DiffusionInfer(BaseTTSInfer):
    def build_model(self):
        f0_stats_fn = f'{hparams["binary_data_dir"]}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            hparams['f0_mean'], hparams['f0_std'] = np.load(f0_stats_fn)
            hparams['f0_mean'] = float(hparams['f0_mean'])
            hparams['f0_std'] = float(hparams['f0_std'])
        model = GaussianDiffusion(
            phone_encoder=self.ph_encoder,
            out_dims=80, denoise_fn=DiffNet(hparams['audio_num_mel_bins']),
            timesteps=hparams['timesteps'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        checkpoint = torch.load(hparams['teacher_ckpt'], map_location='cpu')["state_dict"]['model']
        teacher_timesteps = int(checkpoint['timesteps'].item())
        teacher_timescales = int(checkpoint['timescale'].item())
        student_timesteps = teacher_timesteps // 2
        student_timescales = teacher_timescales * 2
        to_torch = partial(torch.tensor, dtype=torch.float32)
        model.register_buffer('timesteps', to_torch(student_timesteps))      # beta
        model.register_buffer('timescale', to_torch(student_timescales))      # beta
        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')
        print(model)
        return model

    def forward_model(self, inp, message, seed1, compress_messbits):
        random.seed(seed1)
        np.random.seed(seed1)
        torch.manual_seed(seed1)
        torch.cuda.manual_seed(seed1)
        torch.cuda.manual_seed_all(seed1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]

        with torch.no_grad():   
            output = self.model(txt_tokens, infer=True)
            mel_out = output['mel_out']
            wav_out = self.run_vocoder(mel_out,message=message,seed1=seed1,compress_messbits=compress_messbits)
        wav_out = wav_out.squeeze().cpu().numpy()
        return wav_out
    
    def forward_model_extra(self, inp, audio,seed2):
        random.seed(seed2)
        np.random.seed(seed2)
        torch.manual_seed(seed2)
        torch.cuda.manual_seed(seed2)
        torch.cuda.manual_seed_all(seed2)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        with torch.no_grad():
            output = self.model(txt_tokens, infer=True)
            mel_out = output['mel_out']
            bits_out = self.run_vocoder_extra(mel_out,audio=audio,seed2=seed2)
        return bits_out
    
def embed(text,message,seed1):
    set_hparams()
    stego = DiffusionInfer(hp)

    inp = {'text': text}
    message_binary_string = compress_string_to_bits(message) 
    message_binary_list = list(map(int, message_binary_string))

    out = stego.infer_once(inp=inp, message=message, seed1=seed1, compress_messbits = message_binary_list)
    audio_stego = f'infer_out/{text.split()[0]}_stego.wav'

    save_wav(out, audio_stego, hp['audio_sample_rate'])

    return audio_stego


def extra(audio, seed2, text):

    set_hparams()
    stego = DiffusionInfer(hp)

    inp = {'text': text}

    message_bits_extra = stego.infer_once_extra(inp,audio=audio,seed2=seed2)
	
    binary_string = ''.join(str(bit) for bit in message_bits_extra)
    decompress_str = decompress_bits_to_string(binary_string)

    return decompress_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A command-line tool for audio steganography using diffusion models."
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    parser_embed = subparsers.add_parser('embed', help='Embed a secret message into audio.')
    parser_embed.add_argument('-t', '--text', type=str, help='The audio text to generate from.')
    parser_embed.add_argument('-m', '--message', type=str, help='The secret message to embed.')
    parser_embed.add_argument('-s', '--seed', type=int, help='The random seed for embedding (acts as a key).')

    parser_extra = subparsers.add_parser('extract', help='Extract a secret message from audio.')
    parser_extra.add_argument('-a', '--audio', type=str, help='Path to the steganographic audio file.')
    parser_extra.add_argument('-s', '--seed', type=int, help='The random seed for extraction (acts as a key).')
    parser_extra.add_argument('-t', '--text', type=str, default=None, help='The original audio text.')

    args = parser.parse_args()

    if args.command == 'embed':
        print("--- Running Embedding ---")
        output_path = embed(args.text, args.message, args.seed1)
        print(f"Embedding complete. Stego audio saved to: {output_path}")
    elif args.command == 'extract':
        print("--- Running Extraction ---")
        extracted_message = extra(args.audio, args.seed2, args.text)
        print(f"Extraction complete. The secret message is: '{extracted_message}'")


