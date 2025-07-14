import sys
import os
import torch
from inference.base_tts_infer import BaseTTSInfer
from utils.ckpt_utils import load_ckpt, get_last_checkpoint
from utils.hparams import hparams
from modules.ProDiff.model.ProDiff import GaussianDiffusion
from usr.diff.net import DiffNet
import os
import numpy as np
from functools import partial
import random
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import pandas as pd
import librosa
# from speech_reco.sr import speech_reco
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
        print('sample:', sample)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        print('txt_tokens:', txt_tokens.size(), txt_tokens)

        with torch.no_grad():   
            output = self.model(txt_tokens, infer=True)
            mel_out = output['mel_out']
            print('mel_out:', mel_out.size(), mel_out)
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
        # wav_out = wav_out.squeeze().cpu().numpy()
        return bits_out
    
#     DiffusionInfer.example_run()
# 单独使用时的函数
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


def extra(audio, seed2, text=None):
    # if text == None:
    #     # sr_text = speech_reco(audio)
    #     message_extra = DiffusionInfer.extra_run(audio=audio, text=sr_text, seed2=seed2)
    #     return sr_text,message_extra
    # else:
        message_extra = DiffusionInfer.extra_run(audio=audio, text=text, seed2=seed2)
        return text,message_extra
    
if __name__ == '__main__':
    pre_message = '''3D Gaussian Splatting (3DGS) has already become the emerging research focus in the fields of 3D scene reconstruction and novel view synthesis. Given that training a 3DGS requires a significant amount of time and computational cost, it is crucial to protect the copyright, integrity, and privacy of such 3D assets. Steganography, as a crucial technique for encrypted transmission and copyright protection, has been extensively studied. However, it still lacks profound exploration targeted at 3DGS. Unlike its predecessor NeRF, 3DGS possesses two distinct features: 1) explicit 3D representation; and 2) real-time rendering speeds. These characteristics result in the 3DGS point cloud files being public and transparent, with each Gaussian point having a clear physical significance. Therefore, ensuring the security and fidelity of the original 3D scene while embedding information into the 3DGS point cloud files is an extremely challenging task. To solve the above-mentioned issue, we first propose a steganography framework for 3DGS, dubbed GS-Hider, which can embed 3D scenes and images into original GS point clouds in an invisible manner and accurately extract the hidden messages. Specifically, we design a coupled secured feature attribute to replace the original spherical harmonics coefficients and then use a scene decoder and a message decoder to disentangle the original RGB scene and the hidden message. Extensive experiments demonstrated that the proposed GS-Hider can effectively conceal multimodal messages without compromising rendering quality and possesses exceptional security, robustness, capacity, and flexibility.'''

    print(embed(text='The steganography method of COAs system proposed by us is not limited to discop.', message = pre_message, seed1=64))

# print(extra(audio='infer_out/This_stego.wav',text='This is a test audio', seed2=64))

# print(extra(audio='infer_out/When_stego.wav', seed2=64))

