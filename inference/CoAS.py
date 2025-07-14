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
    

    def embedding(self):
        if self.audio_text_embeded == '':
            return 400, '音频文本文件未上传，无法执行嵌入任务！'
        if self.seed1 == None:
            return 400, '您并未上传密钥文件，无法执行嵌入任务！'
        if self.message_embedding == '':
            return 400, '嵌入消息文件未上传，无法执行嵌入任务！'
        try:
            #生成部分代码
            inp = {'text': self.audio_text_embeded}
            out = self.infer_once(inp, message=self.message_embedding, seed1=self.seed1, compress_messbits = self.compress_messbits)

            audio_name = f'{self.audio_text_embeded.split()[0]}_stego.wav'
            audio_folder = self.default_save_uploads_dir

            self.stego_audio_file_save_path = os.path.join(audio_folder, audio_name)
            save_wav(out, os.path.join(audio_folder, audio_name), hp['audio_sample_rate'])
            save_wav(out, os.path.join('../capacity_test/audio_data/audio_test', audio_name), hp['audio_sample_rate'])#前端的文件暂存
        except Exception as e:
            print(e)
            return 400, '生成式音频秘密信息嵌入过程中存在问题，请联系管理员排查。可尝试更换参数等。'
        return 200, '生成式秘密信息嵌入过程已完成。'
    
    def extracting(self):
        if self.audio_text_extracting == '':
            return 400, '音频文本未能获取，无法执行提取任务！'
        if self.seed2 == None:
            return 400, '您并未上传密钥文件，无法执行提取任务！'
        if self.stego_audio_file_upload_path == '':
            return 400, '载密音频文件未上传，无法执行提取任务！'
        try:
            #提取部分代码
            inp = {'text': self.audio_text_extracting}
            bits_out = self.infer_once_extra(inp, audio=self.stego_audio_file_upload_path, seed2=self.seed2)

            # with open(self.extracted_message_file_path, "w") as file:
            #     file.write(message_out)

            self.bits_to_decompress = bits_out
            binary_string = ''.join(str(bit) for bit in self.bits_to_decompress)
            decompress_str = decompress_bits_to_string(binary_string)

            with open(self.extracted_message_file_path, "w") as file:
                file.write(decompress_str)

            with open(os.path.join('../capacity_test/audio_data/message_test', 'extracted_message.txt'), 'w') as f:
                f.write(decompress_str)

        except Exception as e:
            print(e)
            return 400, '生成式音频秘密信息提取过程中存在问题，请联系管理员排查。可尝试更换参数等。'
        return 200, '生成式秘密信息提取过程已完成。'
    

    def _is_txt(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        return ext == '.txt'
    
    def _is_wav(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        return ext == '.wav'
    
    def embedding_download(self):
        return self.stego_audio_file_save_path, self.stego_mimetype
    
    def extracting_download(self):
        return self.extracted_message_file_path, self.extracted_mimetype
    
    #嵌入准备
    def audio_text_embedding_checker(self, audio_text_file):
        audio_text_file_path = os.path.join(self.default_save_uploads_dir, audio_text_file.filename)
        audio_text_file.save(audio_text_file_path)
        if not self._is_txt(audio_text_file_path):
            return 400, '您上传的音频文本文件格式有误，目前只支持文本文件。'
        try:
            print("audio_text_file_path ", audio_text_file_path)
            with open(audio_text_file_path, 'r', encoding="utf-8") as f:
                audio_text = f.read().strip()
                print("audio_text_embedding:", audio_text)
        except Exception:
            return 400, '文件受损或不存在，请重新检查'
        self.audio_text_embeded = audio_text
        return 200, '已成功记录音频文本。'
    
    def seed_embedding_checker(self, seed_file):
        seed_file_path = os.path.join(self.default_save_uploads_dir, seed_file.filename)
        seed_file.save(seed_file_path)
        if not self._is_txt(seed_file_path):
            return 400, '您上传的嵌入密钥文件格式有误，目前只支持文本文件。'
        try:
            with open(seed_file_path, 'r') as file:
                seed1 = int(file.readline().strip())
                print('seed1:', seed1)
        except Exception:
            return 400, '嵌入密钥文件有错误，与模型不兼容！'
        self.seed1 = seed1
        return 200, '嵌入密钥文件已成功加载。'

    def message_embedding_checker(self, message_file):
        message_file_path = os.path.join(self.default_save_uploads_dir, message_file.filename)
        message_file.save(message_file_path)
        if self.audio_text_embeded == '':
            return 400, '您并未上传音频文本文件，无法选择消息文件!'
        if self.seed1 == None:
            return 400, '您并未上传密钥文件，无法选择消息文件!'
        if not self._is_txt(message_file_path):
            return 400, '您上传的消息文件格式有误，目前只支持文本文件。'
        try:
            print("message_file_path ", message_file_path)
            with open(message_file_path, 'r', encoding="utf-8") as f:
                message = f.read().strip()
                print("secret message: ", message)
        except Exception:
            return 400, '消息文件有错误，与模型不兼容！'
        self.message_embedding = message
        message_binary_string = compress_string_to_bits(self.message_embedding) 
        message_binary_list = list(map(int, message_binary_string))
        self.compress_messbits = message_binary_list
        return 200, '消息文件已成功加载。'
    

    #提取准备
    def audio_text_extracting_checker(self, audio_text_file):
        audio_text_file_path = os.path.join(self.default_save_internal_dir, audio_text_file.filename)
        audio_text_file.save(audio_text_file_path)
        if not self._is_txt(audio_text_file_path):
            return 400, '您上传的音频文本文件格式有误，目前只支持文本文件。'
        try:
            print("audio_text_extracting_file_path:", audio_text_file_path)
            with open(audio_text_file_path, 'r', encoding="utf-8") as f:
                audio_text = f.read().strip()
                print("audio_text_extracting:", audio_text)
        except Exception:
            return 400, '文件受损或不存在，请重新检查。'
        self.audio_text_extracting = audio_text
        return 200, '已成功记录音频文本。'

    def seed_extracting_checker(self, seed_file):
        seed_file_path = os.path.join(self.default_save_internal_dir, seed_file.filename)
        seed_file.save(seed_file_path)
        if not self._is_txt(seed_file_path):
            return 400, '您上传的提取密钥文件格式有误，目前只支持文本文件。'
        try:
            with open(seed_file_path, 'r') as file:
                seed2 = int(file.readline().strip())
                print('seed2:', seed2)
        except Exception:
            return 400, '提取密钥文件有错误，与模型不兼容！'
        self.seed2 = seed2
        return 200, '提取密钥文件已成功加载。'
    
    def stego_audio_upload_checker(self, stego_audio_file):
        stego_audio_file_path = os.path.join(self.default_save_internal_dir, stego_audio_file.filename)
        stego_audio_file.save(stego_audio_file_path)
        if not self._is_wav(stego_audio_file_path):
            return 400, '您上传的载密音频文件格式有误，目前只支持WAV音频文件。'
        self.stego_audio_file_upload_path = stego_audio_file_path
        print("stego_audio_path:",self.stego_audio_file_upload_path)
        return 200, '含密文件已成功加载。'
    
    


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

