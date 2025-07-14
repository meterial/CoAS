import os

import torch

from tasks.tts.dataset_utils import FastSpeechWordDataset
from tasks.tts.tts_utils import load_data_preprocessor
import numpy as np
from modules.FastDiff.module.util import compute_hyperparams_given_schedule, sampling_given_noise_schedule, sampling_given_noise_schedule_extra

import os

import torch

from modules.FastDiff.module.FastDiff_model import FastDiff
from utils.ckpt_utils import load_ckpt
from utils.hparams import set_hparams


class BaseTTSInfer:
    def __init__(self, hparams, device=None, default_save_internal_dir='audio_backend/internal', default_save_uploads_dir='audio_backend/uploads'):
        
        self.default_save_internal_dir = default_save_internal_dir
        self.default_save_uploads_dir = default_save_uploads_dir
        os.makedirs(self.default_save_internal_dir, exist_ok=True)
        os.makedirs(self.default_save_uploads_dir, exist_ok=True)
        # os.makedirs('experiment/audio_test', exist_ok=True)
        # os.makedirs('experiment/message_test', exist_ok=True)
        os.makedirs('../capacity_test/audio_data/audio_test', exist_ok=True)
        os.makedirs('../capacity_test/audio_data/message_test', exist_ok=True)


        self.audio_text_embeded = '' #audio text
        self.seed1 = None #seed1
        self.message_embedding = '' #secret message
        self.compress_messbits = [] #compress message bits
        self.embed_bits = []
        self.stego_audio_file_save_path = '' #save stego audio

        self.audio_text_extracting = '' #audio text, upload or recognize
        self.seed2 = None #seed2
        self.stego_audio_file_upload_path = '' #upload stego audio
        self.extract_bits = []
        self.bits_to_decompress = [] #bits to be decompressed
        self.extracted_message_file_path = os.path.join(self.default_save_internal_dir, 'extracted_message.txt')

        self.stego_mimetype = 'audio/wav' 
        self.extracted_mimetype = 'text/plain'
        
        # if device is None:
        #     device = 'cuda' if torch.cuda.is_available() else 'cpu'

        device = 'cuda'
        self.hparams = hparams
        self.device = device
        self.data_dir = hparams['binary_data_dir']
        self.preprocessor, self.preprocess_args = load_data_preprocessor()
        self.ph_encoder = self.preprocessor.load_dict(self.data_dir)
        self.spk_map = self.preprocessor.load_spk_map(self.data_dir)
        self.ds_cls = FastSpeechWordDataset
        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder, self.diffusion_hyperparams, self.noise_schedule = self.build_vocoder()
        self.vocoder.eval()
        self.vocoder.to(self.device)

    def build_model(self):
        raise NotImplementedError

    def forward_model(self, inp):
        raise NotImplementedError
    
    def forward_model_extra(self, inp):
        raise NotImplementedError

    def build_vocoder(self):
        base_dir = self.hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        config = set_hparams(config_path, global_hparams=False)
        vocoder = FastDiff(audio_channels=config['audio_channels'],
                 inner_channels=config['inner_channels'],
                 cond_channels=config['cond_channels'],
                 upsample_ratios=config['upsample_ratios'],
                 lvc_layers_each_block=config['lvc_layers_each_block'],
                 lvc_kernel_size=config['lvc_kernel_size'],
                 kpnet_hidden_channels=config['kpnet_hidden_channels'],
                 kpnet_conv_size=config['kpnet_conv_size'],
                 dropout=config['dropout'],
                 diffusion_step_embed_dim_in=config['diffusion_step_embed_dim_in'],
                 diffusion_step_embed_dim_mid=config['diffusion_step_embed_dim_mid'],
                 diffusion_step_embed_dim_out=config['diffusion_step_embed_dim_out'],
                 use_weight_norm=config['use_weight_norm'])
        load_ckpt(vocoder, base_dir, 'model')

        # Init hyperparameters by linear schedule
        noise_schedule = torch.linspace(float(config["beta_0"]), float(config["beta_T"]), int(config["T"])).cuda()
        diffusion_hyperparams = compute_hyperparams_given_schedule(noise_schedule)

        # map diffusion hyperparameters to gpu
        for key in diffusion_hyperparams:
            if key in ["beta", "alpha", "sigma"]:
                diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()
        diffusion_hyperparams = diffusion_hyperparams

        if config['noise_schedule'] != '':
            noise_schedule = config['noise_schedule']
            if isinstance(noise_schedule, list):
                noise_schedule = torch.FloatTensor(noise_schedule).cuda()
        else:
            # Select Schedule
            try:
                reverse_step = int(self.hparams.get('N'))
            except:
                print(
                    'Please specify $N (the number of revere iterations) in config file. Now denoise with 4 iterations.')
                reverse_step = 4
            if reverse_step == 1000:
                noise_schedule = torch.linspace(0.000001, 0.01, 1000).cuda()
            elif reverse_step == 200:
                noise_schedule = torch.linspace(0.0001, 0.02, 200).cuda()

            # Below are schedules derived by Noise Predictor.
            # We will release codes of noise predictor training process & noise scheduling process soon. Please Stay Tuned!
            elif reverse_step == 8:
                noise_schedule = [6.689325005027058e-07, 1.0033881153503899e-05, 0.00015496854030061513,
                                  0.002387222135439515, 0.035597629845142365, 0.3681158423423767, 0.4735414385795593,
                                  0.5]
            elif reverse_step == 6:
                noise_schedule = [1.7838445955931093e-06, 2.7984189728158526e-05, 0.00043231004383414984,
                                  0.006634317338466644, 0.09357017278671265, 0.6000000238418579]
            elif reverse_step == 4:
                noise_schedule = [3.2176e-04, 2.5743e-03, 2.5376e-02, 7.0414e-01]
            elif reverse_step == 3:
                noise_schedule = [9.0000e-05, 9.0000e-03, 6.0000e-01]
            else:
                raise NotImplementedError

        if isinstance(noise_schedule, list):
            noise_schedule = torch.FloatTensor(noise_schedule).cuda()

        return vocoder, diffusion_hyperparams, noise_schedule

    def run_vocoder(self, c,message,seed1,compress_messbits):
        c = c.transpose(2, 1)
        audio_length = c.shape[-1] * self.hparams["hop_size"]
        print('hop_size:', self.hparams["hop_size"])
        y = sampling_given_noise_schedule(
            self.vocoder, (1, 1, audio_length), self.diffusion_hyperparams, self.noise_schedule, condition=c, ddim=False, return_sequence=False,message=message,seed1=seed1,compress_messbits=compress_messbits)
        return y
    
    def run_vocoder_extra(self, c,audio,seed2):
        c = c.transpose(2, 1)
        audio_length = c.shape[-1] * self.hparams["hop_size"]
        y = sampling_given_noise_schedule_extra(
            self.vocoder, (1, 1, audio_length), self.diffusion_hyperparams, self.noise_schedule, condition=c, ddim=False, return_sequence=False,audio=audio, seed2=seed2)
        return y
    
    def preprocess_input(self, inp):
        """
        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        preprocessor, preprocess_args = self.preprocessor, self.preprocess_args
        text_raw = inp['text']
        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', 'SPK1')
        ph, txt = preprocessor.txt_to_ph(
            preprocessor.txt_processor, text_raw, preprocess_args)
        ph_token = self.ph_encoder.encode(ph)
        spk_id = self.spk_map[spk_name]
        item = {'item_name': item_name, 'text': txt, 'ph': ph, 'spk_id': spk_id, 'ph_token': ph_token}
        item['ph_len'] = len(item['ph_token'])
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)
        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'spk_ids': spk_ids,
        }
        return batch

    def postprocess_output(self, output):
        return output

    def infer_once(self, inp, message, seed1, compress_messbits):
        inp = self.preprocess_input(inp)
        print(f'inp:{inp}')
        output = self.forward_model(inp,message=message,seed1=seed1,compress_messbits=compress_messbits)
        output = self.postprocess_output(output=output)
        return output
    
    def infer_once_extra(self, inp,audio, seed2):
        inp = self.preprocess_input(inp)
        output = self.forward_model_extra(inp,audio=audio,seed2=seed2)
        output = self.postprocess_output(output)
        return output

    @classmethod
    def embd_run(cls,text,message,seed1):
        from utils.hparams import set_hparams
        from utils.hparams import hparams as hp
        from utils.audio import save_wav

        # print(hp['text'])
        hp1 = hp
        print(len(hp1))
        for i, (k, v) in enumerate(sorted(hp1.items())):
            # print(f"\033[;33;m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
            # print(f"{k}:{v}, ", end="\n" if i % 5 == 4 else "")
            print(f"{k}:{v}, ", end="\n")
        set_hparams()
        hp2 = hp
        print(len(hp2))
        for i, (k, v) in enumerate(sorted(hp2.items())):
            # print(f"\033[;33;m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
            # print(f"{k}:{v}, ", end="\n" if i % 5 == 4 else "")
            print(f"{k}:{v}, ", end="\n")

        infer_ins = cls(hp)#实例化

        print(infer_ins)
        
        #由此加载参数完毕
        
        inp = {'text': text}

        out = infer_ins.infer_once(inp,message=message,seed1=seed1)

        os.makedirs('infer_out', exist_ok=True)
        if message == '':
            save_wav(out, f'infer_out/{hp["text"].split()[0]}_cover.wav', hp['audio_sample_rate'])
        else :
            save_wav(out, f'infer_out/{hp["text"].split()[0]}_stego.wav', hp['audio_sample_rate'])

    @classmethod
    def extra_run(cls,audio,text,seed2):
        from utils.hparams import set_hparams
        from utils.hparams import hparams as hp
        from utils.audio import save_wav

        print(hp['text'])

        set_hparams(text=text)

        print(hp['text'])

        inp = {'text': text}

        print(f'inp:{inp}')

        infer_ins = cls(hp)
        message_out = infer_ins.infer_once_extra(inp,audio=audio,seed2=seed2)

        return message_out

        # stego_out = out.copy()

        # os.makedirs('infer_out', exist_ok=True)
        # save_wav(stego_out, f'infer_out/{hp["text"].split()[0]}_stego.wav', hp['audio_sample_rate'])
        # save_wav(out, f'infer_out/{hp["text"].split()[0]}_cover.wav', hp['audio_sample_rate'])


        # set_hparams()
        # inp = {
        #     'text': hp['text']
        # }
        # text_list = [
        #     'But as for me, if I must be bereaved, I will be bereaved!', 
        #     'Joseph`s brother, with his other brothers, for he said: "Perhaps a fatal accident may befall him."',
        #     'one`s bag of money in his sack. When they and their father dsaaw their bags of money, they became afraid.',
        #     'A3nd it was a matter of course that in the Middle Ages, when the craftsmen took care that beautiful form should always be a part of their productions whatever they were.',
        #     'Now,as all books not primarily intended as picture books consist principally of types composed to form letter press.',
        #     'It is of the first importance that the letter used should be fine in form.',
        #     'Especially as no more time is occupied, or cost incurred, in casting, setting, or printing beautiful letters.',
        #     'The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves.'
        #     ]
        # text_list =[
        #     'So when they had finished eating the grain they had brought from Egypt, their father said to them: “Return and buy a little food for us.”',
        #     'Then Judah said to him: “The man clearly warned us, ‘You must not see my face again unless your brother is with you.’”',
        #     'If you send our brother with us, we will go down and buy food for you.'
        # ]
        # text_list = [
        #     # 'Please call Stella.',
        #     # 'Ask her to bring these things with her from the store.',
        #     # 'Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.'
        #     'Researchers believe that the new technology could help doctors find out if the treatment is effective.',
        #     'Researchers believe that the most important factor in determining the risk of developing a cancer is genetic.',
        #     'Researchers believe the most effective way to reduce the amount of radiation that we can receive is by using a more efficient and more accurate radiotherapy technique.'
        # ]

        # text_list = [
        #     'Researchers believe the most effective way to reduce the amount of radiation that we can receive is by using a more efficient and more accurate radiotherapy technique.'
        # ]

        # text_list = [
        #     'Attribuor object has no attribute. '
        # ]
        
        
        # for i in text_list:
        #     inp ={ 'text': i}
        #     infer_ins = cls(hp)
        #     out = infer_ins.infer_once(inp)
        #     os.makedirs('infer_out', exist_ok=True)
        #     save_wav(out, f'infer_out/{i.split()[0]}.wav', hp['audio_sample_rate'])
