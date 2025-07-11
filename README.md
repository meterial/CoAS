# CoAS: Composite Audio Steganography Based on Text and Speech Synthesis
This repository hosts the official PyTorch implementation of the paper: ["**CoAS: Composite Audio Steganography Based on Text and Speech Synthesis**"](https://ieeexplore.ieee.org/abstract/document/11036088) (Accepted by IEEE TIFS 2025).
## Method
![method](fig/overview.png)
We propose Composite Audio Steganography (CoAS), a method based on text and speech synthesis that leverages the multi-carrier characteristic of audio data by utilizing side-channel information from the transcript to facilitate the steganographic process. We first maps the secret message to Gaussian noise in a distribution-preserving manner and embeds it into the generation process of a diffusion model audio sequence. To address the reduced audio diversity caused by using a fixed random seed as a key, we embed the key into the audio text, which is then retrieved by the receiver via speech recognition. This approach allows the system to randomly select a key for each transmission, ensuring both accurate message extraction and the diversity of the generated audio for enhanced concealment.
## Getting Started
## Acknowledgements
We heavily borrow the code from [FastDiff](https://github.com/Rongjiehuang/FastDiff), [ProDiff](https://github.com/Rongjiehuang/ProDiff) and [Discop](https://github.com/comydream/Discop). We appreciate the authors for sharing their code.
## Ciation
If you find our work useful for your research, please consider citing the following paper:
```
@article{li2025coas,
  title={CoAS: Composite Audio Steganography Based on Text and Speech Synthesis},
  author={Li, Yiming and Chen, Kejiang and Wang, Yaofei and Zhang, Xin and Wang, Guanjie and Zhang, Weiming and Yu, Nenghai},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025},
  publisher={IEEE}
}
```