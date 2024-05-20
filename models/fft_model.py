import torch
import torch.nn as nn
import torchaudio.transforms as T

import torchvision.models as models


class FFTModel(nn.Module):
    def __init__(self, n_classes):
        super(FFTModel, self).__init__()
        # https://pytorch.org/audio/stable/tutorials/audio_feature_extractions_tutorial.html#sphx-glr-tutorials-audio-feature-extractions-tutorial-py
        sample_rate = 500
        n_fft = 512
        win_length = None
        hop_length = 265
        n_mels = 128
        self.fft = T.MelSpectrogram(sample_rate=sample_rate,
                                 n_fft=n_fft,
                                 win_length=win_length,
                                 hop_length=hop_length,
                                 center=True,
                                 pad_mode="reflect",
                                 power=2.0,
                                 norm="slaney",
                                 n_mels=n_mels,
                                 mel_scale="htk",
                                 )

        self.bn = nn.BatchNorm2d(12)

        self.resnet18 = models.resnet18(pretrained=False,num_classes=n_classes)

    def forward(self, X):
        X = self.fft(X)
        X = self.bn(X)
        # return self.net(X)
        return self.resnet18(X)
