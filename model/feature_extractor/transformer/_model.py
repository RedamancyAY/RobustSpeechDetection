# %load_ext autoreload
# %autoreload 2

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torchaudio
from copy import deepcopy
import torchaudio

MODEL_URL = "https://github.com/descriptinc/lyrebird-wav2clip/releases/download/v0.1.0-alpha/Wav2CLIP.pt"


def get_model(device="cpu", pretrained=True, frame_length=None, hop_length=None):
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            MODEL_URL, map_location=device, progress=True
        )
        model = ResNetExtractor(
            checkpoint=checkpoint,
            scenario="finetune", #frozen
            transform=True,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    else:
        model = ResNetExtractor(
            scenario="supervise", frame_length=frame_length, hop_length=hop_length
        )
    model.to(device)
    return model


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.model = get_model(pretrained=True).encoder
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=187) # (257, 257)
        # self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=446, hop_length=215) # (224, 224)
        
        # self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=353) # original, (257, 136)

    
    def preprocess(self, x, stage='test'):
        # x = self.model.spectrogram(x)
        x = self.spectrogram(x)
        x = torch.log(x + 1e-7)
        # x = (x - torch.mean(x)) / (torch.std(x) + 1e-9)

        
        x =  (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (
            torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9
        )
        # min_x, _ = torch.min(x.reshape(x.shape[0], -1), dim=1)
        # max_x, _ = torch.max(x.reshape(x.shape[0], -1), dim=1)
        # x = (x - min_x[:, None, None, None]) / (max_x[:,None, None, None])
        
        return x
    
    def build_final_block(self):
        self.layer4_copy = deepcopy(self.model.layer4)

    def copy_final_stage(self):
        # self.block4_copied = self.build_final_block()
        self.build_final_block()


    def get_main_stem(self):
        return [self.model.conv1, self.model.bn1, self.model.layer1, self.model.layer2,self.model.layer3]

    def get_content_stem(self):
        return [self.model.layer4]

    def get_vocoder_stem(self):
        return [self.layer4_copy]

    
    def get_hidden_state(self, x):
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x


    def feature_norm(self, code):
        code_norm = code.norm(p=2,dim=1, keepdim=True) / 10.
        code = torch.div(code, code_norm)
        return code
    
    def get_final_feature(self, x):
        
        conv_feat = self.model.layer4(x)
        x = self.model.avgpool(conv_feat)
        x = x.reshape(x.size(0), -1)

        x = self.feature_norm(x)
        
        return x, conv_feat
    
    def get_final_feature_copyed(self, x): 
        conv_feat = self.layer4_copy(x) # (B, 512, 9, 9)

        
        x = self.model.avgpool(conv_feat)
        x = x.reshape(x.size(0), -1)

        x = self.feature_norm(x)

        
        return x, conv_feat
