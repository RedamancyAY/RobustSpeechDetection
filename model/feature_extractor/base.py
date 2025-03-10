import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):


    def copy_final_stage(self):
        raise NotImplementedError

    def get_hidden_state(self, x):
        raise NotImplementedError

    def get_final_feature(self, x):
        raise NotImplementedError

    def get_final_feature_copyed(self, x): 
        raise NotImplementedError

    
    def forward(self, x):
        x = self.get_hidden_state(x)
        x = self.get_final_feature(x)
        return x

