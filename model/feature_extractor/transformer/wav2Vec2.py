# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, Wav2Vec2Model


# -

class TransformerBaseLine(nn.Module):
    def __init__(self, pretrain_feat="extract_features"):
        super().__init__()

        assert pretrain_feat in ["last_hidden_state", "extract_features"]
        self.pretrain_feat = pretrain_feat
        # The channels of used features for the pretrained model is 512 when using
        # the 'extract_features',  but 768 when ["last_hidden_state"] is used.
        C_features = 512 if pretrain_feat == "extract_features" else 768

        self.pretrain_model = Wav2Vec2Model.from_pretrained(
            "/usr/local/ay_data/0-model_weights/models--facebook--wav2vec2-base-960h"
        )

    def build_final_block(self):
        copied_layers = [deepcopy(self.pretrain_model.encoder.layers[i]) for i in range(6, 12)]
        self.copied_transformer = nn.ModuleList(copied_layers)

    def copy_final_stage(self):
        # self.block4_copied = self.build_final_block()
        self.build_final_block()

    def extract_feature(self, x):
        extract_features = self.pretrain_model.feature_extractor(x)
        extract_features = extract_features.transpose(1, 2)

        hidden_states, extract_features = self.pretrain_model.feature_projection(extract_features)
        hidden_states = self.pretrain_model._mask_hidden_states(
            hidden_states, mask_time_indices=None, attention_mask=None
        )

        #### split encoder process
        encoder = self.pretrain_model.encoder

        position_embeddings = encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = encoder.layer_norm(hidden_states)
        hidden_states = encoder.dropout(hidden_states)
        #### In original Wav2Vec, encoder has 12 layers
        for layer in encoder.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < encoder.config.layerdrop) else False
            if not skip_the_layer:
                layer_outputs = layer(hidden_states, attention_mask=None, output_attentions=None)
                hidden_states = layer_outputs[0]

        return hidden_states

    def get_main_stem(self):
        encoder = self.pretrain_model.encoder
        return [
            self.pretrain_model.feature_extractor,
            self.pretrain_model.feature_projection,
            encoder.pos_conv_embed, encoder.layer_norm, encoder.layers[0:6]
        ]

    def get_content_stem(self):
        encoder = self.pretrain_model.encoder
        return [encoder.layers[6:]]

    def get_vocoder_stem(self):
        return [self.copied_transformer]

    def preprocess(self, x, **kwargs):
        return x[:, 0, :]
    
    def get_hidden_state(self, x):
        extract_features = self.pretrain_model.feature_extractor(x)
        extract_features = extract_features.transpose(1, 2)

        hidden_states, extract_features = self.pretrain_model.feature_projection(extract_features)
        hidden_states = self.pretrain_model._mask_hidden_states(
            hidden_states, mask_time_indices=None, attention_mask=None
        )

        #### split encoder process
        encoder = self.pretrain_model.encoder

        position_embeddings = encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = encoder.layer_norm(hidden_states)
        hidden_states = encoder.dropout(hidden_states)
        #### In original Wav2Vec, encoder has 12 layers
        for layer in encoder.layers[0:6]:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < encoder.config.layerdrop) else False
            if not skip_the_layer:
                layer_outputs = layer(hidden_states, attention_mask=None, output_attentions=None)
                hidden_states = layer_outputs[0]

        return hidden_states


    def get_final_feature(self, hidden_states):
        encoder = self.pretrain_model.encoder
        for layer in encoder.layers[6:]:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < encoder.config.layerdrop) else False
            if not skip_the_layer:
                layer_outputs = layer(hidden_states, attention_mask=None, output_attentions=None)
                hidden_states = layer_outputs[0]

        return hidden_states.mean(1)

    def get_final_feature_copyed(self, hidden_states):
        encoder = self.pretrain_model.encoder
        for layer in self.copied_transformer:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < encoder.config.layerdrop) else False
            if not skip_the_layer:
                layer_outputs = layer(hidden_states, attention_mask=None, output_attentions=None)
                hidden_states = layer_outputs[0]

        return hidden_states.mean(1)

# + tags=["active-ipynb", "style-student"]
# x = torch.rand(10, 69000)
# model = BaseLine()
#
# model.extract_feature(x)
