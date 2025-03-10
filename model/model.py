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
import math
import random
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange
# -

from torchvision.transforms import v2

try:
    from .feature_extractor import ResNet
    from .feature_extractor.transformer import TransformerBaseLine
except ImportError:
    from feature_extractor import ResNet
    from feature_extractor.transformer import TransformerBaseLine


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

    def forward(self, x, y):
        h, w = x.shape[2:4]
        short_cut = x
        x = rearrange(x, "b c h w -> b (h w) c")
        y = rearrange(y, "b c h w -> b (h w) c")
        x, _ = self.multihead_attn(y, x, x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x + short_cut
        # return x


# + editable=true slideshow={"slide_type": ""}
class AudioModel(nn.Module):
    def __init__(
        self,
        feature_extractor: str,
        dims=[32, 64, 64, 64, 128],
        n_blocks=[1, 1, 1, 2, 1],
        n_heads=[1, 2, 2, 4, 1, 1],
        samples_per_frame=640,
        gru_node=128,
        gru_layers=3,
        fc_node=128,
        num_classes=1,
        vocoder_classes=8,
        adv_vocoder=False,
        cfg=None,
        args=None,
    ):
        super().__init__()

        self.cfg = cfg

        # self.norm = LayerNorm(48000)
        self.dims = dims
        self.feature_extractor = feature_extractor
        if feature_extractor == "ResNet":
            self.feature_model = ResNet()
            final_dim = 512
        elif feature_extractor.lower() == "transformer":
            self.feature_model = TransformerBaseLine()
            final_dim = 768
        else:
            raise ValueError(
                f"Unsupported feature extractor: {feature_extractor}, please"
                "choose from ResNet or transformer."
            )

        self.feature_model.copy_final_stage()

        self.dropout = nn.Dropout(0.1)
        self.cls_content = nn.utils.weight_norm(nn.Linear(final_dim, 1, bias=False))
        if cfg.one_stem:
            self.content_based_cls = nn.utils.weight_norm(nn.Linear(final_dim, 1, bias=False))
            if cfg.use_only_vocoder_stem:
                self.vocoder_based_cls = nn.utils.weight_norm(nn.Linear(final_dim, 1, bias=False))

        self.cls_voc = nn.utils.weight_norm(nn.Linear(final_dim, vocoder_classes + 1, bias=False))

        if cfg.use_f0_loss:
            self.cls_f0 = nn.utils.weight_norm(nn.Linear(final_dim, 192, bias=False))

        self.cls_final = nn.Sequential(
            # nn.utils.weight_norm(nn.Linear(final_dim * 2, final_dim * 2, bias=False)),
            # nn.BatchNorm1d(final_dim * 2),
            # nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(final_dim * 2, 1, bias=False)),
        )

        self.cls_speed = nn.utils.weight_norm(nn.Linear(final_dim, 16, bias=False))
        self.cls_compression = nn.utils.weight_norm(nn.Linear(final_dim, 10, bias=False))

        self.debug = 0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cross_attention1 = CrossAttention(512)
        self.cross_attention2 = CrossAttention(512)
        self.transform = v2.RandomErasing()

    def get_content_stream_modules(
        self,
    ):
        return self.feature_model.get_content_stem() + [self.cls_content]

    def module_similaryity(self):
        loss = []
        for p1, p2 in zip(
            self.feature_model.get_final_block_parameters(),
            self.feature_model.get_copied_final_block_parameters(),
        ):
            _loss = 1 - F.cosine_similarity(p1.view(1, -1), p2.view(1, -1))[0]
            loss.append(_loss)
        loss = sum(loss) / len(loss)
        return loss

    def feature_norm(self, code):
        code_norm = code.norm(p=2, dim=1, keepdim=True) / 10.0
        code = torch.div(code, code_norm)
        return code

    def fuse_stem_featurs(self, feat1, feat2):
        ### feat1 and feat2 's dim is 2
        if feat1.ndim == 2:
            feat = torch.concat([feat1, feat2], dim=-1)
            return feat

        ### dim is 4
        feat1 = self.cross_attention1(feat1, feat2)
        feat2 = self.cross_attention2(feat2, feat1)
        feat = torch.concat([feat1, feat2], dim=1)
        feat = self.avgpool(feat)
        feat = feat.reshape(feat.size(0), -1)
        # feat = self.feature_norm(feat)
        return feat

    def forward(self, x:torch.Tensor, stage: str="test", batch: dict=None, one_stem: bool=False):
        """Defines the forward pass of the model.

        This method processes input speech data `x` through various components of the model,
        including feature extraction and classification. It handles different stages
        (train/test) and configurations (e.g., style shuffle, feature blending).

        Args:
            x: Input speech tensor. It should be prrprocessed by the feature extractor in the `_shared_pred` of the lit model.
            stage: Stage of execution, either "train" or "test". Defaults to "test".
            batch: Dictionary containing additional batch information. Defaults to None.
            one_stem: Boolean flag indicating whether to use one stem. Defaults to False.

        Returns:
            A dictionary containing various features and logits:
                hidden_states: Hidden states from the main stream.
                content_feature: Content feature extracted from the speech.
                speed_logit: Logits for speed prediction.
                compression_logit: Logits for compression prediction.
                f0_logit: Logits for F0 prediction if use_f0_loss is enabled.
                vocoder_feature: Vocoder feature extracted from the speech.
                vocoder_logit: Logits for vocoder prediction.
                content_voc_logit: Vocoder logits based on content feature.
                feature: Final blended features.
                logit: Final classification logits.
                shuffle_logit: Logits for shuffled features if feat_shuffle is enabled.

        Raises:
            Exception: If any internal operation fails (e.g., feature extraction).

        Example:
            >>> x = torch.randn(32, 1, 48000)
            >>> outputs = model.forward(x, stage="train")
        """
        
        batch_size = x.shape[0]
        res = {}

        ####  1. Input the speech, the feature extractor first preprocess the speech (for example, Log-Freency Spectrogram),
        #### and then use the `main stream` to get the hidden_states of the speech.
        res["hidden_states"] = self.feature_model.get_hidden_state(x)

        #### 2.1 `get_final_feature` means that uses the content stream to process the hidden_states and get the content feature.
        #### Note that the content stream is the original block 4.
        #### and the synthesizer stream is the copy of the block 4.
        if self.feature_extractor == "ResNet":
            res["content_feature"], conv_feat1 = self.feature_model.get_final_feature(res["hidden_states"])
        else:
            res["content_feature"] = self.feature_model.get_final_feature(res["hidden_states"])

        #### 2.2 Compute the speed and compression logit of the speech.
        if one_stem:
            res["content_based_cls_logit"] = self.content_based_cls(self.dropout(res["content_feature"])).squeeze(-1)
        res["speed_logit"] = self.cls_speed(self.dropout(res["content_feature"]))
        res["compression_logit"] = self.cls_compression(self.dropout(res["content_feature"]))
        if self.cfg.use_f0_loss:
            res["f0_logit"] = self.cls_f0(self.dropout(res["content_feature"]))

        # learn a vocoder feature extractor and classifier

        ##### 3.1 `get_final_feature_copyed` means that uses the vocoder stream to process the hidden_states and get the vocoder feature.
        hidden_states = res["hidden_states"]
        if self.feature_extractor == "ResNet":
            (
                res["vocoder_feature"],
                conv_feat2,
            ) = self.feature_model.get_final_feature_copyed(hidden_states)
        else:
            res["vocoder_feature"] = self.feature_model.get_final_feature_copyed(hidden_states)

        #### 3.2 Compute the vocoder logit of the speech.
        if one_stem and self.cfg.use_only_vocoder_stem:
            res["vocoder_based_cls_logit"] = self.vocoder_based_cls(self.dropout(res["vocoder_feature"])).squeeze(-1)

        res["vocoder_logit"] = self.cls_voc(self.dropout(res["vocoder_feature"]))
        
        #### 4. Computer the vocoder logit of the speech but based on the content feature.
        res["content_voc_logit"] = self.cls_voc(self.dropout(res["content_feature"]))


        #### 5. `Synthesizer Feature Augmentation Strategy`
        #### 5.1 feature blending
        voc_feat = res["vocoder_feature"]
        content_feat = res["content_feature"]
        if stage == "train" and self.cfg.style_shuffle:
            shuffle_id = torch.randperm(batch_size)
            shuffle_id = get_permutationID_by_label(batch["label"])
            voc_feat = exchange_mu_std(res["vocoder_feature"], res["vocoder_feature"][shuffle_id], dim=-1)
            shuffle_id = get_permutationID_by_label(batch["label"])
            content_feat = exchange_mu_std(res["content_feature"], res["content_feature"][shuffle_id], dim=-1)
            
        #### 5.2 make final decision after feature blending
        res["feature"] = self.fuse_stem_featurs(res["content_feature"], res["vocoder_feature"])
        final_feat = self.fuse_stem_featurs(content_feat, voc_feat)
        res["logit"] = self.cls_final(self.dropout(final_feat)).squeeze(-1)

        #### 5.3 feature shuffle operation
        if stage == "train" and self.cfg.feat_shuffle:
            shuffle_id = torch.randperm(batch_size)
            res["shuffle_logit"] = self.cls_final(
                self.dropout(
                    self.fuse_stem_featurs(content_feat, voc_feat[shuffle_id])
                )
            ).squeeze(-1)
            batch["shuffle_label"] = deepcopy(batch["label"])
            for i in range(batch_size):
                if batch["label"][shuffle_id[i]] == 0 or batch["label"][i] == 0:
                    batch["shuffle_label"][i] = 0
                else:
                    batch["shuffle_label"][i] = 1

        if hasattr(self, "gradcam") and self.gradcam:
            logit = torch.sigmoid(res["logit"])[:, None]  # (B, 1)
            logit = torch.concat([1 - logit, logit], dim=-1)  # (B, 2)
            return logit

        return res


# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# model = AudioModel(vocoder_classes=7)
# x = torch.randn(32, 1, 48000)
# _ = model(x)

# + tags=["active-ipynb"]
# # ckpt = torch.load(
# #     "/home/ay/data/DATA/1-model_save/0-Audio/Ours/LibriSeVoc_cross_dataset/version_7/checkpoints/best-epoch=3-val-auc=0.99.ckpt"
# # )
#
# # state_dict = ckpt["state_dict"]
#
# # state_dict2 = {key.replace("model.", "", 1): state_dict[key] for key in state_dict}
#
# # model.load_state_dict(state_dict2)
# -

def get_permutationID_by_label(label):
    x = label.cpu()
    index0 = np.where(x == 0)[0]
    index1 = np.where(x == 1)[0]

    shuffle_index0 = np.random.permutation(index0)
    shuffle_index1 = np.random.permutation(index1)

    new_index = np.ones_like(x)
    for i in range(len(index0)):
        new_index[index0[i]] = shuffle_index0[i]
    for i in range(len(index1)):
        new_index[index1[i]] = shuffle_index1[i]
    return new_index


def exchange_mu_std(x, y, dim=None):
    mu_x = torch.mean(x, dim=dim, keepdims=True)
    mu_y = torch.mean(y, dim=dim, keepdims=True)
    std_x = torch.std(x, dim=dim, keepdims=True)
    std_y = torch.std(y, dim=dim, keepdims=True)

    alpha = np.random.randint(50, 100) / 100
    target_mu = alpha * mu_x + (1 - alpha) * mu_y
    target_std = alpha * std_x + (1 - alpha) * std_y
    z = target_std * ((x - mu_x) / (std_x + 1e-9)) + target_mu


    ### Noisy Feature Mixup (NFM) https://github.com/erichson/NFM
    noise_level = 10
    add_noise_level = np.random.randint(0, noise_level) / 100
    mult_noise_level = np.random.randint(0, noise_level) / 100
    z = _noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level)
    return z


def _noise(x, add_noise_level=0.0, mult_noise_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    if add_noise_level > 0.0:
        add_noise = add_noise_level * np.random.beta(2, 5) * torch.FloatTensor(x.shape).normal_().to(x.device)
    if mult_noise_level > 0.0:
        mult_noise = (
            mult_noise_level * np.random.beta(2, 5) * (2 * torch.FloatTensor(x.shape).uniform_() - 1).to(x.device) + 1
        )
    return mult_noise * x + add_noise
