# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import pytorch_lightning as pl
import torch
import torch.nn as nn
from ay2.torch.deepfake_detection import DeepfakeAudioClassification

# + tags=["active-ipynb", "style-commentate"]
# from wav2Vec2 import BaseLine
# -

from .wav2Vec2 import BaseLine


class Wav2Vec2_lit(DeepfakeAudioClassification):
    def __init__(self, pretrain_feat="last_hidden_state", backend="linear", **kwargs):
        super().__init__()
        self.model = BaseLine(pretrain_feat=pretrain_feat, backend=backend)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=None)
        self.save_hyperparameters()
    
    
    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        loss = self.loss_fn(batch_res["logit"], label.type(torch.float32))
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            # self.model.parameters(), lr=0.0001, weight_decay=0.0001
            self.model.parameters(), lr=0.0001, weight_decay=0.00001
        )
        return [optimizer]

