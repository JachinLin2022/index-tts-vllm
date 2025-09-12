# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import logging
import os
import re
from collections import OrderedDict

import torch
import yaml


def load_checkpoint(model: torch.nn.Module, model_pth: str) -> dict:
    # checkpoint = torch.load(model_pth, map_location='cpu')
    # checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
    # model.load_state_dict(checkpoint, strict=False)
    # info_path = re.sub('.pth$', '.yaml', model_pth)
    # configs = {}
    # if os.path.exists(info_path):
    #     with open(info_path, 'r') as fin:
    #         configs = yaml.load(fin, Loader=yaml.FullLoader)
    # return configs
    
    checkpoint = torch.load(model_pth, map_location='cpu')
    checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # Get the state dict of the current model
    model_state_dict = model.state_dict()

    # Iterate over the checkpoint's state dict
    for key in checkpoint.keys():
        # Check if the key exists in the current model
        if key in model_state_dict:
            # Get the weights from the checkpoint and the model
            ckpt_w = checkpoint[key]
            mdl_w = model_state_dict[key]

            # If shapes do not match, attempt to resize
            if ckpt_w.shape != mdl_w.shape:
                print(f">> Mismatch found for key: {key}. Checkpoint shape: {ckpt_w.shape}, Model shape: {mdl_w.shape}")
                # Handle the specific case of extending an embedding layer (or similar)
                # This logic is for when the model's weight matrix is larger in the first dimension
                if len(ckpt_w.shape) > 0 and len(mdl_w.shape) > 0 and \
                   mdl_w.shape[0] > ckpt_w.shape[0] and \
                   mdl_w.shape[1:] == ckpt_w.shape[1:]:
                    
                    print(f">> Attempting to resize '{key}'...")
                    # Copy the old weights from the checkpoint into the new model's weights
                    num_to_copy = ckpt_w.shape[0]
                    mdl_w.data[:num_to_copy] = ckpt_w.data

                    # Initialize the new, extended part of the weights
                    # Using normal distribution initialization, similar to how embeddings are often initialized
                    mdl_w.data[num_to_copy:].normal_(mean=0.0, std=0.02)
                    
                    print(f">> Successfully resized and initialized '{key}'.")
                    # Update the checkpoint dictionary with the resized tensor
                    checkpoint[key] = mdl_w.data
                else:
                    print(f"!! Warning: Could not automatically resize key '{key}'. It will be skipped.")

    # Load the (potentially modified) state dict. strict=False will ignore
    # any keys that are in one dict but not the other (e.g., if we failed to resize).
    model.load_state_dict(checkpoint, strict=False)

    # Load accompanying config file if it exists
    info_path = re.sub('.pth$', '.yaml', model_pth)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs
