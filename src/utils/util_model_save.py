'''
Author: your name
Date: 2021-10-20 10:41:53
LastEditTime: 2021-10-20 10:52:28
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /mdap-multi-task/src/utils/util_model_save.py
'''
import torch
import os
from transformers import (
    BertTokenizer,
    PreTrainedModel
)


def save_model(tokenizer: BertTokenizer,
               model: PreTrainedModel,
               save_dir: str):
    """Save PreTrainedModel

    Args:
        tokenizer (BertTokenizer): [tokenizer]
        model (PreTrainedModel): [model]
        save_dir (str): [model save dir]
    """
    model.config.to_json_file(
                            os.path.join(save_dir, "config.json")
                            )
    torch.save(
        model.state_dict(),
        os.path.join(save_dir, "pytorch_model.bin")
    )
    tokenizer.save_pretrained(save_dir)