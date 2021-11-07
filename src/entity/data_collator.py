'''
Author: your name
Date: 2021-10-20 11:32:26
LastEditTime: 2021-10-20 11:34:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /mdap-multi-task/src/common/data_collator.py
'''
import torch
from transformers.data.data_collator import InputDataClass, DefaultDataCollator
from typing import List, Union, Dict

class NLPDataCollator:
    def __call__(
        self, features: List[Union[InputDataClass, Dict]]
    ) -> Dict[str, torch.Tensor]:
        first = features[0]
        if isinstance(first, dict):
            # NLP data sets current works presents features as lists of dictionary
            # (one per example), so we  will adapt the collate_batch logic for that
            if "labels" in first and first["labels"] is not None:
                if first["labels"].dtype == torch.int64:
                    labels = torch.tensor(
                        [f["labels"] for f in features], dtype=torch.long
                    )
                else:
                    labels = torch.tensor(
                        [f["labels"] for f in features], dtype=torch.float
                    )
                batch = {"labels": labels}
            for k, v in first.items():
                if k != "labels" and v is not None and not isinstance(v, str):
                    batch[k] = torch.stack([f[k] for f in features])
            return batch
        else:
            # otherwise, revert to using the default collate_batch
            return DefaultDataCollator().collate_batch(features)