'''
Author: your name
Date: 2021-10-20 10:07:05
LastEditTime: 2021-11-07 12:33:34
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /mdap-multi-task/src/utils/util_model_eval.py
'''
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support 
from typing import Dict
from args.arg_common import LOG_DIR
from utils.util_log import get_logger
from transformers import BertTokenizer
from args.arg_common import (
    TASKS
)

loggger = get_logger("model_eval",
                     os.path.join(LOG_DIR, "model_eval.log")
                     )


def multitask_eval(multitask_model,
                   tokenizer: BertTokenizer,
                   features_dict: Dict,
                   le_dict: Dict,
                   average_dict: Dict):
    for task_name in TASKS:
        val_len = len(features_dict[task_name]["test"])
        y_pred = []
        y_true = []
        for index in range(0, val_len):
            
            batch = features_dict[task_name]["test"][index]["text"]         
            labels = features_dict[task_name]["test"][index]["label"]
            labels = le_dict[task_name].transform([labels])[0]
            
            inputs = tokenizer(batch, return_tensors="pt")["input_ids"]
            inputs = inputs.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            logits = multitask_model(inputs, task_name=task_name)[0]

            predictions = torch.argmax(
                torch.FloatTensor(torch.softmax(logits, dim=1).detach().cuda().tolist()),
                dim=1,
            )
            
            y_true.append(labels)
            y_pred.append(predictions.item())
            
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true,
                                                                   y_pred,
                                                                   average=average_dict[task_name]
                                                                   )
        metrics = {
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        loggger.info("The metric of [{}] task based of semh_model is : [{}]".format(task_name,
                                                                                    str(metrics)
                                                                                    )
                     )