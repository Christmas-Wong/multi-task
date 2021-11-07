'''
Author: your name
Date: 2021-10-22 17:55:44
LastEditTime: 2021-11-07 13:23:58
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /mdap/src/utils/util_common.py
'''
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from entity.dataset import Dataset
from transformers import BertTokenizer
from typing import Dict

def generate_feature(task_le: Dict,
                     tokenizer: BertTokenizer,
                     max_length: int,
                     dataset_dict: Dict,
                     columns_dict: Dict):
    
    def convert_to_useful_features(example_batch):
        features = tokenizer(
            example_batch["text"], max_length=max_length, pad_to_max_length=True
        )
        features["labels"] = task_le["location"].transform(example_batch["label"])
        return features

    def convert_to_department_features(example_batch):
        features = tokenizer(
            example_batch["text"], max_length=max_length, pad_to_max_length=True
        )
        features["labels"] = task_le["service"].transform(example_batch["label"])
        return features
    
    convert_func_dict = {
        "location": convert_to_useful_features,
        "service": convert_to_department_features
        }
    
    features_dict = {}
    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                                                                    convert_func_dict[task_name],
                                                                    batched=True,
                                                                    load_from_cache_file=False,
                                                                )
            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
            features_dict[task_name][phase].set_format(
                                                            type="torch", 
                                                            columns=columns_dict[task_name],
                                                        )
    return features_dict

def df_2_dataset(df: DataFrame, le: LabelEncoder, text: str, label: str, tokenizer) -> Dataset:
    """Transform DataFrame into DataSet

    Args:
        df (DataFrame): Source Data
        le (LabelEncoder): LabelEncoder
        text (str): Column Name of Text
        label (str): Column Name of Label
        tokenizer ([type]): Tokenizer

    Returns:
        Dataset: DataSet
    """
    x = list(df[text])
    df["label_id"] = le.transform(df[label].tolist())
    y = list(df["label_id"])
    x_tokenized = tokenizer(x, padding=True, truncation=True, max_length=512)
    result = Dataset(x_tokenized, y)
    return result

