'''
Author: your name
Date: 2021-10-21 22:35:14
LastEditTime: 2021-10-24 22:19:11
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /mdap/src/utils/util_read_file.py
'''
from typing import List
import json
import pandas as pd
import os
from pandas import DataFrame
import codecs
from args.arg_common import DATA_PATH, HELP_DATA
from utils.util_log import get_logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
import pickle



logger = get_logger("Read Data from File", None)

def get_label_encoder(path: str) -> LabelEncoder:
    """Load Label Encoder

    Args:
        path (str): Label Encoder Local File

    Returns:
        LabelEncoder: Label_encoder
    """
    file = open(path, 'rb')
    result = pickle.load(file)
    file.close()
    return result


def generate_label_encoder(df: DataFrame, label: str, save_path) -> LabelEncoder:
    """Generate Label Encoder & save to file

    Args:
        df (DataFrame): Train DataFrame
        label (str): Column Name of Label
        save_path ([type]): File Path to Save

    Returns:
        LabelEncoder: Label Encoder
    """
    if os.path.exists(save_path):
        return get_label_encoder(save_path)
    
    le = LabelEncoder()
    le.fit(df[label].tolist())
    output = open(save_path, 'wb')
    pickle.dump(le, output)
    return le


def get_data_csv(data_dir: str,
             train_file: str, 
             validation_file=None,
             test_file=None,
             train_test_ratio=0.3,
             train_valid_ratio=0.1,
             debug=False) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Get Train & Validation & Test DataFrame

    Args:
        data_dir (str): Data Dir
        train_file (str): Train File Name
        validation_file ([type], optional): Validation File Name. Defaults to None.
        test_file ([type], optional): Test File Name. Defaults to None.

    Returns:
        [type]: Train & Validation & Test DataFrame
    """
    train_df = pd.read_csv(os.path.join(data_dir, train_file), encoding="utf-8")
    valid_df = None
    test_df = None
    if validation_file:
        valid_df = pd.read_csv(os.path.join(data_dir, validation_file), encoding="utf-8")
    # else:
    #     train_df, valid_df = train_test_split(train_df,
    #                                           test_size=train_valid_ratio,
    #                                           random_state=42,
    #                                           shuffle=True,
    #                                           stratify=train_df["label"])
    if test_file:
        test_df = pd.read_csv(os.path.join(data_dir, test_file), encoding="utf-8")
    # else:
    #     train_df, valid_df = train_test_split(train_df,
    #                                             test_size=train_test_ratio,
    #                                             random_state=42,
    #                                             shuffle=True,
    #                                             stratify=train_df["label"])
    if debug:
        train_df = train_df.head(1600)
    return train_df, valid_df, test_df



def get_stopwords() -> List[str]:
    """
    Get Chinese Stopwords

    :return: List of stopwords
    """
    file = codecs.open(os.path.join(DATA_PATH, HELP_DATA["txt_stopwords"]), "r+", encoding="GBK")
    result = []
    for line in file.readlines():
        result.append(line.strip())
    logger.info("读取停用词个数[%d]", len(result))
    return result


def get_delete_words() -> List[str]:
    """
    Get Delete Words

    :return: List of delete words
    """
    file = codecs.open(os.path.join(DATA_PATH, HELP_DATA["txt_delete_words"]), "r+", encoding="UTF-8")
    result = []
    for line in file.readlines():
        result.append(line.strip())
    logger.info("读取删除词个数[%d]", len(result))
    return result


def get_replace_words() -> dict:
    """
    Get Replace Words

    :return: dict
    """
    result = json.load(open(os.path.join(DATA_PATH, HELP_DATA["json_replace_words"]), 'r', encoding='utf8'))
    logger.info("读取替换词表个数[%d]", len(result))
    return result


def get_emoji() -> dict:
    """
    Get emoji & zh

    :return: dict
    """
    result = json.load(open(os.path.join(DATA_PATH, HELP_DATA["json_emoji"]), 'r', encoding='utf8'))
    logger.info("读取emoji个数[%d]", len(result))
    return result

def get_source(file_path_file: str) -> DataFrame:
    """Get Source Data from a file.

    Args:
        file_path_file (str): [description]

    Returns:
        DataFrame: DataFrame of Source data
    """
    result = pd.read_csv(file_path_file, encoding='utf-8')
    logger.info("Read Source File with [{}] rows and [{}] columns".format(result.shape[0], result.shape[1]))
    return result