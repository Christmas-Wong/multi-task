'''
Author: your name
Date: 2021-10-22 10:47:24
LastEditTime: 2021-10-25 09:49:02
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /mdap/src/utils/util_generate_dataset.py
'''
import os
import pandas as pd
from utils.util_log import get_logger
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

logger = get_logger("Task DataSplit", None)

def debug_data(df: DataFrame) -> DataFrame:
    """Debug mode: Get part of Data

    Args:
        df (DataFrame): Source Data

    Returns:
        DataFrame: part Data
    """
    ratio = 0.3
    df_part_1 = df[df["是否无效"]==1].head(int(df.shape[0]*ratio))
    df_part_2 = df[df["是否无效"]==0]
    
    df_part_2, _ = train_test_split(df_part_2, train_size=ratio, stratify=df_part_2[['所属部门']])
    
    result = pd.concat([df_part_1, df_part_2])
    logger.info("Debug数据获取完成，总计[{}]行[{}]列".format(result.shape[0], result.shape[1]))
    return result

def generate_task_data(df: DataFrame, output_dir: str) -> None:
    """Generate and Save Task Data

    Args:
        df (DataFrame): Source  Data
        output_dir (str): output dir

    Returns:
        [type]: None
    """
    df_useful = generate_useful_task_data(df)
    df_department = generate_department_task_data(df)
    
    split_and_save(df_useful, "useful", output_dir)
    split_and_save(df_department, "department", output_dir)

def split_and_save(df: DataFrame, task_name: str, output_dir: str) -> None:
    """Split Data into Train & Test

    Args:
        df (DataFrame): Source data

    Returns:
        [type]: None
    """
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df[['label']])
    
    df_train.dropna(axis=0, how='any', inplace=True)
    df_test.dropna(axis=0, how='any', inplace=True)
    
    df_train.to_csv(os.path.join(output_dir, task_name+"_train.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, task_name+"_test.csv"), index=False)
    
    
def generate_useful_task_data(df: DataFrame) -> None:
    """Generate useful task data

    Args:
        df (DataFrame): Source Data

    Returns:
        [type]: DataFrame of Task Data
    """
    df = df[["text", "label_useful"]]
    df.dropna(how="any", axis=0, inplace=True)
    
    # 取全量的有效数据，无效数据等量抽样
    df_useful = df[df["label_useful"] == 0]
    df_unuseful = df[df["label_useful"] == 1].sample(n=df_useful.shape[0], replace=False, axis=0)

    result = shuffle(pd.concat([df_useful, df_unuseful]))
    result.columns=["text", "label"]
    logger.info("[有效无效标签]数据获取完成，总计[{}]行[{}]列".format(result.shape[0], result.shape[1]))
    
    return result

def generate_department_task_data(df: DataFrame):
    df = df[["text", "label_department"]]
    df.dropna(how="any", axis=0, inplace=True)
    
    result = shuffle(df)
    result.columns=["text", "label"]
    logger.info("[部门标签]数据获取完成，总计[{}]行[{}]列".format(result.shape[0], result.shape[1]))
    
    return result

