'''
Author: your name
Date: 2021-10-21 22:52:58
LastEditTime: 2021-11-07 12:34:03
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /mdap/src/args/arg_common.py
'''
import os

MAX_LENGTH = 64

LOG_DIR = "/mnt/nfsshare/wangfei/project/server_multitask/log/"

DATA_PATH = "/mnt/nfsshare/wangfei/project/server_multitask/data/"

TASK_DATA_PATH = os.path.join(DATA_PATH, 'origin')

HELP_DATA = {
    # emoji中文对照表
    "json_emoji": os.path.join(DATA_PATH, "help/emoji2zh.json"),
    # 替换词表
    "json_replace_words": os.path.join(DATA_PATH, "help/replace_words_hll.json"),
    # 中文停用词表
    "txt_stopwords": os.path.join(DATA_PATH, "help/百度停用词列表.txt"),
    # 删除词表
    "txt_delete_words": os.path.join(DATA_PATH, "help/delete_words_hll.txt")
}

TASKS = ["location", "service"]


# Avarage Param of Comuputing Metric
AVERAGE_DICT = {
    "location": "macro",
    "service": "macro"
}


# Label Encoder File
TASK_LE = {
    "location": os.path.join(TASK_DATA_PATH, "0_encoder.pkl"),
    "service": os.path.join(TASK_DATA_PATH, "1_encoder.pkl")
}

COLUMUNS_DICT = {
    "location": ['input_ids', 'attention_mask', 'labels'],
    "service": ['input_ids', 'attention_mask', 'labels']
}

# Source Data Name
TASK_DATA_FILE = {
    "location":{
        "train":os.path.join(TASK_DATA_PATH, "location_train.csv"),
        "test":os.path.join(TASK_DATA_PATH, "location_test.csv")
    },
    "service":{
        "train":os.path.join(TASK_DATA_PATH, "service_train.csv"),
        "test":os.path.join(TASK_DATA_PATH, "service_test.csv")
    }
}

# Classification Num of Task
TASK_NUM_CLASS = {
    "location": 4,
    "service": 4
}
