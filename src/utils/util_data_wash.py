'''
Author: your name
Date: 2021-10-21 22:33:53
LastEditTime: 2021-10-22 15:10:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /mdap/src/utils/util_data_wash.py
'''
from tqdm import tqdm
from entity.others.help_data import HelpData
import re
import pandas as pd
from pandas import DataFrame
import unicodedata
from sklearn.utils import shuffle
from typing import List
from utils.util_time import calculate_time
from utils.util_log import get_logger

logger = get_logger("Wash Data", None)

@calculate_time(2)
def df_wash(df: DataFrame, help_data: HelpData) -> DataFrame:
    """Wash DataFrame Data

    Args:
        df (DataFrame): Source Data
        help_data (HelpData): Help Data

    Returns:
        DataFrame: Clean DataFrame Data
    """

    text = []
    label_useful = []
    label_department = []
    
    for index, row in tqdm(iterable=df.iterrows(), total=df.shape[0]):
        sentence = str(row["描述"]).strip()
        sentence = text_process(sentence, help_data)
        if not sentence or len(sentence) < 1:
            continue
        
        text.append(sentence)
        label_useful.append(row["是否无效"])
        label_department.append(row["所属部门"])
    
    df_dict = {
        "text": text,
        "label_useful": label_useful,
        "label_department": label_department
    }
    
    result = pd.DataFrame(df_dict)
    logger.info("数据清洗完成[{}]行[{}]列".format(result.shape[0], result.shape[1]))

    return result
        
        

def text_process(text: str, help_data: HelpData) -> str:
    """Text Wash Process.
    1. Delete duplicate Punctuation
    2. text lower
    3. uncode transform(chinese symbol 2 english symbol)
    4. URL replace
    5. Delete words
    6. replace words
    7. emoji replace
    8. Delete English words

    Args:
        text (str): Text
        help_data (HelpData): Help Data for text wash

    Returns:
        str: clean text
    """
    if not text or len(text) < 1:
        return text
    
    # 去除重复的标点符号
    text = remove_duplication_punctuation(text)
    
    # 字母转小写
    text = text.lower()
    
    # 全角(中文)转换成半角(英文)
    text = unicodedata.normalize('NFKC', text)
    
    # 网址替换
    text = replace_url(text, "网址")
    
    # 删除
    if help_data.delete_words:
        text = delete_text_words(text, help_data.delete_words)
    
    # 替换（英文缩写->中文，错别字->正确字）
    if help_data.replace_words:
        text = replace_text_words(text, help_data.replace_words)
        
    # 替换emoji
    if help_data.emoji:
        text = replace_text_words(text, help_data.emoji)

    # 删除句子中的英文
    text = re.sub('[a-zA-Z]', "" , text)
    
    return text


def df_filter(df: DataFrame) -> DataFrame:
    """Filter Data from source dataFrame
    1. Get useful columns
    2. Drop None Data
    3. shuffle Data

    Args:
        df (DataFrame): source dataframe

    Returns:
        DataFrame: No useful DataFrame
    """
    df = df[["描述", "是否无效", "所属部门"]]
    
    df.dropna(subset=["描述", "是否无效"], axis=0, inplace=True)
    df_part_1 = df[df["是否无效"]==1]
    df_part_2 = df[df["是否无效"]==0]
    df_part_2.dropna(how="any", axis=0, inplace=True)
    
    result = pd.concat([df_part_1, df_part_2])
    result = shuffle(result)
    
    logger.info("数据过滤完成[{}]行[{}]列".format(result.shape[0], result.shape[1]))

    
    return result

def remove_duplication_punctuation(text: str) -> str:
    """
    Delete duplicate punctuations

    :param text: Source Text
    :return: New Text
    """
    left_square_brackets_pat = re.compile(r'\[+')
    right_square_brackets_pat = re.compile(r'\]+')
    punctuation = [',', '\\.', '\\!', '，', '。', '！', '、', '\?', '？']

    def replace(string, char):
        pattern = char + '{2,}'
        if char.startswith('\\'):
            char = char[1:]
        string = re.sub(pattern, char, string)
        return string

    text = left_square_brackets_pat.sub('', text)
    text = right_square_brackets_pat.sub('', text)
    for p in punctuation:
        text = replace(text, p)
    return text

def replace_url(text: str, target: str = "") -> str:
    """
    Replace URL

    :param text: Source Text
    :param target: Target Text
    :return: New Text
    """
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|%[0-9a-fA-F][0-9a-fA-F])+', target, text)

def replace_text_words(text: str, inverse_dict: dict) -> str:
    """
    Replace Some words

    :param text: Source Text
    :param inverse_dict: Word Pairs
    :return: New Text
    """
    for source, target in inverse_dict.items():
        text = text.replace(source, target)
    return text

def delete_text_words(text: str, words: List[str]) -> str:
    """
    Delete Words in WordList

    :param text: Source Text
    :param words: WordList which should be deleted
    :return: New Text
    """
    for word in words:
        text = text.replace(word, "")
    return text