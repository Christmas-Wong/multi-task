# -*- coding: utf-8 -*-
"""
@Time    : 2021/9/18 10:31 上午
@Author  : Fei Wang
@File    : Decorator
@Software: PyCharm
@Description: 
"""
from functools import wraps
import time
from utils.util_log import get_logger

logger = get_logger("Decorator", None)


def calculate_time(place=2):
    """
    计算函数运行时间装饰器

    :param place: 显示秒的位数，默认为2位
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            beg = time.time()
            f = func(*args, **kwargs)
            end = time.time()
            s = '{}：{:.%sf} s' % place
            logger.info(s.format("函数处理耗时", end - beg))
            return f

        return wrapper

    return decorator