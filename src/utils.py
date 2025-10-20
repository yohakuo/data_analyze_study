import datetime
from functools import wraps
import random
import time

from influxdb_client import InfluxDBClient

from src import config  # 导入中央配置


def timing_decorator(func):
    """
    打印函数执行时间。
    """

    # @wraps(func) 能帮助保留原函数的名称等信息
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()

        # *args, **kwargs 是魔法语法，它能让“助理”服务于任何类型的函数
        # 无论有多少个参数
        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"✅  {func.__name__} 执行完毕，耗时: {duration:.2f} 秒")

        return result

    return wrapper
