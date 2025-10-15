# 先在Python 环境中安装 pyarrow。
# 然后根据下面的例子加载 .parquet 文件就行。

import pandas as pd

# (确保这个 .parquet 文件和脚本在同一个文件夹，或者提供完整路径)
DATA_FILENAME = "preprocessed_humidity_data.parquet"

try:
    df = pd.read_parquet(DATA_FILENAME)
    print("加载成功！")
except FileNotFoundError:
    print(f"错误：找不到文件 '{DATA_FILENAME}'。")
    exit()
