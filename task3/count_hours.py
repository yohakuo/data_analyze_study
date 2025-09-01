import pandas as pd

# 使用我们从日志里读到的开始和结束时间
start_time = pd.Timestamp("2021-01-01 00:00:00+00:00")
end_time = pd.Timestamp("2021-11-02 07:53:00+00:00") # 请根据你的实际日志修改结束时间

# 计算两个时间点之间的总秒数
total_seconds = (end_time - start_time).total_seconds()

# 将总秒数转换为小时数
total_hours = total_seconds / 3600

# 因为resample是按小时的“格子”算的，所以我们需要向上取整
import math
number_of_hourly_features = math.ceil(total_hours)

print(f"数据总时长约为 {total_hours:.2f} 小时")
print(f"因此，应该生成的小时级特征数量为: {number_of_hourly_features}")