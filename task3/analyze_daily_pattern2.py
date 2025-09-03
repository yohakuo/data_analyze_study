import pandas as pd
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import seaborn as sns
from influxdb_client import InfluxDBClient
import datetime
import numpy as np # 导入 numpy 用于设置坐标轴刻度
from zoneinfo import ZoneInfo

# --- 1. InfluxDB 连接配置 (保持不变) ---
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "study2025"
INFLUXDB_ORG = "task3"
INFLUXDB_BUCKET = "cave45"
MEASUREMENT_NAME = 'adata'
FIELD_NAME = '空气湿度'

def get_all_data() -> pd.DataFrame:
    """从 InfluxDB 查询【全部】湿度数据。"""
    # ... (这个函数的所有代码都保持不变)
    with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: 0)
          |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT_NAME}")
          |> filter(fn: (r) => r["_field"] == "{FIELD_NAME}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        print("正在从 InfluxDB 查询所有原始数据...")
        df = client.query_api().query_data_frame(query=query, org=INFLUXDB_ORG)
        
        if not df.empty:
            df_cleaned = df[['_time', FIELD_NAME]].set_index('_time')
            df_cleaned.index = pd.to_datetime(df_cleaned.index)
            df_cleaned.index = df_cleaned.index.tz_convert('Asia/Shanghai')
            df_cleaned[FIELD_NAME] = pd.to_numeric(df_cleaned[FIELD_NAME], errors='coerce')
            df_cleaned = df_cleaned.dropna()
            print("数据提取并转换为本地时间完成！")
            return df_cleaned
        return pd.DataFrame()

# --- 2. 核心逻辑：按小时分析 ---
def analyze_hourly_pattern(df: pd.DataFrame):
    """按【每个小时】分析日变化模式"""
    if df.empty:
        print("数据为空，无法分析。")
        return
        
    print("\n正在按小时分析日变化规律...")
    
    # 1. ===== 关键修改点：直接按小时(0-23)进行分组 =====
    # df.index.hour 会直接返回每个时间戳对应的小时数 (0-23)
    hourly_avg_humidity = df.groupby(df.index.hour)[FIELD_NAME].mean()
    
    # --- 生成报告 ---
    print("\n" + "="*40)
    print("      *** 湿度小时级变化规律分析报告 ***")
    print("="*40)
    print("在整个数据周期内，每个小时的平均湿度如下：")
    print(hourly_avg_humidity.to_string())
    print("="*40)
    
    # --- 绘制图表 ---
    plt.figure(figsize=(15, 7))
    # 2. ===== 关键修改点：使用折线图来展示连续变化 =====
    sns.lineplot(x=hourly_avg_humidity.index, y=hourly_avg_humidity.values, marker='o', linestyle='-')
    plt.title('24小时平均湿度变化曲线', fontproperties="SimHei", fontsize=16)
    plt.xlabel('小时 (0-23)', fontproperties="SimHei", fontsize=12)
    plt.ylabel('平均湿度', fontproperties="SimHei", fontsize=12)
    plt.xticks(np.arange(0, 24, 1)) # 确保X轴显示所有24个小时的刻度
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 设置 Matplotlib 以正确显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 模块一：获取并准备数据
    all_data_df = get_all_data()
    # 模块二：按小时分析
    analyze_hourly_pattern(all_data_df)