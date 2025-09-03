import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import seaborn as sns
from influxdb_client import InfluxDBClient
import datetime
from zoneinfo import ZoneInfo

# --- 1. InfluxDB 连接配置 ---
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "study2025"
INFLUXDB_ORG = "task3"
INFLUXDB_BUCKET = "cave45"

# --- 2. 查询参数配置 ---
# 为了让周期性更明显，我们提取一个月的数据作为样本
START_TIME_LOCAL = datetime.datetime(2021, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
STOP_TIME_LOCAL = datetime.datetime(2021, 2, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
MEASUREMENT_NAME = 'adata'
FIELD_NAME = '空气湿度'

def get_data(start_time, stop_time) -> pd.DataFrame:
    """从 InfluxDB 查询指定时间范围的数据。"""
    start_utc_str = start_time.astimezone(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
    stop_utc_str = stop_time.astimezone(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
    
    with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: {start_utc_str}, stop: {stop_utc_str})
          |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT_NAME}")
          |> filter(fn: (r) => r["_field"] == "{FIELD_NAME}")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> drop(columns: ["_start", "_stop", "_measurement"])
        '''
        print(f"正在从 InfluxDB 查询从 {start_time} 到 {stop_time} 的数据...")
        df = client.query_api().query_data_frame(query=query, org=INFLUXDB_ORG)
        
        if not df.empty:
            df = df.set_index('_time')
            df.index = pd.to_datetime(df.index)
            df[FIELD_NAME] = pd.to_numeric(df[FIELD_NAME], errors='coerce')
            df = df.dropna()
            # 确保数据是按时间排序的
            df = df.sort_index()
            # 对数据进行重采样，确保每分钟都有一个点，缺失的值用前后值填充
            df = df.resample('1T').ffill().bfill() 
            print("数据提取和预处理完成！")
            return df
        return pd.DataFrame()

def analyze_periodicity(df: pd.DataFrame):
    """对给定的 DataFrame 进行周期性分析"""
    if df.empty:
        print("数据为空，无法进行分析。")
        return

    # --- 分组统计与可视化  ---
    print("\n正在执行分组统计与可视化...")
    hourly_avg = df.groupby(df.index.hour)[FIELD_NAME].mean()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, marker='o')
    plt.title('24小时平均湿度变化 (直观法)', fontproperties="SimHei", fontsize=16)
    plt.xlabel('小时 (0-23)', fontproperties="SimHei", fontsize=12)
    plt.ylabel('平均湿度', fontproperties="SimHei", fontsize=12)
    plt.grid(True)
    plt.xticks(np.arange(0, 24, 1))
    plt.show()

    # --- 频谱分析 (FFT) ---
    print("\n正在执行频谱分析 (FFT)...")
    
    humidity_values = df[FIELD_NAME].values
    N = len(humidity_values)
    T = 60.0

    yf = fft(humidity_values)
    # a. 先获取所有正频率（不包括0频率）
    xf = fftfreq(N, T)[:N//2]
    # b. 创建一个足够大的数组来存放周期，默认值为无穷大
    periods_in_hours = np.full_like(xf, np.inf)
    # c. 只对大于0的频率计算周期，避免 1/0 的情况
    non_zero_indices = xf > 0
    periods_in_hours[non_zero_indices] = 1 / xf[non_zero_indices] / 3600

    amplitude = 2.0/N * np.abs(yf[0:N//2])

    plt.figure(figsize=(12, 6))
    valid_indices = (periods_in_hours > 2) & (periods_in_hours < 48)
    plt.plot(periods_in_hours[valid_indices], amplitude[valid_indices])
    plt.title('频谱分析图 (FFT)', fontproperties="SimHei", fontsize=16)
    plt.xlabel('周期 (小时)', fontproperties="SimHei", fontsize=12)
    plt.ylabel('幅度', fontproperties="SimHei", fontsize=12)
    plt.grid(True)
    
    strongest_period_index = np.argmax(amplitude[valid_indices])
    # 需要在 valid_indices 的基础上再进行索引
    subset_periods = periods_in_hours[valid_indices]
    subset_amplitudes = amplitude[valid_indices]
    strongest_period = subset_periods[strongest_period_index]
    
    plt.axvline(x=strongest_period, color='r', linestyle='--', label=f'最强周期: {strongest_period:.2f} 小时')
    plt.legend(prop={"family":"SimHei"})
    plt.show()

if __name__ == "__main__":
    # 为了能正确显示中文，设置 Matplotlib
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    raw_data_df = get_data(START_TIME_LOCAL, STOP_TIME_LOCAL)
    analyze_periodicity(raw_data_df)