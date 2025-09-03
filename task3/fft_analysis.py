import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('Agg') # 使用 'Agg' 引擎，专门用于保存文件，不弹窗
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
MEASUREMENT_NAME = 'adata'
FIELD_NAME = '空气湿度'

def get_all_data() -> pd.DataFrame:
    """从 InfluxDB 查询【全部】数据。"""
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
            df_cleaned[FIELD_NAME] = pd.to_numeric(df_cleaned[FIELD_NAME], errors='coerce')
            df_cleaned = df_cleaned.dropna()
            df_cleaned = df_cleaned.sort_index()
            # 以1分钟为频率重采样，填充缺失值，确保FFT的输入是均匀的
            df_cleaned = df_cleaned.resample('1T').ffill().bfill()
            print("数据提取和预处理完成！")
            return df_cleaned
        return pd.DataFrame()


    if df.empty:
        print("数据为空，无法进行分析。")
        return

    print("\n正在对全量数据进行频谱分析 (FFT)...")
    
    humidity_values = df['空气湿度'].values
    
    # ===== 关键修改点 2: 在 FFT 之前，对数据进行“去除趋势”处理 =====
    print("正在去除数据的长期趋势...")
    detrended_values = detrend(humidity_values)
    
    # --- 为了方便你理解，我们画一张对比图 ---
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, humidity_values, label='原始数据 (带趋势)')
    plt.plot(df.index, detrended_values, label='去除趋势后的数据 (仅含波动)', alpha=0.7)
    plt.title('去除趋势 (Detrending) 效果对比', fontproperties="SimHei", fontsize=16)
    plt.xlabel('时间', fontproperties="SimHei", fontsize=12)
    plt.ylabel('湿度', fontproperties="SimHei", fontsize=12)
    plt.legend(prop={"family":"SimHei"})
    plt.grid(True)
    plt.savefig('detrending_comparison.png')
    print("\n📈 去除趋势的效果对比图已保存为: detrending_comparison.png")
    # =========================================================

    N = len(detrended_values)
    T = 60.0

    # ===== 关键修改点 3: 使用“去除趋势后”的数据进行 FFT =====
    yf = fft(detrended_values)
    # (后续的 FFT 计算和绘图代码保持不变)
    xf = fftfreq(N, T)[:N//2]
    
    amplitude = 2.0/N * np.abs(yf[0:N//2])
    
    periods_in_hours = np.full_like(xf, np.inf)
    non_zero_indices = xf > 0
    periods_in_hours[non_zero_indices] = 1 / xf[non_zero_indices] / 3600
    
    spectrum_df = pd.DataFrame({
        '周期(小时)': periods_in_hours,
        '强度(幅度)': amplitude
    })
    
    spectrum_df = spectrum_df[ (spectrum_df['周期(小时)'] > 2) & (spectrum_df['周期(小时)'] < 2160) ]
    top_10_periods = spectrum_df.sort_values(by='强度(幅度)', ascending=False).head(10)
    
    print("\n" + "="*40)
    print("      *** 周期性分析报告 (已去除趋势) ***")
    print("="*40)
    print("在整个数据集中，信号最强的 Top 10 周期为：")
    print(top_10_periods.to_string(index=False))
    print("="*40)
    
    # (后续的频谱图绘制和保存代码保持不变)
    plt.figure(figsize=(15, 7))
    plt.plot(spectrum_df['周期(小时)'], spectrum_df['强度(幅度)'])
    plt.title('全量数据频谱分析图 (已去除趋势)', fontproperties="SimHei", fontsize=16)
    plt.xlabel('周期 (小时)', fontproperties="SimHei", fontsize=12)
    plt.ylabel('幅度', fontproperties="SimHei", fontsize=12)
    plt.grid(True)
    plt.xscale('log')
    
    for index, row in top_10_periods.head(5).iterrows():
        plt.axvline(x=row['周期(小时)'], color='r', linestyle='--', alpha=0.7)
        plt.text(row['周期(小时)'], row['强度(幅度)'], f' {row["周期(小时)"]:.1f}h', color='r')

    plot_filename = 'fft_spectrum_analysis_detrended.png'
    plt.savefig(plot_filename)
    print(f"\n📈 已去除趋势的频谱分析图已保存为文件: {plot_filename}")

    """对全量数据进行FFT分析，并找出最主要的周期"""
    if df.empty:
        print("数据为空，无法进行分析。")
        return

    print("\n正在对全量数据进行频谱分析 (FFT)...")
    
    humidity_values = df[FIELD_NAME].values
    N = len(humidity_values)
    T = 60.0  # 采样间隔60秒

    yf = fft(humidity_values)
    xf = fftfreq(N, T)[:N//2]
    
    amplitude = 2.0/N * np.abs(yf[0:N//2])
    
    periods_in_hours = np.full_like(xf, np.inf)
    non_zero_indices = xf > 0
    periods_in_hours[non_zero_indices] = 1 / xf[non_zero_indices] / 3600
    
    spectrum_df = pd.DataFrame({
        '周期(小时)': periods_in_hours,
        '强度(幅度)': amplitude
    })
    
    # 过滤掉噪音和超长周期。90天约为 90*24 = 2160小时
    spectrum_df = spectrum_df[ (spectrum_df['周期(小时)'] > 2) & (spectrum_df['周期(小时)'] < 2160) ]
    
    top_10_periods = spectrum_df.sort_values(by='强度(幅度)', ascending=False).head(10)
    
    print("\n" + "="*40)
    print("      *** FFT 周期性分析报告 ***")
    print("="*40)
    print("在整个数据集中，信号最强的 Top 10 周期为：")
    print(top_10_periods.to_string(index=False))
    print("="*40)
    
    plt.figure(figsize=(15, 7))
    plt.plot(spectrum_df['周期(小时)'], spectrum_df['强度(幅度)'])
    plt.title('全量数据频谱分析图 (FFT)', fontproperties="SimHei", fontsize=16)
    plt.xlabel('周期 (小时)', fontproperties="SimHei", fontsize=12)
    plt.ylabel('幅度', fontproperties="SimHei", fontsize=12)
    plt.grid(True)
    plt.xscale('log') # 使用对数坐标轴，可以更好地观察长周期的峰值
    
    for index, row in top_10_periods.head(5).iterrows():
        plt.axvline(x=row['周期(小时)'], color='r', linestyle='--', alpha=0.7)
        plt.text(row['周期(小时)'], row['强度(幅度)'], f' {row["周期(小时)"]:.1f}h', color='r')

    plot_filename = 'fft_spectrum_analysis.png'
    plt.savefig(plot_filename)
    print(f"\n📈 频谱分析图已保存为文件: {plot_filename}")

def analyze_with_fft(df: pd.DataFrame):
    if df.empty:
        print("数据为空，无法进行分析。")
        return

    print("\n正在对全量数据进行频谱分析 (FFT)...")
    
    # ===== 关键修改点：使用“移动平均法”进行去趋势 =====
    print("正在使用移动平均法计算并去除长期趋势...")
    
    # 1. 定义一个移动平均的窗口大小，例如7天
    # 窗口大小需要远大于我们想找的周期(24h)，以便只捕捉长期趋势
    # 7天 = 7 * 24 * 60 = 10080分钟
    window_size = 7 * 24 * 60
    
    # 2. 计算移动平均趋势线
    # center=True 让计算更平滑准确
    long_term_trend = df['空气湿度'].rolling(window=window_size, center=True).mean()
    
    # 3. 移动平均会在数据的开头和结尾产生空值(NaN)，我们需要填充它们
    long_term_trend = long_term_trend.ffill().bfill()
    
    # 4. 从原始数据中减去趋势
    detrended_values = df['空气湿度'].values - long_term_trend.values
    # =========================================================
    
    # --- 绘制对比图，这次你会看到一条曲线趋势 ---
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['空气湿度'].values, label='原始数据')
    plt.plot(df.index, long_term_trend.values, label=f'{window_size//1440}天移动平均趋势线', linestyle='--', color='red')
    plt.plot(df.index, detrended_values, label='去除趋势后的数据', alpha=0.7)
    plt.title('移动平均去趋势法效果对比', fontproperties="SimHei", fontsize=16)
    plt.xlabel('时间', fontproperties="SimHei", fontsize=12)
    plt.ylabel('湿度', fontproperties="SimHei", fontsize=12)
    plt.legend(prop={"family":"SimHei"})
    plt.grid(True)
    plt.savefig('detrending_ma_comparison.png')
    print("\n📈 新的去趋势效果对比图已保存为: detrending_ma_comparison.png")
    
    # ... (后续的 FFT 计算、生成报告、绘制频谱图的代码完全不变,
    #      只需确保它使用的是 detrended_values 变量) ...
    N = len(detrended_values)
    T = 60.0

    yf = fft(detrended_values)
    xf = fftfreq(N, T)[:N//2]
    
    amplitude = 2.0/N * np.abs(yf[0:N//2])
    
    periods_in_hours = np.full_like(xf, np.inf)
    non_zero_indices = xf > 0
    periods_in_hours[non_zero_indices] = 1 / xf[non_zero_indices] / 3600
    
    spectrum_df = pd.DataFrame({
        '周期(小时)': periods_in_hours,
        '强度(幅度)': amplitude
    })
    
    spectrum_df = spectrum_df[ (spectrum_df['周期(小时)'] > 2) & (spectrum_df['周期(小时)'] < 2160) ]
    top_10_periods = spectrum_df.sort_values(by='强度(幅度)', ascending=False).head(10)
    
    print("\n" + "="*40)
    print("      *** 周期性分析报告 (移动平均去趋势) ***")
    print("="*40)
    print("在整个数据集中，信号最强的 Top 10 周期为：")
    print(top_10_periods.to_string(index=False))
    print("="*40)
    
    plt.figure(figsize=(15, 7))
    plt.plot(spectrum_df['周期(小时)'], spectrum_df['强度(幅度)'])
    plt.title('全量数据频谱分析图 (移动平均去趋势)', fontproperties="SimHei", fontsize=16)
    plt.xlabel('周期 (小时)', fontproperties="SimHei", fontsize=12)
    plt.ylabel('幅度', fontproperties="SimHei", fontsize=12)
    plt.grid(True)
    plt.xscale('log')
    
    for index, row in top_10_periods.head(5).iterrows():
        plt.axvline(x=row['周期(小时)'], color='r', linestyle='--', alpha=0.7)
        plt.text(row['周期(小时)'], row['强度(幅度)'], f' {row["周期(小时)"]:.1f}h', color='r')

    plot_filename = 'fft_spectrum_analysis_detrended_ma.png'
    plt.savefig(plot_filename)
    print(f"\n📈 新的频谱分析图已保存为文件: {plot_filename}")

if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    full_data_df = get_all_data()
    analyze_with_fft(full_data_df)