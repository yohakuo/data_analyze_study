import pandas as pd
import matplotlib
matplotlib.use('TkAgg') # 确保绘图窗口可以弹出
import matplotlib.pyplot as plt
import seaborn as sns
from influxdb_client import InfluxDBClient
import datetime
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
            # 将UTC时间转换为东八区本地时间，方便我们按本地小时划分
            df_cleaned.index = df_cleaned.index.tz_convert('Asia/Shanghai')
            df_cleaned[FIELD_NAME] = pd.to_numeric(df_cleaned[FIELD_NAME], errors='coerce')
            df_cleaned = df_cleaned.dropna()
            print("数据提取并转换为本地时间完成！")
            return df_cleaned
        return pd.DataFrame()

# --- 2. 核心逻辑：定义时段并进行分析 ---
def get_time_period_label(hour):
    """根据你定义的小时，返回对应的时段标签"""
    if 1 <= hour <= 4:
        return "凌晨 (01-04)"
    elif 5 <= hour <= 7:
        return "早晨 (05-07)"
    elif 8 <= hour <= 10:
        return "上午 (08-10)"
    elif 11 <= hour <= 12:
        return "中午 (11-12)"
    elif 13 <= hour <= 16:
        return "下午 (13-16)"
    elif 17 <= hour <= 18:
        return "傍晚 (17-18)"
    elif 19 <= hour <= 22:
        return "晚上 (19-22)"
    # 子夜包含 23 点和 0 点
    elif hour == 23 or hour == 0:
        return "子夜 (23-00)"
    else:
        return "未知"

def analyze_diurnal_pattern(df: pd.DataFrame):
    """按自定义时段分析日变化模式"""
    if df.empty:
        print("数据为空，无法分析。")
        return
        
    print("\n正在按自定义时段分析日变化规律...")
    
    # 1. 为每一行数据打上你定义的时段标签
    df['时段'] = df.index.hour.map(get_time_period_label)
    
    # 2. 按“时段”标签进行分组，并计算每个时段的平均湿度
    period_avg_humidity = df.groupby('时段')[FIELD_NAME].mean()
    
    # 3. 按一天的自然顺序对结果进行排序
    day_order = [
        "凌晨 (01-04)", "早晨 (05-07)", "上午 (08-10)", "中午 (11-12)",
        "下午 (13-16)", "傍晚 (17-18)", "晚上 (19-22)", "子夜 (23-00)"
    ]
    period_avg_humidity = period_avg_humidity.reindex(day_order)
    
    # --- 生成报告 ---
    print("\n" + "="*40)
    print("      *** 湿度日变化规律分析报告 ***")
    print("="*40)
    print("在整个数据周期内，各时段的平均湿度如下：")
    print(period_avg_humidity.to_string())
    print("="*40)
    
    # --- 绘制图表 ---
    plt.figure(figsize=(15, 8))
    # 使用条形图更适合展示分类型的数据
    sns.barplot(x=period_avg_humidity.index, y=period_avg_humidity.values, palette="viridis")
    plt.title('各时段平均湿度分布', fontproperties="SimHei", fontsize=16)
    plt.xlabel('一天中的时段', fontproperties="SimHei", fontsize=12)
    plt.ylabel('平均湿度', fontproperties="SimHei", fontsize=12)
    plt.xticks(rotation=45) # 让X轴的标签旋转一下，避免重叠
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() # 自动调整布局
    plt.show()

if __name__ == "__main__":
    # 设置 Matplotlib 以正确显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 模块一：获取并准备数据
    all_data_df = get_all_data()
    # 模块二：按自定义时段分析
    analyze_diurnal_pattern(all_data_df)