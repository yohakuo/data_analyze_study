# FFT 分析
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import detrend


def analyze_with_fft(df: pd.DataFrame, field_name: str):
    """
    对 DataFrame 中的指定字段进行FFT分析，并找出最主要的周期
    """
    if df.empty:
        return None

    print(f"\n正在对字段 '{field_name}' 进行频谱分析 (FFT)...")

    # 1. 去除趋势
    print("正在去除数据的长期趋势...")
    detrended_values = detrend(df[field_name].values)

    # 2. FFT 计算
    N = len(detrended_values)
    T = (df.index[1] - df.index[0]).total_seconds()  # 自动计算采样间隔
    yf = fft(detrended_values)
    xf = fftfreq(N, T)[: N // 2]
    amplitude = 2.0 / N * np.abs(yf[0 : N // 2])

    # 3. 将频率转换为小时为单位的周期
    periods_in_hours = np.full_like(xf, np.inf)
    non_zero_indices = xf > 0
    periods_in_hours[non_zero_indices] = 1 / xf[non_zero_indices] / 3600

    # 4. 组合成 DataFrame 并返回
    spectrum_df = pd.DataFrame({"周期(小时)": periods_in_hours, "强度(幅度)": amplitude})
    return spectrum_df
