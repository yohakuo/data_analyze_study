import pandas as pd

from src.calculator import FeatureCalculator

calc = FeatureCalculator()


def _standardize_output(wide_df: pd.DataFrame, cycle_label: str = None) -> pd.DataFrame:
    """
    标准化输出：转长表 + 注入 stats_cycle
    """
    if wide_df.empty:
        return pd.DataFrame()

    # 基础转换 (Reset Index & Melt)
    df = wide_df.reset_index()
    time_col = df.columns[0]
    melted = df.melt(id_vars=[time_col], var_name="feature_key", value_name="value")
    melted.rename(columns={time_col: "time"}, inplace=True)

    # 注入周期标签
    # 如果没传 label，就默认填个 'unknown' 防止空值报错
    melted["stats_cycle"] = cycle_label if cycle_label else "unknown"

    return melted


def process_statistical(df: pd.DataFrame, field: str, params: dict) -> pd.DataFrame:
    """
    处理器：统计特征
    特点：依赖 freq (resample)，输出连续时间序列
    """
    freq = params.get("freq", "h")
    cycle = params.get("cycle_label", freq)
    res = calc.calculate_statistical_features(
        df, field, params.get("metrics", []), freq
    )
    return _standardize_output(res, cycle_label=cycle)


def process_volatility(df: pd.DataFrame, field: str, params: dict) -> pd.DataFrame:
    # 如果配置里没写 freq，默认就按 'D' (天) 算
    freq = params.get("freq", "D")
    metrics = params.get("metrics", None)
    cycle = params.get("cycle_label", "1d")
    # 把配置里的参数透传进去
    res = calc.calculate_volatility_features(
        df,
        field_name=field,
        feature_list=metrics,
        freq=freq,
    )
    return _standardize_output(res, cycle_label=cycle)


def process_spectral(df: pd.DataFrame, field: str, params: dict) -> pd.DataFrame:
    """
    处理器：频谱/周期特征 (对应你特征表的第3类)
    特点：不依赖 freq 重采样，而是对整个输入窗口做 FFT
    """
    cycle = params.get("cycle_label", "batch")
    # 假设输入的是过去30天的数据
    spectrum_df = calc.analyze_spectral(df, field)

    if spectrum_df is None or spectrum_df.empty:
        return pd.DataFrame()

    # 频谱返回的是 [周期, 强度]，我们需要把它变成特征行
    # 策略：取强度最大的 Top N 周期作为特征
    top_n = params.get("top_n", 1)
    top_periods = spectrum_df.nlargest(top_n, "强度(幅度)")

    features = {}
    for i, (_, row) in enumerate(top_periods.iterrows()):
        p_val = row["周期(小时)"]
        s_val = row["强度(幅度)"]
        # 命名特征：周期_1, 强度_1, 周期_2...
        features[f"main_period_{i + 1}_hours"] = p_val
        features[f"main_period_{i + 1}_strength"] = s_val

    # 频谱特征的时间点通常标记为这段数据的“结束时间”或“开始时间”
    # 这里我们取数据的最后时间点作为特征的时间戳
    timestamp = df.index.max()

    # 构造单行 DataFrame
    df = pd.DataFrame([features], index=[timestamp])
    return _standardize_output(df, cycle_label=cycle)


def process_vapor_pressure_gradient(
    df_in: pd.DataFrame, df_out: pd.DataFrame, params: dict
) -> pd.DataFrame:
    """
    处理器：水汽扩散方向
    Args:
        df_in: 洞内数据 DataFrame
        df_out: 洞外数据 DataFrame
        params: 参数字典，可包含：
            - temp_col_in: 洞内温度列名 (默认 "temperature")
            - humidity_col_in: 洞内湿度列名 (默认 "humidity")
            - temp_col_out: 洞外温度列名 (默认 "temperature")
            - humidity_col_out: 洞外湿度列名 (默认 "humidity")
            - cycle_label: 周期标签 (默认 "raw")
    Returns:
        标准化的长表格式 DataFrame
    """
    temp_col_in = params.get("temp_col_in", "temperature")
    humidity_col_in = params.get("humidity_col_in", "humidity")
    temp_col_out = params.get("temp_col_out", "temperature")
    humidity_col_out = params.get("humidity_col_out", "humidity")
    cycle = params.get("cycle_label", "raw")

    # 计算水汽压梯度
    result_df = calc.calculate_vapor_pressure_gradient(
        df_in=df_in,
        df_out=df_out,
        temp_col_in=temp_col_in,
        humidity_col_in=humidity_col_in,
        temp_col_out=temp_col_out,
        humidity_col_out=humidity_col_out,
    )

    if result_df.empty:
        return pd.DataFrame()

    # 标准化输出
    return _standardize_output(result_df, cycle_label=cycle)


def process_high_humidity_exposure(
    df: pd.DataFrame, field: str, params: dict
) -> pd.DataFrame:
    """
    处理器：高湿暴露特征
    Args:
        df: 包含湿度数据的 DataFrame
        field: 湿度字段名 (通常是 "humidity")
        params: 参数字典，可包含：
            - threshold: 高湿阈值 (默认 62.0%)
            - freq: 统计周期 (默认 "D")
            - cycle_label: 周期标签
    Returns:
        标准化的长表格式 DataFrame
    """
    threshold = params.get("threshold", 62.0)
    freq = params.get("freq", "D")
    cycle = params.get("cycle_label", freq)

    # 计算高湿暴露
    result_df = calc.calculate_high_humidity_exposure(
        df=df,
        humidity_col=field,
        threshold=threshold,
        freq=freq,
    )

    if result_df.empty:
        return pd.DataFrame()

    # 标准化输出
    return _standardize_output(result_df, cycle_label=cycle)


def process_rainfall_intensity(
    df: pd.DataFrame, field: str, params: dict
) -> pd.DataFrame:
    """
    处理器：降雨强度特征
    Args:
        df: 包含降雨数据的 DataFrame
        field: 降雨量字段名 (如 "rainfall")
        params: 参数字典，可包含：
            - window: 滑动窗口大小 (默认 "10min")
            - freq: 统计周期 (默认 "D")
            - cycle_label: 周期标签
    Returns:
        标准化的长表格式 DataFrame
    """
    window = params.get("window", "10min")
    freq = params.get("freq", "D")
    cycle = params.get("cycle_label", freq)

    # 计算降雨强度
    result_df = calc.calculate_rainfall_intensity(
        df=df,
        rainfall_col=field,
        window=window,
        freq=freq,
    )

    if result_df.empty:
        return pd.DataFrame()

    # 标准化输出
    return _standardize_output(result_df, cycle_label=cycle)
