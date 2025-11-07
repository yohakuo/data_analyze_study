# 调用 calculate_features 并处理格式塔
import pandas as pd

from src.features.statistica import FEATURE_CALCULATORS
from src.features.statistica import calculate_features as _calculate_single_field_features


def transform_device_data(
    device_df: pd.DataFrame, fields_to_process: list, features_to_calc: list, freq: str
) -> pd.DataFrame:
    """
    [T] 转换包装器：
    接收【单个设备】的数据，为【多个字段】计算【多个特征】。
    """
    if device_df.empty:
        return pd.DataFrame()

    all_features_list = []

    try:
        device_df_indexed = device_df.set_index("time")
        device_id = device_df["device_id"].iloc[0]
    except KeyError as e:
        print(f"❌ [T] 转换失败：数据中缺少 'time' 或 'device_id' 列。{e}")
        return pd.DataFrame()

    # 循环处理每个【字段】(e.g., 'humidity', 'temperature')
    for field_name in fields_to_process:
        # -----------------------------------------------------------------
        # 2. 调用 (Call)
        # -----------------------------------------------------------------
        # ‼️ 在这里，你【调用】了从 statistica.py 导入的函数
        # -----------------------------------------------------------------
        wide_df = _calculate_single_field_features(
            device_df_indexed, field_name=field_name, feature_list=features_to_calc, freq=freq
        )

        if wide_df.empty:
            continue

        # ... (后续的 Melt 和元数据添加) ...
        long_df = wide_df.reset_index().melt(
            id_vars=["time"], var_name="feature_key", value_name="value"
        )
        long_df["device_id"] = device_id
        long_df["field_name"] = field_name
        all_features_list.append(long_df)

    if not all_features_list:
        return pd.DataFrame()

    final_df = pd.concat(all_features_list)
    return final_df.dropna(subset=["value"])
