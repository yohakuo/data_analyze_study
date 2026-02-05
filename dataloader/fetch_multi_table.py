#!/usr/bin/env python3
"""
通用多表数据获取脚本

功能：
1. 从 ClickHouse 多个表中按时间范围获取数据
2. 支持字段选择
3. 统一时间粒度（重采样）
4. 合并多个数据源
5. 输出为 CSV 或 Excel

使用方法：
    # 使用配置文件
    python fetch_multi_table_data.py --config data_config.yaml -s "2024-01-01 00:00:00" -e "2024-01-31 23:59:59"
    
    # 使用命令行参数
    python fetch_multi_table_data.py \
        --temple 108 \
        --device 20A6 \
        --start "2024-01-01 00:00:00" \
        --end "2024-01-31 23:59:59" \
        --resample 10min
"""

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

# 导入你的 io 模块
try:
    from src import io_v2
except ImportError:
    print("错误: 无法导入 src.io_v2 模块")
    print("请确保脚本在正确的项目目录下运行")
    sys.exit(1)


# ====================================================================
# 数据查询和处理
# ====================================================================


def query_table_data(
    client,
    database: str,
    table: str,
    fields: List[str],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    temple_id: Optional[str] = None,
    device_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    从单个表查询数据

    Args:
        client: ClickHouse 客户端
        database: 数据库名
        table: 表名
        fields: 需要查询的字段列表
        start_time: 开始时间
        end_time: 结束时间
        temple_id: 窟号（可选）
        device_id: 设备ID（可选）

    Returns:
        DataFrame
    """
    # 构建 SELECT 字段
    # 确保 time 字段始终被查询
    if "time" not in fields:
        fields = ["time"] + fields

    # 转义字段名（处理包含特殊字符的字段）
    def escape_field_name(field: str) -> str:
        """为字段名添加反引号（如果需要）"""
        # 如果字段已经有反引号或双引号，直接返回
        if field.startswith("`") or field.startswith('"'):
            return field
        # 如果包含特殊字符（括号、空格等），添加反引号
        if any(char in field for char in ["(", ")", " ", "-", "/"]):
            return f"`{field}`"
        return field

    escaped_fields = [escape_field_name(f) for f in fields]
    select_fields = ", ".join(escaped_fields)

    # 构建 WHERE 条件
    where_conditions = [
        f"time >= '{start_time.strftime('%Y-%m-%d %H:%M:%S')}'",
        f"time <= '{end_time.strftime('%Y-%m-%d %H:%M:%S')}'",
    ]

    if temple_id:
        where_conditions.append(f"temple_id = '{temple_id}'")

    if device_id:
        where_conditions.append(f"device_id = '{device_id}'")

    where_clause = " AND ".join(where_conditions)

    # 构建查询
    query = f"""
    SELECT {select_fields}
    FROM {database}.{table}
    WHERE {where_clause}
    ORDER BY time
    """

    print(f"  查询 {database}.{table}: {len(fields)} 个字段")

    try:
        result = client.execute(query)

        # 转换为 DataFrame
        # 使用原始字段名（不含反引号）作为列名
        original_fields = [f.strip("`").strip('"') for f in fields]
        df = pd.DataFrame(result, columns=original_fields)

        # 转换时间列
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])

        print(f"    ✓ 获取 {len(df)} 条记录")

        return df

    except Exception as e:
        print(f"    ✗ 查询失败: {e}")
        return pd.DataFrame()


def resample_dataframe(
    df: pd.DataFrame, freq: str = "10min", agg_method: Dict[str, str] = None
) -> pd.DataFrame:
    """
    重采样 DataFrame 到指定时间粒度

    Args:
        df: 输入 DataFrame（必须包含 time 列）
        freq: 重采样频率（如 '10min', '1H', '1D'）
        agg_method: 聚合方法字典 {字段名: 方法}
                   方法可以是: 'mean', 'sum', 'max', 'min', 'first', 'last'

    Returns:
        重采样后的 DataFrame
    """
    if df.empty:
        return df

    if "time" not in df.columns:
        print("警告: DataFrame 缺少 time 列，无法重采样")
        return df

    # 设置时间索引
    df = df.set_index("time")

    # 如果指定了聚合方法，清理字段名（去除反引号）
    if agg_method is not None:
        # 清理聚合方法字典中的键（去除反引号和双引号）
        cleaned_agg_method = {}
        for key, value in agg_method.items():
            cleaned_key = key.strip("`").strip('"')
            # 只保留 DataFrame 中实际存在的列
            if cleaned_key in df.columns:
                cleaned_agg_method[cleaned_key] = value
        agg_method = cleaned_agg_method

    # 如果没有指定聚合方法，使用默认方法
    if not agg_method:
        agg_method = {}
        for col in df.columns:
            # 温度、湿度、压力等用平均值
            if any(
                keyword in col.lower()
                for keyword in ["temperature", "humidity", "pressure", "vapor", "avg"]
            ):
                agg_method[col] = "mean"
            # 降雨量、辐射量等用总和
            elif any(
                keyword in col.lower() for keyword in ["rainfall", "radiation", "total"]
            ):
                agg_method[col] = "sum"
            # ID 类字段用 first
            elif any(keyword in col.lower() for keyword in ["id", "uuid"]):
                agg_method[col] = "first"
            # 其他默认用平均值
            else:
                agg_method[col] = "mean"

    # 执行重采样
    df_resampled = df.resample(freq).agg(agg_method)

    # 重置索引
    df_resampled = df_resampled.reset_index()

    return df_resampled


def merge_dataframes(
    dataframes: List[pd.DataFrame], on: str = "time", how: str = "outer"
) -> pd.DataFrame:
    """
    合并多个 DataFrame

    Args:
        dataframes: DataFrame 列表
        on: 合并的键（通常是 'time'）
        how: 合并方式 ('inner', 'outer', 'left', 'right')

    Returns:
        合并后的 DataFrame
    """
    if not dataframes:
        return pd.DataFrame()

    # 过滤空 DataFrame
    dataframes = [df for df in dataframes if not df.empty]

    if not dataframes:
        return pd.DataFrame()

    if len(dataframes) == 1:
        return dataframes[0]

    # 逐个合并
    result = dataframes[0]
    for df in dataframes[1:]:
        result = pd.merge(result, df, on=on, how=how, suffixes=("", "_dup"))

        # 删除重复列（带 _dup 后缀的）
        dup_cols = [col for col in result.columns if col.endswith("_dup")]
        if dup_cols:
            result = result.drop(columns=dup_cols)

    # 按时间排序
    if on in result.columns:
        result = result.sort_values(on).reset_index(drop=True)

    return result


# ====================================================================
# 配置文件处理
# ====================================================================


def load_config(config_file: str) -> Dict:
    """从 YAML 文件加载配置"""
    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: Dict) -> bool:
    """验证配置文件格式"""
    required_keys = ["data_sources"]

    for key in required_keys:
        if key not in config:
            print(f"错误: 配置文件缺少必需的键: {key}")
            return False

    if not isinstance(config["data_sources"], list):
        print("错误: data_sources 必须是列表")
        return False

    for idx, source in enumerate(config["data_sources"]):
        required_source_keys = ["database", "table", "fields"]
        for key in required_source_keys:
            if key not in source:
                print(f"错误: data_sources[{idx}] 缺少必需的键: {key}")
                return False

    return True


# ====================================================================
# 主处理流程
# ====================================================================


def fetch_multi_table_data(
    client,
    data_sources: List[Dict],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    temple_id: Optional[str] = None,
    device_id: Optional[str] = None,
    resample_freq: Optional[str] = None,
) -> pd.DataFrame:
    """
    从多个表获取数据并合并

    Args:
        client: ClickHouse 客户端
        data_sources: 数据源配置列表
        start_time: 开始时间
        end_time: 结束时间
        temple_id: 窟号（可选）
        device_id: 设备ID（可选）
        resample_freq: 重采样频率（可选）

    Returns:
        合并后的 DataFrame
    """
    dataframes = []

    for idx, source in enumerate(data_sources, 1):
        database = source["database"]
        table = source["table"]
        fields = source["fields"]

        # 查询数据
        df = query_table_data(
            client=client,
            database=database,
            table=table,
            fields=fields,
            start_time=start_time,
            end_time=end_time,
            temple_id=temple_id if source.get("use_temple_id", True) else None,
            device_id=device_id if source.get("use_device_id", True) else None,
        )

        if df.empty:
            print("    警告: 未获取到数据，跳过")
            continue

        # 重采样（如果指定）
        if resample_freq:
            print(f"    重采样到 {resample_freq}...")
            agg_method = source.get("agg_method", None)
            df = resample_dataframe(df, freq=resample_freq, agg_method=agg_method)
            print(f"    ✓ 重采样后: {len(df)} 条记录")

        # 添加数据源标识（可选）
        if source.get("add_source_label", False):
            df["data_source"] = f"{database}.{table}"

        dataframes.append(df)

    # 合并数据
    if not dataframes:
        print("\n警告: 所有数据源都为空")
        return pd.DataFrame()

    print(f"\n合并 {len(dataframes)} 个数据源...")
    merged_df = merge_dataframes(dataframes, on="time", how="outer")

    print(f"✓ 合并完成: {len(merged_df)} 条记录, {len(merged_df.columns)} 个字段")

    return merged_df


# ====================================================================
# 命令行接口
# ====================================================================


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="通用多表数据获取脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  1. 使用配置文件:
     python fetch_multi_table_data.py \\
       --config data_config.yaml \\
       --start "2024-01-01 00:00:00" \\
       --end "2024-01-31 23:59:59" \\
       --temple 108 \\
       --device 20A6

  2. 快速模式（使用默认配置）:
     python fetch_multi_table_data.py \\
       --start "2024-01-01 00:00:00" \\
       --end "2024-01-31 23:59:59" \\
       --temple 108

  3. 自定义重采样频率:
     python fetch_multi_table_data.py \\
       --config data_config.yaml \\
       --start "2024-01-01 00:00:00" \\
       --end "2024-01-31 23:59:59" \\
       --resample 30min
""",
    )

    # 时间参数（必需）
    parser.add_argument(
        "-s", "--start", required=True, help="开始时间 (格式: YYYY-MM-DD HH:MM:SS)"
    )

    parser.add_argument(
        "-e", "--end", required=True, help="结束时间 (格式: YYYY-MM-DD HH:MM:SS)"
    )

    # 过滤参数（可选）
    parser.add_argument("-t", "--temple", help="窟号/Temple ID (例如: 108)")

    parser.add_argument("-d", "--device", help="设备ID (例如: 20A6)")

    # 配置文件
    parser.add_argument("--config", help="配置文件路径 (YAML 格式)")

    # 重采样参数
    parser.add_argument(
        "--resample",
        default="10min",
        help="重采样频率 (例如: 10min, 1H, 1D) (默认: 10min)",
    )

    parser.add_argument("--no-resample", action="store_true", help="不进行重采样")

    # 连接参数
    parser.add_argument(
        "--target",
        default="shared",
        choices=["local", "shared"],
        help="连接目标 (默认: shared)",
    )

    parser.add_argument(
        "--database", default="original_data", help="默认数据库 (默认: original_data)"
    )

    # 输出参数
    parser.add_argument(
        "-o", "--output", default="./data/output", help="输出目录 (默认: ./data/output)"
    )

    parser.add_argument(
        "-f",
        "--format",
        choices=["csv", "excel"],
        default="csv",
        help="输出格式 (默认: csv)",
    )

    parser.add_argument("--filename", help="自定义输出文件名（不含扩展名）")

    return parser.parse_args()


def build_default_config(args) -> Dict:
    """构建默认配置（当未提供配置文件时）"""
    return {
        "data_sources": [
            {
                "database": "original_data",
                "table": "sensor_temp_humidity",
                "fields": ["temple_id", "device_id", "time", "humidity", "temperature"],
                "use_temple_id": True,
                "use_device_id": True,
                "agg_method": {
                    "temple_id": "first",
                    "device_id": "first",
                    "humidity": "mean",
                    "temperature": "mean",
                },
            },
            {
                "database": "original_data",
                "table": "cave_entrance",
                "fields": ["time", "avg_vapor_pressure", "`total_rainfall(10min)`"],
                "use_temple_id": False,
                "use_device_id": False,
                "agg_method": {
                    "avg_vapor_pressure": "mean",
                    "total_rainfall(10min)": "sum",  # 注意：这里使用原始字段名（不含反引号）
                },
            },
        ]
    }


# ====================================================================
# 主函数
# ====================================================================


def main():
    """主函数"""
    args = parse_arguments()

    # 1. 解析时间
    try:
        start_time = datetime.datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print("错误: 时间格式错误，请使用 'YYYY-MM-DD HH:%M:%S'")
        sys.exit(1)

    print("\n查询参数:")
    print(f"  时间范围: {start_time} → {end_time}")
    if args.temple:
        print(f"  窟号: {args.temple}")
    if args.device:
        print(f"  设备ID: {args.device}")

    # 2. 加载配置
    if args.config:
        print(f"\n加载配置文件: {args.config}")
        try:
            config = load_config(args.config)
            if not validate_config(config):
                sys.exit(1)
        except Exception as e:
            print(f"错误: 加载配置文件失败: {e}")
            sys.exit(1)
    else:
        print("\n使用默认配置")
        config = build_default_config(args)

    print(f"  数据源数量: {len(config['data_sources'])}")

    # 3. 连接数据库
    print(f"\n连接到 {args.target} 数据库...")
    try:
        client = io_v2.get_clickhouse_client(target=args.target, database=args.database)
    except Exception as e:
        print(f"   连接失败: {e}")
        sys.exit(1)

    # 4. 获取数据
    try:
        resample_freq = None if args.no_resample else args.resample

        merged_df = fetch_multi_table_data(
            client=client,
            data_sources=config["data_sources"],
            start_time=start_time,
            end_time=end_time,
            temple_id=args.temple,
            device_id=args.device,
            resample_freq=resample_freq,
        )
    except Exception as e:
        print(f"\n错误: 数据查询失败: {e}")
        import traceback

        traceback.print_exc()
        client.disconnect()
        sys.exit(1)
    finally:
        client.disconnect()

    # 5. 保存结果
    if merged_df.empty:
        print("\n警告: 未获取到任何数据")
        sys.exit(0)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建文件名
    if args.filename:
        base_filename = args.filename
    else:
        time_str = start_time.strftime("%Y%m%d") + "_" + end_time.strftime("%Y%m%d")
        parts = []
        if args.temple:
            parts.append(f"temple_{args.temple}")
        if args.device:
            parts.append(f"device_{args.device}")
        parts.append(time_str)
        base_filename = "_".join(parts) if parts else f"data_{time_str}"

    # 保存文件
    print("\n保存结果...")
    print(f"  记录数: {len(merged_df)}")
    print(f"  字段数: {len(merged_df.columns)}")
    print(f"  字段: {', '.join(merged_df.columns)}")

    if args.format == "csv":
        output_file = output_dir / f"{base_filename}.csv"
        merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    else:
        output_file = output_dir / f"{base_filename}.xlsx"
        merged_df.to_excel(output_file, index=False, engine="openpyxl")

    print(f"\n 数据已保存至: {output_file}")


if __name__ == "__main__":
    main()
