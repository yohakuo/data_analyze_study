#!/usr/bin/env python3
"""
风险特征计算主脚本

功能：
1. 从CSV文件读取温湿度和降雨数据
2. 调用 calculator.py 中封装好的函数计算三个风险特征：
   - 水汽扩散方向 (vapor_pressure_gradient)
   - 高湿暴露 (high_humidity_exposure)
   - 降雨强度 (rainfall_intensity)
3. 输出结果到CSV文件
4. 生成可视化图表（可选）

使用方法：
    python run_risk_features.py --config config_risk.yaml

或使用命令行参数：
    python run_risk_features.py --cave-inside data/cave_in.csv --cave-outside data/cave_out.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

from src.calculator import FeatureCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("risk_features.log"),
    ],
)
logger = logging.getLogger(__name__)


# ====================================================================
# 数据读取和验证
# ====================================================================


def load_csv_data(
    filepath: str | Path, time_column: str = "time", encoding: str = "utf-8"
) -> pd.DataFrame:
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    logger.info(f"正在读取文件: {filepath}")

    # 读取CSV
    df = pd.read_csv(filepath, encoding=encoding)

    # 检查时间列
    if time_column not in df.columns:
        raise ValueError(f"时间列 '{time_column}' 不存在。可用列: {list(df.columns)}")

    # 解析时间并设置为索引
    try:
        df[time_column] = pd.to_datetime(df[time_column])
        df.set_index(time_column, inplace=True)
        df.sort_index(inplace=True)  # 确保时间序列有序
    except Exception as e:
        raise ValueError(f"解析时间列失败: {e}")

    logger.info(f"✓ 读取成功: {len(df)} 行")
    logger.info(f"  时间范围: {df.index.min()} 至 {df.index.max()}")
    logger.info(f"  包含列: {', '.join(df.columns)}")

    return df


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"缺少必需的列: {missing}")


# ====================================================================
# 特征计算主流程
# ====================================================================


def calculate_all_features(config: Dict) -> Dict[str, pd.DataFrame]:
    """
    根据配置计算所有风险特征

    Args:
        config: 配置字典

    Returns:
        包含各特征结果的字典 {feature_name: DataFrame}
    """
    results = {}

    # 初始化特征计算器
    calc = FeatureCalculator()

    # ========================================
    # 1. 水汽扩散方向
    # ========================================
    if config.get("vapor_pressure_gradient", {}).get("enabled", False):
        logger.info("\n" + "=" * 70)
        logger.info("【特征 1】计算水汽扩散方向")
        logger.info("=" * 70)

        vp_cfg = config["vapor_pressure_gradient"]

        try:
            # 加载洞内数据
            df_in = load_csv_data(
                vp_cfg["cave_inside_file"],
                time_column=vp_cfg.get("time_column", "time"),
                encoding=vp_cfg.get("encoding", "utf-8"),
            )

            # 加载洞外数据
            df_out = load_csv_data(
                vp_cfg["cave_outside_file"],
                time_column=vp_cfg.get("time_column", "time"),
                encoding=vp_cfg.get("encoding", "utf-8"),
            )

            # 验证洞内列（需要温度和湿度）
            temp_in = vp_cfg.get("temp_col_in", "temperature")
            humidity_in = vp_cfg.get("humidity_col_in", "humidity")
            validate_columns(df_in, [temp_in, humidity_in])

            # 验证洞外列（需要水汽压）
            vapor_pressure_out = vp_cfg.get(
                "vapor_pressure_col_out", "avg_vapor_pressure"
            )
            validate_columns(df_out, [vapor_pressure_out])

            # 调用 calculator.py 的方法
            logger.info("调用 FeatureCalculator.calculate_vapor_pressure_gradient()")
            logger.info(f"  洞内: {temp_in} + {humidity_in} → 计算水汽压")
            logger.info(f"  洞外: {vapor_pressure_out} → 直接使用")

            result = calc.calculate_vapor_pressure_gradient(
                df_in=df_in,
                df_out=df_out,
                temp_col_in=temp_in,
                humidity_col_in=humidity_in,
                vapor_pressure_col_out=vapor_pressure_out,
            )

            if not result.empty:
                results["vapor_pressure_gradient"] = result
                logger.info(f" 成功计算 {len(result)} 个时间点的水汽扩散方向")

                # 统计扩散方向分布
                if "diffusion_direction" in result.columns:
                    dist = result["diffusion_direction"].value_counts()
                    logger.info("  扩散方向分布:")
                    for direction, count in dist.items():
                        logger.info(
                            f"    {direction}: {count} ({count / len(result):.1%})"
                        )
            else:
                logger.warning(" 水汽扩散方向计算结果为空")

        except Exception as e:
            logger.error(f" 计算水汽扩散方向时出错: {e}", exc_info=True)

    # ========================================
    # 2. 高湿暴露特征
    # ========================================
    if config.get("high_humidity_exposure", {}).get("enabled", False):
        logger.info("\n" + "=" * 70)
        logger.info("【特征 2】计算高湿暴露")
        logger.info("=" * 70)

        hh_cfg = config["high_humidity_exposure"]

        try:
            # 加载数据
            df = load_csv_data(
                hh_cfg["data_file"],
                time_column=hh_cfg.get("time_column", "time"),
                encoding=hh_cfg.get("encoding", "utf-8"),
            )

            # 验证列
            humidity_col = hh_cfg.get("humidity_column", "humidity")
            validate_columns(df, [humidity_col])

            # 调用 calculator.py 的方法
            threshold = hh_cfg.get("threshold", 62.0)
            freq = hh_cfg.get("freq", "D")

            logger.info(f"参数: 阈值={threshold}%, 统计周期={freq}")
            logger.info("调用 FeatureCalculator.calculate_high_humidity_exposure()")

            result = calc.calculate_high_humidity_exposure(
                df=df,
                humidity_col=humidity_col,
                threshold=threshold,
                freq=freq,
            )

            if not result.empty:
                results["high_humidity_exposure"] = result
                logger.info(f"✓ 成功计算 {len(result)} 个周期的高湿暴露特征")

                # 统计关键指标
                if "total_exposure_hours" in result.columns:
                    avg_hours = result["total_exposure_hours"].mean()
                    max_hours = result["total_exposure_hours"].max()
                    logger.info(f"  平均暴露时长: {avg_hours:.2f} 小时/周期")
                    logger.info(f"  最大暴露时长: {max_hours:.2f} 小时/周期")

                if "exposure_ratio" in result.columns:
                    high_risk = (result["exposure_ratio"] > 0.5).sum()
                    logger.info(f"  高风险周期数 (暴露>50%): {high_risk}")
            else:
                logger.warning("✗ 高湿暴露特征计算结果为空")

        except Exception as e:
            logger.error(f"✗ 计算高湿暴露特征时出错: {e}", exc_info=True)

    # ========================================
    # 3. 降雨强度特征
    # ========================================
    if config.get("rainfall_intensity", {}).get("enabled", False):
        logger.info("\n" + "=" * 70)
        logger.info("【特征 3】计算降雨强度")
        logger.info("=" * 70)

        rain_cfg = config["rainfall_intensity"]

        try:
            # 加载数据
            df = load_csv_data(
                rain_cfg["data_file"],
                time_column=rain_cfg.get("time_column", "time"),
                encoding=rain_cfg.get("encoding", "utf-8"),
            )

            # 验证列
            rainfall_col = rain_cfg.get("rainfall_column", "rainfall")
            validate_columns(df, [rainfall_col])

            # 调用 calculator.py 的方法
            window = rain_cfg.get("window", "10min")
            freq = rain_cfg.get("freq", "D")

            logger.info(f"参数: 窗口={window}, 统计周期={freq}")
            logger.info("调用 FeatureCalculator.calculate_rainfall_intensity()")

            result = calc.calculate_rainfall_intensity(
                df=df,
                rainfall_col=rainfall_col,
                window=window,
                freq=freq,
            )

            if not result.empty:
                results["rainfall_intensity"] = result
                logger.info(f"✓ 成功计算 {len(result)} 个周期的降雨强度特征")

                # 统计关键指标
                if "rainfall_total" in result.columns:
                    total_rain = result["rainfall_total"].sum()
                    avg_rain = result["rainfall_total"].mean()
                    logger.info(f"  总降雨量: {total_rain:.2f} mm")
                    logger.info(f"  平均降雨量: {avg_rain:.2f} mm/周期")

                if "rainfall_peak" in result.columns:
                    max_peak = result["rainfall_peak"].max()
                    logger.info(f"  最大降雨强度: {max_peak:.2f} mm/{window}")
            else:
                logger.warning("✗ 降雨强度特征计算结果为空")

        except Exception as e:
            logger.error(f"✗ 计算降雨强度特征时出错: {e}", exc_info=True)

    # ========================================
    # 4. 相关性分析（可选）
    # ========================================
    if (
        config.get("correlation_analysis", {}).get("enabled", False)
        and "high_humidity_exposure" in results
        and "rainfall_intensity" in results
    ):
        logger.info("\n" + "=" * 70)
        logger.info("【附加分析】湿度-降雨相关性")
        logger.info("=" * 70)

        try:
            corr_cfg = config["correlation_analysis"]
            humidity_col = corr_cfg.get("humidity_col", "total_exposure_hours")
            rainfall_col = corr_cfg.get("rainfall_col", "rainfall_peak")

            logger.info(
                "调用 FeatureCalculator.analyze_humidity_rainfall_correlation()"
            )

            correlation = calc.analyze_humidity_rainfall_correlation(
                humidity_exposure_df=results["high_humidity_exposure"],
                rainfall_df=results["rainfall_intensity"],
                humidity_col=humidity_col,
                rainfall_col=rainfall_col,
            )

            # 输出相关性结果
            logger.info("✓ 相关性分析结果:")
            logger.info(f"  皮尔逊相关系数: {correlation['correlation']:.4f}")
            logger.info(f"  显著性 p 值: {correlation['p_value']:.4f}")
            logger.info(f"  决定系数 R²: {correlation['r_squared']:.4f}")
            logger.info(
                f"  线性方程: y = {correlation['slope']:.4f}x + {correlation['intercept']:.4f}"
            )
            logger.info(f"  样本数: {correlation['n_samples']}")

            # 保存相关性结果
            corr_df = pd.DataFrame([correlation])
            results["correlation_analysis"] = corr_df

        except Exception as e:
            logger.error(f"✗ 相关性分析时出错: {e}", exc_info=True)

    return results


# ====================================================================
# 结果保存
# ====================================================================


def save_results(results: Dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """
    保存计算结果到CSV文件

    Args:
        results: 特征结果字典
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 70)
    logger.info("保存结果")
    logger.info("=" * 70)

    for feature_name, df in results.items():
        output_file = output_dir / f"{feature_name}.csv"
        df.to_csv(output_file, encoding="utf-8-sig")  # 使用 utf-8-sig 确保 Excel 兼容
        logger.info(f"✓ {output_file.name} ({len(df)} 行)")

    logger.info(f"\n所有结果已保存到: {output_dir.absolute()}")


def generate_summary(results: Dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """
    生成汇总报告

    Args:
        results: 特征结果字典
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    report_file = output_dir / "summary_report.txt"

    logger.info("\n" + "=" * 70)
    logger.info("生成汇总报告")
    logger.info("=" * 70)

    with open(report_file, "w", encoding="utf-8") as f:
        from datetime import datetime

        f.write("=" * 70 + "\n")
        f.write("风险特征计算汇总报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"特征数量: {len(results)}\n\n")

        for feature_name, df in results.items():
            f.write("\n" + "-" * 70 + "\n")
            f.write(f"特征: {feature_name}\n")
            f.write("-" * 70 + "\n")
            f.write(f"数据点数: {len(df)}\n")

            if not df.empty and hasattr(df.index, "min"):
                f.write(f"时间范围: {df.index.min()} 至 {df.index.max()}\n")

            f.write(f"包含列: {', '.join(df.columns)}\n\n")

            # 统计信息
            f.write("描述统计:\n")
            f.write(df.describe().to_string())
            f.write("\n")

            # 特定特征的额外分析
            if (
                feature_name == "vapor_pressure_gradient"
                and "diffusion_direction" in df.columns
            ):
                f.write("\n扩散方向分布:\n")
                dist = df["diffusion_direction"].value_counts()
                for direction, count in dist.items():
                    f.write(f"  {direction}: {count} ({count / len(df):.2%})\n")

            elif feature_name == "high_humidity_exposure":
                if "total_exposure_hours" in df.columns:
                    f.write(
                        f"\n平均暴露时长: {df['total_exposure_hours'].mean():.2f} 小时\n"
                    )
                    f.write(
                        f"最大暴露时长: {df['total_exposure_hours'].max():.2f} 小时\n"
                    )

                if "exposure_ratio" in df.columns:
                    high_risk = (df["exposure_ratio"] > 0.5).sum()
                    f.write(f"高风险周期数: {high_risk}\n")

            elif feature_name == "rainfall_intensity":
                if "rainfall_total" in df.columns:
                    f.write(f"\n总降雨量: {df['rainfall_total'].sum():.2f} mm\n")
                    f.write(f"平均降雨量: {df['rainfall_total'].mean():.2f} mm\n")

    logger.info(f"✓ 汇总报告已保存: {report_file.name}")


# ====================================================================
# 可视化（可选）
# ====================================================================


def generate_plots(results: Dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """
    生成可视化图表

    Args:
        results: 特征结果字典
        output_dir: 输出目录
    """
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        # 设置中文字体
        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    except ImportError:
        logger.warning("未安装 matplotlib，跳过可视化")
        return

    output_dir = Path(output_dir)
    logger.info("\n" + "=" * 70)
    logger.info("生成可视化图表")
    logger.info("=" * 70)

    # 水汽扩散方向
    if "vapor_pressure_gradient" in results:
        df = results["vapor_pressure_gradient"]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df["delta_VP"], linewidth=1)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.fill_between(
            df.index,
            0,
            df["delta_VP"],
            where=(df["delta_VP"] > 0),
            alpha=0.3,
            color="blue",
            label="向内扩散",
        )
        ax.fill_between(
            df.index,
            0,
            df["delta_VP"],
            where=(df["delta_VP"] < 0),
            alpha=0.3,
            color="orange",
            label="向外扩散",
        )
        ax.set_xlabel("时间")
        ax.set_ylabel("水汽压差 (hPa)")
        ax.set_title("水汽扩散方向")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_file = output_dir / "vapor_pressure_gradient.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"✓ {plot_file.name}")

    # 高湿暴露
    if "high_humidity_exposure" in results:
        df = results["high_humidity_exposure"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 子图1：暴露时长
        axes[0].bar(df.index, df["total_exposure_hours"], alpha=0.7, label="总暴露时长")
        axes[0].plot(
            df.index,
            df["max_continuous_hours"],
            color="red",
            marker="o",
            markersize=3,
            label="最长连续暴露",
        )
        axes[0].set_ylabel("时长 (小时)")
        axes[0].set_title("高湿暴露时长")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 子图2：暴露比例
        axes[1].bar(df.index, df["exposure_ratio"], alpha=0.7, color="coral")
        axes[1].axhline(
            y=0.5, color="red", linestyle="--", alpha=0.5, label="50%基准线"
        )
        axes[1].set_xlabel("时间")
        axes[1].set_ylabel("暴露比例")
        axes[1].set_title("高湿暴露比例")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = output_dir / "high_humidity_exposure.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"✓ {plot_file.name}")

    # 降雨强度
    if "rainfall_intensity" in results:
        df = results["rainfall_intensity"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 子图1：降雨强度
        axes[0].plot(
            df.index,
            df["rainfall_peak"],
            color="blue",
            marker="o",
            markersize=3,
            label="峰值强度",
        )
        axes[0].fill_between(df.index, 0, df["rainfall_peak"], alpha=0.3)
        axes[0].set_ylabel("强度 (mm/10min)")
        axes[0].set_title("降雨强度峰值")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 子图2：降雨总量
        axes[1].bar(df.index, df["rainfall_total"], alpha=0.7, color="steelblue")
        axes[1].set_xlabel("时间")
        axes[1].set_ylabel("降雨量 (mm)")
        axes[1].set_title("日降雨量")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = output_dir / "rainfall_intensity.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"✓ {plot_file.name}")


# ====================================================================
# 命令行接口
# ====================================================================


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="风险特征计算工具 - 调用 calculator.py 封装的函数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  1. 使用配置文件（推荐）:
     python main_risk_features.py --config config.yaml
  
  2. 快速计算水汽扩散方向:
     python main_risk_features.py \\
       --cave-inside data/cave_in.csv \\
       --cave-outside data/cave_out.csv
  
  3. 快速计算高湿暴露:
     python main_risk_features.py \\
       --humidity data/humidity.csv \\
       --threshold 62
  
  4. 快速计算降雨强度:
     python main_risk_features.py \\
       --rainfall data/rainfall.csv
  
  5. 同时计算多个特征:
     python main_risk_features.py \\
       --cave-inside data/cave_in.csv \\
       --cave-outside data/cave_out.csv \\
       --humidity data/humidity.csv \\
       --rainfall data/rainfall.csv \\
       --output results/
""",
    )

    parser.add_argument("--config", type=str, help="配置文件路径 (YAML格式)")
    parser.add_argument("--cave-inside", type=str, help="洞内数据CSV文件")
    parser.add_argument("--cave-outside", type=str, help="洞外数据CSV文件")
    parser.add_argument("--humidity", type=str, help="湿度数据CSV文件")
    parser.add_argument("--rainfall", type=str, help="降雨数据CSV文件")
    parser.add_argument(
        "--threshold", type=float, default=62.0, help="高湿阈值 (默认: 62%%)"
    )
    parser.add_argument(
        "--output", type=str, default="output", help="输出目录 (默认: output)"
    )
    parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    parser.add_argument("--no-report", action="store_true", help="不生成汇总报告")

    return parser.parse_args()


def build_config_from_args(args) -> Dict:
    """从命令行参数构建配置"""
    config = {}

    # 水汽扩散方向
    if args.cave_inside and args.cave_outside:
        config["vapor_pressure_gradient"] = {
            "enabled": True,
            "cave_inside_file": args.cave_inside,
            "cave_outside_file": args.cave_outside,
        }

    # 高湿暴露
    if args.humidity:
        config["high_humidity_exposure"] = {
            "enabled": True,
            "data_file": args.humidity,
            "threshold": args.threshold,
        }

    # 降雨强度
    if args.rainfall:
        config["rainfall_intensity"] = {
            "enabled": True,
            "data_file": args.rainfall,
        }

    return config


def load_config_file(config_path: str | Path) -> Dict:
    """从YAML文件加载配置"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    try:
        if args.config:
            logger.info(f"从配置文件加载: {args.config}")
            config = load_config_file(args.config)
        else:
            logger.info("从命令行参数构建配置")
            config = build_config_from_args(args)

        if not config:
            logger.error("未提供有效的配置或参数")
            logger.info("使用 --help 查看帮助信息")
            sys.exit(1)

    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        sys.exit(1)

    # 计算特征
    try:
        results = calculate_all_features(config)

        if not results:
            logger.warning("未计算任何特征")
            sys.exit(1)

    except Exception as e:
        logger.error(f"计算特征时出错: {e}", exc_info=True)
        sys.exit(1)

    # 保存结果
    try:
        save_results(results, args.output)
    except Exception as e:
        logger.error(f"保存结果时出错: {e}", exc_info=True)

    # 生成报告
    if not args.no_report:
        try:
            generate_summary(results, args.output)
        except Exception as e:
            logger.error(f"生成报告时出错: {e}", exc_info=True)

    # 生成图表
    if not args.no_plot:
        try:
            generate_plots(results, args.output)
        except Exception as e:
            logger.warning(f"生成图表时出错: {e}")

    logger.info(f"结果已保存到: {Path(args.output).absolute()}")


if __name__ == "__main__":
    main()
