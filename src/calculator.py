"""
Feature Calculator Module

This module provides a unified interface for calculating statistical, spectral,
and volatility features from time series data. The FeatureCalculator class
encapsulates all feature calculation methods and can be exposed via MCP.

Updated: Added risk assessment features including vapor pressure analysis,
high humidity exposure, and rainfall intensity metrics.
"""

import logging
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import detrend

# Configure module logger
logger = logging.getLogger(__name__)


class FeatureCalculator:
    """
    Unified feature calculator for time series data.

    This class provides methods for computing statistical features,
    spectral analysis (FFT), and volatility metrics from pandas DataFrames
    containing time series data.

    Attributes:
        _feature_calculators: Dictionary mapping Chinese feature names to
                            calculation functions or pandas aggregation strings.
    """

    def __init__(self):
        """
        Initialize the feature calculator.

        Sets up the internal dictionary mapping feature names to their
        corresponding calculation functions.
        """
        self._feature_calculators: dict[str, str | Callable] = {
            "均值": self._calculate_mean,
            "中位数": self._calculate_median,
            "最大值": self._calculate_max,
            "最小值": self._calculate_min,
            "Q1": self._calculate_q1,
            "Q3": self._calculate_q3,
            "P10": self._calculate_p10,
            "超过Q3占时比": self._calculate_percent_above_q3,
            "极差": self._calculate_range,
            "标准差": self._calculate_std,
            "平均绝对偏差_均值": self._mad_from_mean,
            "平均绝对偏差_中位数": self._mad_from_median,
            "变异系数": self._calculate_coefficient_of_variation,
            "一阶自相关": self._autocorr_lag1,
        }

    def calculate_statistical_features(
        self,
        df: pd.DataFrame,
        field_name: str,
        feature_list: list[str],
        freq: str = "h",
    ) -> pd.DataFrame:
        """
        Calculate statistical features from time series data.

        This method resamples the time series at the specified frequency and
        computes the requested statistical features. It automatically calculates
        range (极差) when both max and min are requested, and computes the rate
        of change for the range.

        Args:
            df: DataFrame with DatetimeIndex containing time series data
            field_name: Name of the column to calculate features for
            feature_list: List of feature names to calculate. Valid options:
                         均值, 中位数, 最大值, 最小值, 标准差, Q1, Q3, P10, 超过Q3占时比,
                         平均绝对偏差_均值, 平均绝对偏差_中位数, 变异系数, 一阶自相关
            freq: Resampling frequency. Options:
                  'h' - hourly, 'D' - daily, 'W' - weekly, 'M' - monthly

        Returns:
            DataFrame with calculated features indexed by time. Returns empty
            DataFrame if input is invalid.

        Raises:
            ValueError: If df is empty or field_name not in columns

        Example:
            >>> calculator = FeatureCalculator()
            >>> features = calculator.calculate_statistical_features(
            ...     df, "temperature", ["均值", "最大值", "最小值"], freq="D"
            ... )
        """
        # Input validation
        if df.empty:
            logger.error(
                "Empty DataFrame provided to calculate_statistical_features",
                extra={"field_name": field_name, "freq": freq},
            )
            return pd.DataFrame()

        if field_name not in df.columns:
            logger.error(
                f"Field '{field_name}' not found in DataFrame",
                extra={
                    "field_name": field_name,
                    "available_columns": list(df.columns),
                    "freq": freq,
                },
            )
            return pd.DataFrame()

        # Filter requested features to only those available
        agg_functions = {
            feature: self._feature_calculators[feature]
            for feature in feature_list
            if feature in self._feature_calculators
        }

        if not agg_functions:
            logger.error(
                "No valid features found in feature_list",
                extra={
                    "field_name": field_name,
                    "requested_features": feature_list,
                    "available_features": list(self._feature_calculators.keys()),
                },
            )
            return pd.DataFrame()

        logger.info(
            f"Calculating statistical features for field '{field_name}'",
            extra={
                "field_name": field_name,
                "features": list(agg_functions.keys()),
                "freq": freq,
                "data_shape": df.shape,
            },
        )

        # Resample and calculate all features at once using .agg()
        resampled_stats = df[field_name].resample(freq).agg(**agg_functions)

        if "极差" in resampled_stats.columns:
            if (
                "最大值" in resampled_stats.columns
                and "最小值" in resampled_stats.columns
            ):
                resampled_stats["极差"] = (
                    resampled_stats["最大值"] - resampled_stats["最小值"]
                )

        return resampled_stats

    def calculate_volatility_features(
        self,
        df: pd.DataFrame,
        field_name: str,
        feature_list: list[str] = None,
        freq: str = "D",
    ) -> pd.DataFrame:
        """
        Calculate volatility features dynamically based on the requested list.

        Args:
            df: DataFrame with DatetimeIndex containing time series data
            field_name: Name of the column to calculate volatility for
            feature_list: List of features to calculate (e.g., ["标准差", "变异系数"]).
                          If None, calculates a default set of volatility metrics.
            freq: Resampling frequency (default "D").

        Returns:
            DataFrame with calculated volatility metrics indexed by time.
        """
        if df.empty:
            logger.error(
                "Empty DataFrame provided to calculate_volatility_features",
                extra={"field_name": field_name},
            )
            return pd.DataFrame()

        if field_name not in df.columns:
            logger.error(
                f"Field '{field_name}' not found in DataFrame",
                extra={"field_name": field_name, "available_columns": list(df.columns)},
            )
            return pd.DataFrame()

        # 设置一个默认集合
        if feature_list is None:
            feature_list = [
                "标准差",
                "变异系数",
                "一阶自相关",
                "平均绝对偏差_均值",
                "平均绝对偏差_中位数",
            ]

        agg_functions = {}
        for feature in feature_list:
            if feature in self._feature_calculators:
                agg_functions[feature] = self._feature_calculators[feature]
            else:
                logger.warning(f"Feature '{feature}' is not registered in calculator.")

        if not agg_functions:
            logger.warning("No valid volatility features requested.")
            return pd.DataFrame()

        logger.info(
            f"Calculating volatility features for '{field_name}'",
            extra={
                "features": list(agg_functions.keys()),
                "freq": freq,
                "data_points": len(df),
            },
        )

        # 执行计算
        volatility_stats = df[field_name].resample(freq).agg(**agg_functions)

        # 清理无效值
        volatility_stats.dropna(inplace=True)
        return volatility_stats

    def calculate_vapor_pressure(
        self, temperature: float | pd.Series, relative_humidity: float | pd.Series
    ) -> float | pd.Series:
        """
        计算水汽压 (Vapor Pressure)

        使用 Magnus-Tetens 公式：
        饱和水汽压: e_s(T) = 6.112 * exp((17.67 * T) / (T + 243.5))
        实际水汽压: VP = (RH / 100) * e_s(T)

        Args:
            temperature: 温度 (℃)，可以是单个值或 Series
            relative_humidity: 相对湿度 (%)，可以是单个值或 Series

        Returns:
            水汽压 (hPa)，返回类型与输入一致

        Example:
            >>> calc = FeatureCalculator()
            >>> vp = calc.calculate_vapor_pressure(25.0, 60.0)
            >>> print(f"水汽压: {vp:.2f} hPa")
        """
        # 计算饱和水汽压 (Magnus-Tetens 公式)
        saturated_vp = 6.112 * np.exp((17.67 * temperature) / (temperature + 243.5))

        # 计算实际水汽压
        actual_vp = (relative_humidity / 100.0) * saturated_vp

        return actual_vp

    def calculate_vapor_pressure_gradient(
        self,
        df_in: pd.DataFrame,
        df_out: pd.DataFrame,
        temp_col_in: str = "temperature",
        humidity_col_in: str = "humidity",
        vapor_pressure_col_out: str = "avg_vapor_pressure",
    ) -> pd.DataFrame:
        """
        计算水汽扩散方向 (洞外 - 洞内的水汽压差)

        ΔVP = VP_out - VP_in

        正值表示水汽从洞外向洞内扩散，负值表示从洞内向洞外扩散。

        **新逻辑**（适配实际数据）:
        - 洞内水汽压：从 temperature 和 humidity 计算
        - 洞外水汽压：直接使用 avg_vapor_pressure 字段

        Args:
            df_in: 洞内数据 DataFrame (需要包含温度和湿度列)
            df_out: 洞外数据 DataFrame (需要包含水汽压列)
            temp_col_in: 洞内温度列名 (默认 "temperature")
            humidity_col_in: 洞内湿度列名 (默认 "humidity")
            vapor_pressure_col_out: 洞外水汽压列名 (默认 "avg_vapor_pressure")

        Returns:
            DataFrame 包含以下列:
                - VP_in: 洞内水汽压 (hPa)
                - VP_out: 洞外水汽压 (hPa)
                - delta_VP: 水汽压差 (hPa)
                - diffusion_direction: 扩散方向 ("向内", "向外", "平衡")

        Example:
            >>> calc = FeatureCalculator()
            >>> # df_in 包含 temperature 和 humidity
            >>> # df_out 包含 avg_vapor_pressure
            >>> gradient_df = calc.calculate_vapor_pressure_gradient(
            ...     df_in, df_out
            ... )
        """
        if df_in.empty or df_out.empty:
            logger.error("输入的 DataFrame 不能为空")
            return pd.DataFrame()

        # 验证洞内数据列
        required_cols_in = [temp_col_in, humidity_col_in]
        missing_cols_in = [col for col in required_cols_in if col not in df_in.columns]
        if missing_cols_in:
            logger.error(f"洞内数据缺少必需的列: {missing_cols_in}")
            logger.error(f"可用列: {list(df_in.columns)}")
            return pd.DataFrame()

        # 验证洞外数据列
        if vapor_pressure_col_out not in df_out.columns:
            logger.error(f"洞外数据缺少必需的列: {vapor_pressure_col_out}")
            logger.error(f"可用列: {list(df_out.columns)}")
            return pd.DataFrame()

        # 确保两个 DataFrame 的索引对齐
        df_aligned = df_in.join(df_out, how="inner", lsuffix="_in", rsuffix="_out")

        if df_aligned.empty:
            logger.warning("洞内和洞外数据没有重叠的时间戳")
            return pd.DataFrame()

        # 计算洞内水汽压（从温度和湿度计算）
        temp_in_col = (
            f"{temp_col_in}_in"
            if f"{temp_col_in}_in" in df_aligned.columns
            else temp_col_in
        )
        humidity_in_col = (
            f"{humidity_col_in}_in"
            if f"{humidity_col_in}_in" in df_aligned.columns
            else humidity_col_in
        )

        vp_in = self.calculate_vapor_pressure(
            df_aligned[temp_in_col], df_aligned[humidity_in_col]
        )

        # 洞外水汽压（直接使用 avg_vapor_pressure）
        vp_out_col = (
            f"{vapor_pressure_col_out}_out"
            if f"{vapor_pressure_col_out}_out" in df_aligned.columns
            else vapor_pressure_col_out
        )
        vp_out = df_aligned[vp_out_col] * 10

        # 计算水汽压差
        delta_vp = vp_out - vp_in

        # 确定扩散方向
        diffusion_direction = pd.Series(index=delta_vp.index, dtype=str)
        diffusion_direction[delta_vp > 0.1] = "向内"  # 阈值 0.1 hPa
        diffusion_direction[delta_vp < -0.1] = "向外"
        diffusion_direction[np.abs(delta_vp) <= 0.1] = "平衡"

        result_df = pd.DataFrame(
            {
                "VP_in": vp_in,
                "VP_out": vp_out,
                "delta_VP": delta_vp,
                "diffusion_direction": diffusion_direction,
            }
        )

        logger.info(
            f"计算了 {len(result_df)} 个时间点的水汽压梯度, "
            f"向内扩散比例: {(diffusion_direction == '向内').sum() / len(result_df):.2%}"
        )

        return result_df

    def calculate_high_humidity_exposure(
        self,
        df: pd.DataFrame,
        humidity_col: str = "humidity",
        threshold: float = 62.0,
        freq: str = "D",
    ) -> pd.DataFrame:
        """
        计算高湿暴露特征 (超过阈值的持续时间)

        识别相对湿度超过阈值的连续时间段，并统计每个周期内的暴露时长。

        Args:
            df: DataFrame with DatetimeIndex containing humidity data
            humidity_col: 湿度列名 (默认 "humidity")
            threshold: 高湿阈值 (默认 62.0%)
            freq: 统计周期 (默认 "D" 天)

        Returns:
            DataFrame 包含以下列:
                - total_exposure_hours: 总暴露时长 (小时)
                - max_continuous_hours: 最长连续暴露时长 (小时)
                - exposure_episodes: 暴露事件次数
                - exposure_ratio: 暴露时间占比

        Example:
            >>> calc = FeatureCalculator()
            >>> exposure = calc.calculate_high_humidity_exposure(
            ...     df, threshold=26.0, freq="D"
            ... )
        """
        if df.empty:
            logger.error("输入的 DataFrame 为空")
            return pd.DataFrame()

        if humidity_col not in df.columns:
            logger.error(f"列 '{humidity_col}' 不存在于 DataFrame 中")
            return pd.DataFrame()

        # 标记高湿时刻
        df = df.copy()
        df["is_high_humidity"] = df[humidity_col] > threshold

        # 按周期分组统计
        resampled = df.resample(freq)

        results = []
        for timestamp, group in resampled:
            if group.empty:
                continue

            # 计算总暴露时长 (小时)
            total_high_humidity_points = group["is_high_humidity"].sum()
            # 假设数据采集间隔
            time_interval_hours = (
                (group.index[1] - group.index[0]).total_seconds() / 3600
                if len(group) > 1
                else 1.0
            )
            total_exposure_hours = total_high_humidity_points * time_interval_hours

            # 计算最长连续暴露时长
            # 识别连续的 True 段
            is_high = group["is_high_humidity"].values
            # 使用 diff 找到转换点
            transitions = np.diff(
                np.concatenate([[False], is_high, [False]]).astype(int)
            )
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]

            if len(starts) > 0 and len(ends) > 0:
                continuous_durations = ends - starts
                max_continuous_points = continuous_durations.max()
                max_continuous_hours = max_continuous_points * time_interval_hours
                exposure_episodes = len(starts)
            else:
                max_continuous_hours = 0.0
                exposure_episodes = 0

            # 计算暴露时间占比
            exposure_ratio = (
                total_high_humidity_points / len(group) if len(group) > 0 else 0.0
            )

            results.append(
                {
                    "time": timestamp,
                    "total_exposure_hours": total_exposure_hours,
                    "max_continuous_hours": max_continuous_hours,
                    "exposure_episodes": exposure_episodes,
                    "exposure_ratio": exposure_ratio,
                }
            )

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df.set_index("time", inplace=True)

        logger.info(
            f"计算了 {len(result_df)} 个周期的高湿暴露, "
            f"平均暴露时长: {result_df['total_exposure_hours'].mean():.2f} 小时"
        )

        return result_df

    def calculate_rainfall_intensity(
        self,
        df: pd.DataFrame,
        rainfall_col: str = "rainfall",
        window: str = "1h",
        freq: str = "6h",
    ) -> pd.DataFrame:
        """
        计算降雨强度特征 (峰值、均值、积分)

        在指定的滑动窗口内计算降雨强度，然后按周期统计峰值、均值和累积量。

        Args:
            df: DataFrame with DatetimeIndex containing rainfall data
            rainfall_col: 降雨量列名 (默认 "rainfall")
            window: 滑动窗口大小 (默认 "1h")
                freq: 统计周期 (默认 "6h")

        Returns:
            DataFrame 包含以下列:
                - rainfall_peak: 降雨强度峰值 (mm/窗口)
                - rainfall_mean: 降雨强度均值 (mm/窗口)
                - rainfall_total: 降雨总量 (mm)
                - rainfall_hours: 降雨持续时长 (小时)

        Example:
            >>> calc = FeatureCalculator()
            >>> intensity = calc.calculate_rainfall_intensity(
            ...     df, window="10min", freq="D"
            ... )
        """
        if df.empty:
            logger.error("输入的 DataFrame 为空")
            return pd.DataFrame()

        if rainfall_col not in df.columns:
            logger.error(f"列 '{rainfall_col}' 不存在于 DataFrame 中")
            return pd.DataFrame()

        # 计算滑动窗口内的降雨强度
        df = df.copy()
        df["rainfall_intensity"] = df[rainfall_col].rolling(window=window).sum()

        # 按周期分组统计
        resampled = df.resample(freq)

        results = []
        for timestamp, group in resampled:
            if group.empty:
                continue

            # 峰值
            peak = group["rainfall_intensity"].max()

            # 均值 (排除零值)
            non_zero_intensity = group["rainfall_intensity"][
                group["rainfall_intensity"] > 0
            ]
            mean = non_zero_intensity.mean() if len(non_zero_intensity) > 0 else 0.0

            # 总量 (积分)
            total = group[rainfall_col].sum()

            # 降雨时长 (降雨量 > 0 的小时数)
            time_interval_hours = (
                (group.index[1] - group.index[0]).total_seconds() / 3600
                if len(group) > 1
                else 1.0
            )
            rainfall_points = (group[rainfall_col] > 0).sum()
            rainfall_hours = rainfall_points * time_interval_hours

            results.append(
                {
                    "time": timestamp,
                    "rainfall_peak": peak,
                    "rainfall_mean": mean,
                    "rainfall_total": total,
                    "rainfall_hours": rainfall_hours,
                }
            )

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df.set_index("time", inplace=True)

        logger.info(
            f"计算了 {len(result_df)} 个周期的降雨强度, "
            f"平均降雨量: {result_df['rainfall_total'].mean():.2f} mm"
        )

        return result_df

    def analyze_humidity_rainfall_correlation(
        self,
        humidity_exposure_df: pd.DataFrame,
        rainfall_df: pd.DataFrame,
        humidity_col: str = "total_exposure_hours",
        rainfall_col: str = "rainfall_peak",
    ) -> dict:
        """
        分析高湿暴露与降雨强度的线性关系

        计算两者之间的相关系数、线性回归参数等统计指标。

        Args:
            humidity_exposure_df: 高湿暴露 DataFrame
            rainfall_df: 降雨强度 DataFrame
            humidity_col: 高湿特征列名 (默认 "total_exposure_hours")
            rainfall_col: 降雨特征列名 (默认 "rainfall_peak")

        Returns:
            字典包含:
                - correlation: 皮尔逊相关系数
                - p_value: 显著性检验 p 值
                - slope: 线性回归斜率
                - intercept: 线性回归截距
                - r_squared: 决定系数 R²
                - n_samples: 样本数

        Example:
            >>> calc = FeatureCalculator()
            >>> exposure = calc.calculate_high_humidity_exposure(df, threshold=62)
            >>> rainfall = calc.calculate_rainfall_intensity(rain_df)
            >>> correlation = calc.analyze_humidity_rainfall_correlation(
            ...     exposure, rainfall
            ... )
            >>> print(f"相关系数: {correlation['correlation']:.3f}")
        """
        from scipy import stats

        # 对齐两个 DataFrame 的时间索引
        df_merged = humidity_exposure_df.join(rainfall_df, how="inner")

        if df_merged.empty or len(df_merged) < 3:
            logger.warning("样本数不足，无法进行相关性分析")
            return {
                "correlation": np.nan,
                "p_value": np.nan,
                "slope": np.nan,
                "intercept": np.nan,
                "r_squared": np.nan,
                "n_samples": 0,
            }

        x = df_merged[humidity_col].values
        y = df_merged[rainfall_col].values

        # 计算皮尔逊相关系数
        correlation, p_value = stats.pearsonr(x, y)

        # 线性回归
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        r_squared = r_value**2

        result = {
            "correlation": correlation,
            "p_value": p_value,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "n_samples": len(df_merged),
        }

        logger.info(
            f"湿度-降雨相关性分析完成: "
            f"相关系数={correlation:.3f}, p值={p_value:.4f}, R²={r_squared:.3f}"
        )

        return result

    # ====================================================================
    #   内部函数-计算统计特征
    # ====================================================================
    def _calculate_mean(self, series: pd.Series) -> float:
        """
        Calculate the mean of a series.

        Args:
            series: Time series data

        Returns:
            Mean value of the series
        """
        return series.mean()

    def _calculate_median(self, series: pd.Series) -> float:
        """
        Calculate the median of a series.

        Args:
            series: Time series data

        Returns:
            Median value of the series
        """
        return series.median()

    def _calculate_max(self, series: pd.Series) -> float:
        """
        Calculate the maximum value of a series.

        Args:
            series: Time series data

        Returns:
            Maximum value of the series
        """
        return series.max()

    def _calculate_min(self, series: pd.Series) -> float:
        """
        Calculate the minimum value of a series.

        Args:
            series: Time series data

        Returns:
            Minimum value of the series
        """
        return series.min()

    def _calculate_q1(self, series: pd.Series) -> float:
        """
        Calculate the first quartile (Q1) of a series.

        Args:
            series: Time series data

        Returns:
            First quartile (Q1) value of the series
        """
        return series.quantile(0.25)

    def _calculate_q3(self, series: pd.Series) -> float:
        """
        Calculate the third quartile (Q3) of a series.

        Args:
            series: Time series data

        Returns:
            Third quartile (Q3) value of the series
        """
        return series.quantile(0.75)

    def _calculate_p10(self, series: pd.Series) -> float:
        """
        Calculate the 10th percentile (P10) of a series.

        Args:
            series: Time series data

        Returns:
            10th percentile (P10) value of the series
        """
        return series.quantile(0.10)

    def _calculate_percent_above_q3(self, series: pd.Series) -> float:
        """
        Calculate the percentage of values above the third quartile (Q3).

        Args:
            series: Time series data

        Returns:
            Percentage of values above Q3 (between 0.0 and 1.0)

        Note:
            Returns 0.0 if series has fewer than 4 points or if Q3 equals max.
        """
        if len(series) < 4:
            return 0.0
        q3 = series.quantile(0.75)
        # Avoid case where Q3 equals max value (no values can be above Q3)
        if q3 == series.max():
            return 0.0
        count_above_q3 = (series > q3).sum()
        percent_above_q3 = count_above_q3 / len(series)
        return percent_above_q3

    def _calculate_range(self, series: pd.Series) -> float:
        """Calculate range (Max - Min)."""
        if series.empty:
            return np.nan
        return series.max() - series.min()

    def _calculate_range_rate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate the percentage change of the range."""
        if "极差" not in df.columns:
            return pd.Series(index=df.index, dtype=float)
        return df["极差"].pct_change(fill_method=None).fillna(0)

    # ====================================================================
    #   内部函数-计算波动性特征
    # ====================================================================
    def _calculate_std(self, series: pd.Series) -> float:
        """
        Calculate the standard deviation of a series.

        Args:
            series: Time series data

        Returns:
            Standard deviation of the series
        """
        return series.std()

    def _mad_from_mean(self, series: pd.Series) -> float:
        """
        Calculate mean absolute deviation from mean.

        Args:
            series: Time series data

        Returns:
            Mean absolute deviation from the series mean
        """
        return (series - series.mean()).abs().mean()

    def _mad_from_median(self, series: pd.Series) -> float:
        """
        Calculate mean absolute deviation from median.

        Args:
            series: Time series data

        Returns:
            Mean absolute deviation from the series median
        """
        return (series - series.median()).abs().mean()

    def _autocorr_lag1(self, series: pd.Series) -> float | None:
        """
        Calculate first-order autocorrelation.

        Args:
            series: Time series data

        Returns:
            First-order autocorrelation coefficient, or None if series
            has fewer than 2 data points
        """
        if len(series) < 2:
            return None
        return series.autocorr(lag=1)

    def _calculate_coefficient_of_variation(self, series: pd.Series) -> float | None:
        """
        Calculate coefficient of variation (CV).

        The coefficient of variation is the ratio of the standard deviation
        to the mean, expressed as a percentage. It's a normalized measure
        of dispersion.

        Args:
            series: Time series data

        Returns:
            Coefficient of variation (std/mean), or None if mean is zero
            or series is empty

        Note:
            Returns None if the mean is zero to avoid division by zero.
        """
        if series.empty:
            return None
        mean = series.mean()
        if mean == 0 or np.isnan(mean):
            return None
        std = series.std()
        if np.isnan(std):
            return None
        return std / mean

    # ====================================================================
    #   其它函数
    # ====================================================================
    def analyze_spectral(
        self, df: pd.DataFrame, field_name: str
    ) -> pd.DataFrame | None:
        """
        Perform FFT spectral analysis on time series data.

        This method removes the long-term trend from the data and applies
        Fast Fourier Transform to identify dominant periodic patterns.
        Frequencies are converted to periods in hours for interpretability.

        Args:
            df: DataFrame with DatetimeIndex containing time series data
            field_name: Name of the column to analyze

        Returns:
            DataFrame with two columns:
                - 周期(小时): Periods in hours
                - 强度(幅度): Corresponding amplitudes
            Returns None if df is empty.

        Example:
            >>> calculator = FeatureCalculator()
            >>> spectrum = calculator.analyze_spectral(df, "temperature")
            >>> # Find dominant period
            >>> dominant_period = spectrum.loc[spectrum["强度(幅度)"].idxmax(), "周期(小时)"]
        """
        if df.empty:
            logger.warning(
                "Empty DataFrame provided to analyze_spectral",
                extra={"field_name": field_name},
            )
            return None

        if field_name not in df.columns:
            logger.error(
                f"Field '{field_name}' not found in DataFrame for spectral analysis",
                extra={"field_name": field_name, "available_columns": list(df.columns)},
            )
            return None

        logger.info(
            f"Starting spectral analysis (FFT) for field '{field_name}'",
            extra={"field_name": field_name, "data_points": len(df)},
        )

        # 1. Remove long-term trend
        logger.debug("Detrending data for spectral analysis")
        detrended_values = detrend(df[field_name].values)

        # 2. Compute FFT
        N = len(detrended_values)
        # Automatically calculate sampling interval from DataFrame index
        T = (df.index[1] - df.index[0]).total_seconds()
        yf = fft(detrended_values)
        xf = fftfreq(N, T)[: N // 2]
        amplitude = 2.0 / N * np.abs(yf[0 : N // 2])

        # 3. Convert frequencies to periods in hours
        periods_in_hours = np.full_like(xf, np.inf)
        non_zero_indices = xf > 0
        periods_in_hours[non_zero_indices] = 1 / xf[non_zero_indices] / 3600

        # 4. Build result DataFrame
        spectrum_df = pd.DataFrame(
            {"周期(小时)": periods_in_hours, "强度(幅度)": amplitude}
        )

        logger.info(
            f"Spectral analysis completed for field '{field_name}'",
            extra={"field_name": field_name, "spectrum_points": len(spectrum_df)},
        )

        return spectrum_df

    def find_stable_period(
        self, volatility_df: pd.DataFrame, threshold: float
    ) -> tuple[date | None, date | None, int]:
        """
        Find the longest consecutive stable period.

        A stable period is defined as consecutive days where the standard
        deviation remains below the specified threshold.

        Args:
            volatility_df: DataFrame from calculate_volatility_features,
                          must contain a '标准差' column
            threshold: Standard deviation threshold for stability

        Returns:
            Tuple of (start_date, end_date, length_in_days):
                - start_date: First date of the longest stable period
                - end_date: Last date of the longest stable period
                - length_in_days: Number of consecutive stable days
            Returns (None, None, 0) if no stable period exists.

        Example:
            >>> calculator = FeatureCalculator()
            >>> volatility = calculator.calculate_volatility_features(df, "temperature")
            >>> start, end, length = calculator.find_stable_period(volatility, threshold=1.5)
            >>> print(f"Stable period: {start} to {end} ({length} days)")
        """
        logger.info(
            f"Finding stable period with threshold {threshold}",
            extra={"threshold": threshold, "total_days": len(volatility_df)},
        )

        volatility_df["is_stable"] = volatility_df["标准差"] < threshold

        # Use groupby and shift to identify consecutive stable periods
        streaks = (
            volatility_df[volatility_df["is_stable"]]
            .groupby(
                (
                    volatility_df["is_stable"] != volatility_df["is_stable"].shift()
                ).cumsum()
            )
            .size()
        )

        if streaks.empty:
            logger.info("No stable period found", extra={"threshold": threshold})
            return None, None, 0

        # Find the longest streak
        longest_streak_len = streaks.max()
        streak_group_id = streaks.idxmax()

        # Extract dates for the longest streak
        streak_df = volatility_df[
            (volatility_df["is_stable"])
            & (
                (
                    volatility_df["is_stable"] != volatility_df["is_stable"].shift()
                ).cumsum()
                == streak_group_id
            )
        ]

        start_date = streak_df.index.min().date()
        end_date = streak_df.index.max().date()

        logger.info(
            f"Found stable period: {start_date} to {end_date} ({longest_streak_len} days)",
            extra={
                "start_date": str(start_date),
                "end_date": str(end_date),
                "length_days": longest_streak_len,
                "threshold": threshold,
            },
        )

        return start_date, end_date, longest_streak_len

    def recalculate_single_window(self, series: pd.Series) -> pd.Series:
        """
        Recalculate features for a single time window.

        This method computes all non-rate-of-change statistical features
        for a single time window (e.g., one hour of data). Useful for
        validation and testing purposes.

        Args:
            series: Time series data for one window

        Returns:
            Series with calculated feature values, indexed by feature names.
            Returns empty Series if input is empty.

        Example:
            >>> calculator = FeatureCalculator()
            >>> # Get one hour of data
            >>> hour_data = df.loc["2024-01-01 00:00":"2024-01-01 01:00", "temperature"]
            >>> features = calculator.recalculate_single_window(hour_data)
            >>> print(features["均值"])
        """
        if series.empty:
            logger.warning("Empty series provided to recalculate_single_window")
            return pd.Series(dtype="object")

        logger.debug(
            "Recalculating features for single window",
            extra={"data_points": len(series)},
        )

        recalculated_features = pd.Series(
            {
                "均值": series.mean(),
                "中位数_Q2": series.median(),
                "最大值": series.max(),
                "最小值": series.min(),
                "Q1": series.quantile(0.25),
                "Q3": series.quantile(0.75),
                "P10": series.quantile(0.10),
                "极差": series.max() - series.min(),
                "超过Q3占时比": self._calculate_percent_above_q3(series),
            }
        )

        return recalculated_features

    def calculate_autocorrelation(
        self, series: pd.Series, max_lag: int | None = None
    ) -> pd.Series:
        """
        Calculate autocorrelation function for multiple lags.

        This method computes the autocorrelation coefficients for lags
        from 0 to max_lag (or up to len(series)-1 if max_lag is None).

        Args:
            series: Time series data
            max_lag: Maximum lag to calculate. If None, calculates up to
                    len(series) - 1. Default is None.

        Returns:
            Series with autocorrelation values indexed by lag.
            Returns empty Series if input is invalid.

        Example:
            >>> calculator = FeatureCalculator()
            >>> acf = calculator.calculate_autocorrelation(series, max_lag=10)
            >>> print(acf[1])  # First-order autocorrelation
        """
        if series.empty or len(series) < 2:
            logger.warning(
                "Series too short for autocorrelation calculation",
                extra={"series_length": len(series)},
            )
            return pd.Series(dtype=float)

        if max_lag is None:
            max_lag = len(series) - 1
        else:
            max_lag = min(max_lag, len(series) - 1)

        autocorrs = []
        lags = []

        for lag in range(max_lag + 1):
            if lag == 0:
                # Lag 0 is always 1.0 (perfect correlation with itself)
                autocorrs.append(1.0)
            else:
                autocorr = series.autocorr(lag=lag)
                if autocorr is None or np.isnan(autocorr):
                    break
                autocorrs.append(autocorr)
            lags.append(lag)

        result = pd.Series(autocorrs, index=lags, name="自相关系数")
        logger.debug(
            f"Calculated autocorrelation for {len(result)} lags",
            extra={"max_lag": max_lag, "valid_lags": len(result)},
        )
        return result
