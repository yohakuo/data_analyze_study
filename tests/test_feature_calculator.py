import pandas as pd
import pytest

from src.calculator import FeatureCalculator


class TestRealData:
    def transform_benchmark_to_wide(
        self,
        benchmark_df,
        target_variable,
        time_col="stats_start_time",
        value_col="feature_value",
        key_col="feature_key",
    ):
        """
        更健壮的长表转宽表函数
        """
        if target_variable:
            if "monitored_variable" in benchmark_df.columns:
                benchmark_df = benchmark_df[
                    benchmark_df["monitored_variable"] == target_variable
                ]

        # 2. 确保时间列是 datetime 类型
        if time_col in benchmark_df.columns:
            benchmark_df[time_col] = pd.to_datetime(benchmark_df[time_col])
        else:
            raise ValueError(f"基准数据中找不到时间列: {time_col}")

        # 3. 透视表 (Pivot)
        # 使用 pivot_table 并指定 aggfunc='first' 可以防止因重复时间戳导致的报错
        wide_df = benchmark_df.pivot_table(
            index=time_col, columns=key_col, values=value_col, aggfunc="first"
        )

        # 4. 清洗索引和列名
        wide_df.columns.name = None  # 去掉 feature_key 这个名字
        wide_df.index.name = "time"  # 统一索引名为 time

        # 5. 确保按时间排序
        wide_df = wide_df.sort_index()

        # 6. 统一去时区 (防止一个是 UTC 一个是 None)
        if wide_df.index.tz is not None:
            wide_df.index = wide_df.index.tz_localize(None)

        return wide_df

    @pytest.fixture
    def real_data(self):
        df = pd.read_csv(
            "tests/data/real_input.csv", parse_dates=["time"], index_col="time"
        )
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.sort_index()

    @pytest.fixture
    def benchmark_data_wide(self):
        # 读取 CSV，注意处理 header
        df = pd.read_csv("tests/data/old_benchmark.csv")

        wide_df = self.transform_benchmark_to_wide(
            df,
            target_variable="空气湿度（%）",
        )
        return wide_df

    def test_features_match(
        self,
        real_data,
        benchmark_data_wide,
    ):
        calculator = FeatureCalculator()
        # 运行计算
        actual_result = calculator.calculate_statistical_features(
            df=real_data,
            field_name="humidity",
            feature_list=["均值"],
            freq="h",
        )
        # print(actual_result.columns)

        column_mapping = {
            "均值": "均值"  # 左边是基准的列名，右边是你计算函数的列名
        }

        for bench_col in column_mapping.keys():
            assert bench_col in benchmark_data_wide.columns, (
                f"基准数据转换后缺少列: {bench_col}"
            )

        # 重命名基准数据，使其与实际结果的列名一致
        expected_df = benchmark_data_wide.rename(columns=column_mapping)

        # 只保留我们需要对比的列 (即映射中存在的列)
        cols_to_compare = list(column_mapping.values())
        expected_df = expected_df[cols_to_compare]

        # 3. 时间对齐 (Index Alignment)
        # 确保 actual_result 也是去除了时区的
        if actual_result.index.tz is not None:
            actual_result.index = actual_result.index.tz_localize(None)

        # 求时间交集
        common_index = actual_result.index.intersection(expected_df.index)

        if len(common_index) == 0:
            print("\n=== DEBUG INFO ===")
            print("Actual Index Head:", actual_result.index[:5])
            print("Expected Index Head:", expected_df.index[:5])
            pytest.fail(
                "时间索引完全没有交集！请检查：1.是否有时区差异？2.基准时间是整点开始还是结束？"
            )

        # 4. 数据切片
        actual_aligned = actual_result.loc[common_index, cols_to_compare]
        expected_aligned = expected_df.loc[common_index, cols_to_compare]

        # 5. 处理 NaN
        # 实际计算中，如果有缺失值，pandas计算均值通常会忽略；
        # 确保两边都 dropna，只比对都有数据的时刻
        combined = pd.concat([actual_aligned, expected_aligned], axis=1).dropna()

        # 重新分离
        actual_final = combined.iloc[:, : len(cols_to_compare)]
        expected_final = combined.iloc[:, len(cols_to_compare) :]

        # 6. 断言
        print(f"\n正在比对 {len(actual_final)} 行数据...")
        pd.testing.assert_frame_equal(
            actual_final,
            expected_final,
            check_freq=False,  # 忽略频率属性差异
            check_dtype=False,  # 忽略 float64 vs float32
            rtol=0.05,  # 允许 5% 的相对误差
            atol=0.01,  # 允许 0.01 的绝对误差
        )

    # 使用 parametrize，可以一次性测试多种组合
    @pytest.mark.parametrize(
        "feature_list",
        [
            ["均值"],
            ["均值", "最大值", "Q3"],
            [
                "均值",
                "中位数",
                "最大值",
                "最小值",
                "Q1",
                "Q3",
                "P10",
            ],
        ],
    )
    def test_hourly_mean_calculation(self, real_data, feature_list):
        field_name = "humidity"
        freq = "M"
        calculator = FeatureCalculator()
        actual_result = calculator.calculate_statistical_features(
            df=real_data,
            field_name=field_name,
            feature_list=feature_list,
            freq=freq,
        )

        pandas_funcs = []
        try:
            mapping = {
                "均值": "mean",
                "中位数": "median",
                "最大值": "max",
                "最小值": "min",
                "标准差": "std",
                "Q1": lambda x: x.quantile(0.25),
                "Q3": lambda x: x.quantile(0.75),
                "P10": lambda x: x.quantile(0.10),
            }
            pandas_funcs = [mapping[f] for f in feature_list]
        except KeyError as e:
            pytest.fail(f"测试脚本暂不支持基准计算特征: {e}, 请在 mapping 中添加")

        # resample(freq)[col].agg(['mean', 'max']) 会直接生成对应的列
        expected_df = real_data.resample(freq)[field_name].agg(pandas_funcs)

        # 列名、时间对齐
        expected_df.columns = feature_list
        common_index = actual_result.index.intersection(expected_df.index)
        if len(common_index) == 0:
            pytest.fail("时间索引无交集，请检查时区或重采样频率")

        actual_aligned = actual_result.loc[common_index]
        expected_aligned = expected_df.loc[common_index]
        # 确保列顺序一致
        actual_aligned = actual_aligned[feature_list]
        expected_aligned = expected_aligned[feature_list]

        # 断言
        pd.testing.assert_frame_equal(
            actual_aligned,
            expected_aligned,
            check_like=True,  # 忽略列的顺序
            check_dtype=False,  # 忽略 int vs float
            rtol=0.01,  # 允许 1% 的误差
        )
