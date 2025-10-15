## 核心功能

- **数据提取**: 从 InfluxDB 中高效查询指定时间范围的时间序列数据。
- **数据预处理**: 标准化的数据清洗流程，包括时区转换、重采样和缺失值填充。
- **特征工程**: 提供多种特征计算方法（例如，统计特征、窗口特征等）。
- **数据存储**: 将计算好的特征数据持久化存储到 ClickHouse 中，便于后续的分析和建模。
- **交互式分析**: 集成 Jupyter Notebook/Lab，方便进行探索性数据分析（EDA）和算法验证。

## 项目结构

```
ts_fe/
│
├── data/              # 存放原始、中间和处理后的数据
├── notebooks/         # 用于交互式分析的 Jupyter Notebooks
├── src/               # 项目核心源代码
│   ├── config.py      # 存放数据库连接等所有配置信息
│   ├── dataset.py     # 数据加载和存储模块
│   ├── features.py    # 特征工程模块
│   └── ...
├── tests/             # 自动化测试脚本
│
├── run_features_cal.py # 示例：运行特征计算的主脚本
├── requirements.txt   # 项目依赖
└── README.md          # 项目说明文档
```

## 快速开始

### 1. 环境准备

- Python 3.9+
- Git

### 2. 安装与配置

**a. 克隆项目**
```bash
git clone <your-repository-url>
cd ts_fe
```

**b. 创建并激活虚拟环境** (推荐)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**c. 安装依赖**
```bash
pip install -r requirements.txt
```

**d. 配置数据库连接**

在 `src/config.py` 文件中，根据您的实际环境填写 InfluxDB 和 ClickHouse 的连接信息。

```python
# src/config.py

# InfluxDB settings
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "your-influxdb-token"
INFLUXDB_ORG = "your-org"
INFLUXDB_BUCKET = "your-bucket"

# ClickHouse settings
CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123
# ... 其他配置
```

### 3. 如何使用

**a. 运行特征工程流水线**

您可以直接运行根目录下的 `run_*.py` 脚本来执行完整的任务。例如，运行特征计算：

```bash
python run_features_cal.py
```

**b. 进行交互式分析**

项目已配置好使用 Jupyter。在项目根目录下启动 Jupyter Lab：

```bash
jupyter lab
```

然后在 `notebooks` 文件夹中创建或打开一个 `.ipynb` 文件，您就可以轻松导入 `src` 目录下的函数进行数据分析和可视化。
