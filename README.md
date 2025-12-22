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
├── docs/              # 项目文档
├── models/            # 训练好的模型
├── notebooks/         # 用于交互式分析的 Jupyter Notebooks
├── reports/           # 生成的报告和图表
├── scripts/           # 辅助和独立的脚本集合
├── src/               # 项目核心源代码
│   ├── __init__.py
│   ├── config.py      # 核心配置文件，包含数据库连接、文件路径等
│   ├── dataset.py     # 数据加载、预处理和存储模块
│   ├── features.py    # 特征工程模块
│   ├── plots.py       # 可视化函数
│   └── ...
├── tests/             # 自动化测试脚本
│
├── run_preprocessing.py # 运行数据预处理
├── run_features_cal.py  # 运行特征计算
├── run_query.py         # 交互式查询ClickHouse数据库
│
├── Makefile           # 自动化命令集合
├── requirements.txt   # 项目依赖
└── README.md          # 项目说明文档
```

dataset
```
  1. 数据库连接 (Database Clients)
  2. 数据读写 (Data I/O)
  3. 数据预处理 (Preprocessing)
  4. 高级工作流 (High-Level Workflows)
```


## 快速开始

### 1. 环境准备
- Python 3.12+
- Git
- [uv](https://github.com/astral-sh/uv) (推荐, 用于快速创建虚拟环境和安装依赖)

### 2. 安装与配置
提供了两种安装方式：使用 `Makefile` 的自动化方式（推荐）和传统的手动方式。

---
#### **方式一：使用 `Makefile` (推荐)**

`Makefile` 提供了一系列快捷命令来简化环境配置和项目管理。

**a. 克隆项目**
```bash
git clone <your-repository-url>
cd ts_fe
```

**b. 创建虚拟环境并安装依赖**

执行以下命令，它会自动创建虚拟环境并安装 `requirements.txt` 中所有的依赖。
```bash
make create_environment
make requirements
```
根据提示激活虚拟环境：
```bash
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source ./.venv/bin/activate
```

---

#### **方式二：手动安装**

**a. 克隆项目**
```bash
git clone <your-repository-url>
cd ts_fe
```

**b. 创建并激活虚拟环境**
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

**c. 安装依赖**
```bash
pip install -r requirements.txt
```

---

### 3. 修改核心配置

**重要**: 在运行任何脚本之前，请务必修改 `src/config.py` 文件。

此文件包含了所有关键的配置信息，远不止数据库连接。你需要根据你的本地开发环境或生产环境，仔细检查并修改以下部分：

- **数据库连接**:
  - `CLICKHOUSE_SHARED_*`: 远程 ClickHouse 服务器配置。
  - `INFLUXDB_*`: 本地 InfluxDB 配置。
  - `CLICKHOUSE_HOST`, `CLICKHOUSE_PORT`, etc.: 本地 ClickHouse 配置。
- **文件与路径**:
  - `PROCESSED_DATA_PATH`: 处理后数据的存放路径。
  - `FIGURES_PATH`: 图表输出路径。
- **数据处理规则**:
  - `RAW_FILE_PARSING_CONFIG`: 定义了如何从原始文件名中解析元数据（如设备ID、传感器类型）。
  - `RAW_SENSOR_MAPPING_CONFIG`: 定义了不同传感器的数据如何映射到数据库的表中。
- **查询参数**:
  - `MEASUREMENT_NAME`, `FIELD_NAME`: 执行查询和计算时默认的表名和字段名。

### 4. 如何使用

项目根目录下提供了一系列 `run_*.py` 脚本，作为不同任务的入口。

**a. 数据预处理**

运行 `run_preprocessing.py` 脚本，它会从数据源（如 InfluxDB）提取原始数据，执行清洗、重采样和插值，然后将处理后的数据存入 ClickHouse 或保存为 Parquet 文件。

```bash
python run_preprocessing.py
```

**b. 运行特征计算**

在数据预处理完成后，运行 `run_features_cal.py` 来执行特征工程。该脚本会加载预处理好的数据，计算配置好的特征（如统计特征、波动性特征等），并将结果存入 ClickHouse。

```bash
python run_features_cal.py
```

**c. 交互式数据库查询**

运行 `run_query.py` 脚本，可以启动一个交互式的命令行客户端，直接对 ClickHouse 数据库执行 SQL 查询，方便快速验证数据和进行探索性分析。

```bash
python run_query.py
```

**d. 进行探索性分析 (Jupyter)**

项目已配置好使用 Jupyter。在项目根目录下启动 Jupyter Lab：
```bash
jupyter lab
```
然后在 `notebooks` 文件夹中创建或打开 `.ipynb` 文件，你可以轻松导入 `src` 目录下的函数进行自定义的数据分析和可视化。

**e. 代码质量检查 (可选)**

`Makefile` 中还提供了代码格式化和静态检查的命令：
```bash
# 检查代码格式和规范
make lint

# 自动修复和格式化代码
make format
```

**f. 运行测试**

使用 `pytest` 运行项目中的自动化测试。

要运行所有测试 (包括 `tests/` 目录下的测试文件):
```bash
python -m pytest
```

要运行指定文件中的所有测试 (例如 `test_feature_calculator.py`):
```bash
python -m pytest tests/test_feature_calculator.py
```

如果你正在使用 `uv` 并且没有激活虚拟环境，可以使用 `uv run`：
```bash
uv run pytest
```

