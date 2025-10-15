import os

from src import config
from src.data.make_dataset import import_csv_to_influx

RAW_DATA_DIR = "data/raw"
DATA_SUBFOLDER_TO_IMPORT = "dongnan"  # 如果要导入子文件夹，指定子文件夹名称，否则留空


def main():
    """
    主函数，用于执行数据导入任务。
    可以指定导入单个文件，或自动导入文件夹下的所有CSV文件。
    """

    # os.path.join 会智能地处理路径，即使 DATA_SUBFOLDER_TO_IMPORT 为空也没问题
    target_dir = os.path.join(RAW_DATA_DIR, DATA_SUBFOLDER_TO_IMPORT)

    print(f"▶️ 正在扫描目标文件夹: '{target_dir}'")

    try:
        filenames_to_import = [f for f in os.listdir(target_dir) if f.lower().endswith(".csv")]
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件夹 '{target_dir}'。请检查 config.py 中的路径设置是否正确。")
        return

    if not filenames_to_import:
        print("ℹ️ 在目标文件夹中没有找到任何 .csv 文件，流程结束。")
        return

    print(f"准备导入以下文件: {filenames_to_import}")

    for filename in filenames_to_import:
        full_path = os.path.join(target_dir, filename)

        import_csv_to_influx(full_path)


if __name__ == "__main__":
    main()
