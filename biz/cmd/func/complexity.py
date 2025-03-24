import os
from heapq import nlargest
from pathlib import Path

import lizard
from tqdm import tqdm

from biz.cmd.func.base import BaseReviewFunc


class ComplexityReviewFunc(BaseReviewFunc):
    """
    计算代码复杂度.
    """

    def __init__(self):
        super().__init__()
        self.directory = None

    def validate_directory(self, directory):
        """
        验证用户输入的目录是否存在。
        :param directory: 用户输入的目录路径
        :return: 如果目录存在返回 True，否则返回 False
        """
        try:
            return Path(directory).resolve().is_dir()
        except Exception:
            return False

    def parse_arguments(self):
        """
        使用交互方式获取用户输入的参数。
        """

        # 获取项目目录
        while True:
            self.directory = input("请输入代码项目的根目录路径: ").strip()
            if self.validate_directory(self.directory):
                break
            print("❌ 目录不存在，请输入有效路径")

    # def find_most_complex_files(self, top_n=5, ):
    #     analysis_result = lizard.analyze([self.directory])
    #     top_files = nlargest(top_n, analysis_result, key=lambda x: x.average_cyclomatic_complexity)
    #     return top_files

    def find_most_complex_functions(self, top_n=5):
        analysis_result = lizard.analyze([self.directory])
        functions = []
        # for file_info in analysis_result:
        for file_info in tqdm(analysis_result, desc="分析文件", unit="file"):
            functions.extend(file_info.function_list)  # 提取所有函数

        top_functions = nlargest(top_n, functions, key=lambda f: f.cyclomatic_complexity)
        return top_functions

    # def process(self):
    #     self.parse_arguments()
    #     top_files = self.find_most_complex_files(top_n=10)
    #     print("🔥 以下是最复杂的文件：")
    #     for file in top_files:
    #         print(f"{file.filename} - 平均圈复杂度: {file.average_cyclomatic_complexity:.2f}")

    def process(self):
        self.parse_arguments()
        top_functions = self.find_most_complex_functions(top_n=10)
        print("🔥 以下是最复杂的函数：")
        for func in top_functions:
            print(f"{func.name} (文件: {func.filename}, 复杂度: {func.cyclomatic_complexity})")


if __name__ == '__main__':
    ComplexityReviewFunc().process()
