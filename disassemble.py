"""
该脚本用于自动创建索引编辑器项目结构和文件内容
从Claude的输出中提取所有代码块并创建相应的文件
"""

import os
import re
import sys

# 定义项目根目录
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# 文件内容提取正则表达式
head = "#" * 3
FILE_PATTERN = "\n" + head + r" (.*?)[\r\n]+[.\r\n]*```.*?\n(.*?)```"


def extract_files_from_content(content):
    """从内容字符串中提取文件路径和代码内容"""
    matches = re.findall(FILE_PATTERN, content, re.DOTALL)
    files = []

    for file_path, code in matches:
        if file_path.find(".") < 0:
            continue
        # 清理文件路径
        length = 0
        for fp in file_path.strip().split(" "):
            if "." in fp and len(fp) > length:
                fp = fp.strip("`")
                reg = re.findall(r'([A-Za-z0-9_]+\.[A-Za-z0-9]+)', fp)
                if reg:
                    length = len(fp)
                    file_name = reg[0]
                    file_path = fp[:fp.find(file_name) + len(file_name)]
                    # file_path = "merge_model/" + file_path

        # 将requirements.txt单独处理
        if file_path == "requirements.txt":
            files.append((file_path, code))
        else:
            files.append((file_path, code))

    return files


def create_project_structure(files):
    """创建项目目录结构并写入文件内容"""
    # 创建项目根目录
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    print(f"创建项目根目录: {PROJECT_ROOT}")

    # 创建文件并写入内容
    for file_path, content in files:
        # 构建完整文件路径
        full_path = os.path.join(PROJECT_ROOT, file_path)

        # 确保目录存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # 写入文件内容
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"创建文件: {full_path}")


def read_claude_response():
    """读取Claude回答内容"""
    print("请粘贴Claude的完整回答，然后按Ctrl+D (Unix/Linux/Mac)或Ctrl+Z (Windows)结束输入:")
    content = sys.stdin.read()
    return content


def main():
    # 读取Claude的回答
    # content = read_claude_response()
    with open("response.txt", encoding='utf-8') as fd:
        content = fd.read()

    # 提取文件
    files = extract_files_from_content(content)
    print(f"从回答中提取了 {len(files)} 个文件")

    # 创建项目结构
    create_project_structure(files)

    print(f"\n项目创建完成! 项目位置: {os.path.abspath(PROJECT_ROOT)}")
    print("目录结构:")
    for root, dirs, files in os.walk(PROJECT_ROOT):
        level = root.replace(PROJECT_ROOT, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")


if __name__ == "__main__":
    main()