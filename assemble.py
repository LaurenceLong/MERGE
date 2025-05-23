import os
import sys

# --- 配置 ---

# 要排除的目录名称 (精确匹配)
EXCLUDE_DIRS = {
    '.git',
    '__pycache__',
    '.vscode',
    '.idea',
    'venv',
    '.env',
    'node_modules',
    'build',
    'dist',
    'checkpoints',
    'data',
    # 添加其他你想要排除的目录
}

# 要排除的文件名称 (精确匹配)
EXCLUDE_FILES = {
    '.DS_Store',
    'assemble.py',
    'disassemble.py',
    'response.txt',
    'project_prompt.txt',
    'README.md',
    # 添加其他你想要排除的文件
}

# 要包含的文件扩展名 (如果列表不为空，则只包含这些扩展名的文件)
# 如果为空列表 `[]` 或 `None`，则包含所有未被 EXCLUDE_FILES 排除的文件
INCLUDE_EXTENSIONS = [
    '.py',
    '.md',
    '.txt',
    '.json',
    '.yaml',
    '.yml',
    '.sh',
    # 添加其他你想要包含的文件扩展名
]
#INCLUDE_EXTENSIONS = [] # 取消注释此行以包含所有文件类型

# 项目根目录 (默认是当前脚本所在的目录的父目录，或者就是当前工作目录)
# 通常你会在项目根目录下运行这个脚本
ROOT_DIR = os.getcwd()


# --- 脚本主体 ---

def get_language_hint(filename):
    """根据文件扩展名猜测代码块的语言提示"""
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext == '.py':
        return 'python'
    elif ext == '.md':
        return 'markdown'
    elif ext == '.js':
        return 'javascript'
    elif ext == '.html':
        return 'html'
    elif ext == '.css':
        return 'css'
    elif ext == '.json':
        return 'json'
    elif ext == '.yaml' or ext == '.yml':
        return 'yaml'
    elif ext == '.sh':
        return 'bash'
    elif ext == '.txt':
        return 'text'
    else:
        # 默认或对于未知类型返回 'text' 或空字符串
        return 'text'


def generate_project_structure(root_dir, exclude_dirs, exclude_files, include_extensions):
    """生成项目目录结构的字符串表示"""
    tree_lines = []
    project_root_name = os.path.basename(root_dir)
    tree_lines.append(f"{project_root_name}/")

    # 使用集合进行快速查找
    exclude_dirs_set = set(exclude_dirs)
    exclude_files_set = set(exclude_files)
    include_extensions_set = set(include_extensions) if include_extensions else None

    # 获取脚本自身的文件名，以便在遍历时排除它
    try:
        script_name = os.path.basename(sys.argv[0])
        exclude_files_set.add(script_name)
    except IndexError:
        # 如果无法获取脚本名称（例如在交互式环境中运行），则忽略
        pass

    for root, dirs, files in os.walk(root_dir, topdown=True):
        # 过滤掉需要排除的目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs_set]

        relative_root = os.path.relpath(root, root_dir)
        if relative_root == ".":
            level = 0
        else:
            level = relative_root.count(os.sep) + 1

        indent = '    ' * level  # 使用4个空格作为缩进

        # 添加目录到树结构
        if level > 0:  # 不重复添加根目录
            dir_name = os.path.basename(root)
            tree_lines.append(f"{indent}├── {dir_name}/")
            indent += '│   '  # 下一级文件/目录的连接线

        # 过滤并排序文件
        valid_files = []
        for f in files:
            if f in exclude_files_set:
                continue
            if include_extensions_set:
                _, ext = os.path.splitext(f)
                if ext.lower() not in include_extensions_set:
                    continue
            valid_files.append(f)

        valid_files.sort()  # 保证文件顺序一致

        # 添加文件到树结构
        num_files = len(valid_files)
        for i, f in enumerate(valid_files):
            prefix = '└── ' if i == num_files - 1 else '├── '
            tree_lines.append(f"{indent}{prefix}{f}")

    return "\n".join(tree_lines)


def read_files_content(root_dir, exclude_dirs, exclude_files, include_extensions):
    """读取项目中符合条件的文件内容"""
    file_contents = []
    exclude_dirs_set = set(exclude_dirs)
    exclude_files_set = set(exclude_files)
    include_extensions_set = set(include_extensions) if include_extensions else None

    # 获取脚本自身的文件名
    try:
        script_name = os.path.basename(sys.argv[0])
        exclude_files_set.add(script_name)
    except IndexError:
        pass

    collected_files = []  # 用于排序

    for root, dirs, files in os.walk(root_dir, topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude_dirs_set]

        for f in files:
            if f in exclude_files_set:
                continue

            file_path = os.path.join(root, f)
            relative_path = os.path.relpath(file_path, root_dir)

            if include_extensions_set:
                _, ext = os.path.splitext(f)
                if ext.lower() not in include_extensions_set:
                    continue

            collected_files.append((relative_path, file_path))

    collected_files.sort()  # 按相对路径排序

    for relative_path, file_path in collected_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                content = infile.read()
            language = get_language_hint(relative_path)
            file_contents.append(f"## {relative_path}\n```{language}\n{content}\n```")
        except Exception as e:
            file_contents.append(f"## {relative_path}\n```text\nError reading file: {e}\n```")

    return "\n---\n\n".join(file_contents)  # 使用 '---' 分隔文件内容


def main():
    """主函数，生成并打印 prompt"""
    print("Generating project overview and file contents...")

    # 1. 生成项目结构树
    project_structure = generate_project_structure(
        ROOT_DIR,
        EXCLUDE_DIRS,
        EXCLUDE_FILES,
        INCLUDE_EXTENSIONS
    )

    # 2. 读取文件内容
    files_content = read_files_content(
        ROOT_DIR,
        EXCLUDE_DIRS,
        EXCLUDE_FILES,
        INCLUDE_EXTENSIONS
    )

    # 3. 组装最终的 prompt
    final_prompt = "\n--- Start ---\n"
    final_prompt += f"# 项目总览\n\n```\n{project_structure}\n```\n---\n\n{files_content}"

    # 4. 打印或保存 prompt
    final_prompt += "\n--- End ---"
    print(final_prompt)

    # 可选：将 prompt 保存到文件
    output_filename = "project_prompt.txt"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_prompt)
        print(f"\nPrompt successfully saved to {output_filename}")
    except Exception as e:
        print(f"\nError saving prompt to file: {e}")


if __name__ == "__main__":
    main()
