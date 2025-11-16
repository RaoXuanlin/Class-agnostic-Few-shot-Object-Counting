# 如何在 VS Code 中打开和运行更新后的仓库

本指南将帮助您在 VS Code 中克隆、打开并运行 FSC-147 数据集的 class-agnostic object counting 项目。

## 第一步：安装必要工具

### 1.1 安装 VS Code
如果还没有安装 VS Code：
- 访问：https://code.visualstudio.com/
- 下载并安装适合您操作系统的版本

### 1.2 安装 Git
- Windows：https://git-scm.com/download/win
- macOS：通常已预装，或运行 `brew install git`
- Linux：`sudo apt-get install git` (Ubuntu/Debian) 或 `sudo yum install git` (CentOS/RHEL)

### 1.3 安装 Python
- 下载 Python 3.8 或更高版本：https://www.python.org/downloads/
- 安装时请勾选 "Add Python to PATH"

## 第二步：克隆仓库

### 方法 1：使用 VS Code 内置功能克隆

1. 打开 VS Code
2. 按 `Ctrl+Shift+P` (Windows/Linux) 或 `Cmd+Shift+P` (macOS) 打开命令面板
3. 输入 `Git: Clone` 并选择
4. 粘贴仓库地址：
   ```
   https://github.com/RaoXuanlin/Class-agnostic-Few-shot-Object-Counting.git
   ```
5. 选择本地存储位置
6. 克隆完成后，点击 "Open" 打开仓库

### 方法 2：使用命令行克隆

1. 打开终端（Terminal）
2. 进入您想要存放项目的目录：
   ```bash
   cd /path/to/your/projects
   ```
3. 克隆仓库：
   ```bash
   git clone https://github.com/RaoXuanlin/Class-agnostic-Few-shot-Object-Counting.git
   ```
4. 在 VS Code 中打开：
   - 打开 VS Code
   - 选择 `File` → `Open Folder...`
   - 选择刚克隆的 `Class-agnostic-Few-shot-Object-Counting` 文件夹

## 第三步：设置 Python 环境

### 3.1 在 VS Code 中打开终端
- 快捷键：`` Ctrl+` `` (Windows/Linux) 或 `` Cmd+` `` (macOS)
- 或者：菜单栏 `Terminal` → `New Terminal`

### 3.2 创建虚拟环境（推荐）

在终端中运行：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

激活后，您会看到终端提示符前面有 `(venv)` 标记。

### 3.3 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装 pycocotools (如果需要使用 COCO 数据集)
pip install pycocotools
```

## 第四步：准备 FSC-147 数据集

### 4.1 下载数据集
1. 访问：https://github.com/cvlab-stonybrook/LearningToCountEverything
2. 下载 FSC-147_384_V2.zip
3. 解压到一个位置，例如：`D:\datasets\FSC147_384_V2`

### 4.2 配置数据集路径
1. 在 VS Code 中打开 `configs/config_fsc147.yaml`
2. 修改 `data_path` 为您的数据集路径：
   ```yaml
   data:
     dataset: fsc147
     data_path: D:\datasets\FSC147_384_V2  # 改为您的路径
     num_references: 3
   ```
3. 保存文件 (`Ctrl+S` 或 `Cmd+S`)

## 第五步：运行项目

### 5.1 训练模型

在 VS Code 终端中运行：

```bash
python main.py --config=config_fsc147.yaml --doc=my_first_experiment --train
```

参数说明：
- `--config=config_fsc147.yaml`：使用 FSC-147 配置
- `--doc=my_first_experiment`：实验名称（可以自定义）
- `--train`：训练模式

训练日志和模型将保存在：`exp/logs/my_first_experiment/`

### 5.2 测试/评估模型

1. 首先，在 `configs/config_fsc147.yaml` 中设置模型路径：
   ```yaml
   eval:
     checkpoint: ./exp/logs/my_first_experiment/model_epoch_4.pth
     sample: true
     image_folder: ./eval_results
   ```

2. 运行测试：
   ```bash
   python main.py --config=config_fsc147.yaml --doc=my_test --test
   ```

### 5.3 查看结果

- 训练日志：`exp/logs/my_first_experiment/stdout.txt`
- 模型检查点：`exp/logs/my_first_experiment/model_epoch_*.pth`
- 测试结果：终端会显示 MAE 和 MSE 指标

## 第六步：VS Code 推荐设置

### 6.1 安装推荐的扩展

在 VS Code 中安装以下扩展：
1. **Python** (Microsoft) - Python 语言支持
2. **Pylance** (Microsoft) - Python 智能提示
3. **GitLens** (可选) - Git 增强工具

安装方法：
- 点击左侧边栏的扩展图标（或按 `Ctrl+Shift+X`）
- 搜索并安装

### 6.2 选择 Python 解释器

1. 按 `Ctrl+Shift+P` 打开命令面板
2. 输入 `Python: Select Interpreter`
3. 选择您创建的虚拟环境（应该显示为 `./venv/bin/python` 或 `.\venv\Scripts\python.exe`）

## 常见问题解决

### 问题 1：找不到 Python 或 pip

**解决方案：**
- 确认 Python 已正确安装：`python --version`
- 如果提示找不到命令，重新安装 Python 并确保勾选 "Add Python to PATH"

### 问题 2：ModuleNotFoundError

**解决方案：**
- 确认虚拟环境已激活（终端前有 `(venv)` 标记）
- 重新安装依赖：`pip install -r requirements.txt`

### 问题 3：Failed to load image

**解决方案：**
- 检查 `config_fsc147.yaml` 中的 `data_path` 是否正确
- 确认数据集已下载并解压
- 确认路径中的 `images_384_VarV2/` 文件夹存在

### 问题 4：CUDA out of memory

**解决方案：**
- 在 `config_fsc147.yaml` 中减少 `batch_size`：
  ```yaml
  train:
    batch_size: 4  # 从 12 减少到 4
  ```

### 问题 5：Git 克隆速度慢

**解决方案：**
- 使用镜像地址（如果在中国）
- 或者使用浅克隆：`git clone --depth 1 https://github.com/...`

## 快速开始示例

完整的运行流程：

```bash
# 1. 克隆仓库
git clone https://github.com/RaoXuanlin/Class-agnostic-Few-shot-Object-Counting.git
cd Class-agnostic-Few-shot-Object-Counting

# 2. 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
pip install pycocotools

# 4. 修改配置文件
# 在 configs/config_fsc147.yaml 中设置您的数据集路径

# 5. 开始训练
python main.py --config=config_fsc147.yaml --doc=exp1 --train

# 6. 测试（训练完成后）
# 先在 config_fsc147.yaml 中设置 checkpoint 路径
python main.py --config=config_fsc147.yaml --doc=exp1_test --test
```

## 项目结构说明

```
Class-agnostic-Few-shot-Object-Counting/
├── configs/
│   ├── config.yaml              # COCO 数据集配置
│   └── config_fsc147.yaml       # FSC-147 数据集配置 ⭐
├── data/
│   ├── coco.py                  # COCO 数据加载器
│   └── fsc147.py                # FSC-147 数据加载器 ⭐
├── model/
│   └── CFOCNet.py               # 模型定义
├── main.py                      # 主程序入口
├── runner.py                    # 训练/测试逻辑
├── FSC147_USAGE.md              # FSC-147 英文使用指南 ⭐
├── FSC147_使用说明.md           # FSC-147 中文使用指南 ⭐
└── README.md                    # 项目说明
```

⭐ 标记的是新增的 FSC-147 相关文件

## 更多帮助

- **详细的 FSC-147 使用指南**：查看 `FSC147_使用说明.md`
- **英文文档**：查看 `FSC147_USAGE.md`
- **原项目 README**：查看 `README.md`

## 视频教程推荐

如果您是 VS Code 新手，推荐观看以下教程：
- VS Code Python 入门教程
- Git 和 GitHub 基础教程
- PyTorch 深度学习环境配置

祝您使用愉快！如有问题，欢迎在 GitHub Issues 中提问。
