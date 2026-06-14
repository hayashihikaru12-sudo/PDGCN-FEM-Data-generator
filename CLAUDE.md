# 项目协作说明

## 项目目标

本代码库用于为 PDGCN 项目的 HDF5 输入文件执行数据预处理任务。处理对象是项目中的 HDF5 示例文件，输出结果应服务于 PDGCN 架构核心代码的训练、验证或数据读取流程。

## 目录结构

- `PDGCN/`：PDGCN 架构核心代码。
- `HDF5_Files/`：待修改的 HDF5 文件示例。

## 运行环境

本项目的 Python 命令统一使用 `fenics-env` conda 环境运行。

- 环境名称：`fenics-env`
- Python 解释器：`/root/miniconda3/envs/fenics-env/bin/python`
- Python 版本：`3.8.20`
- FEniCS/DOLFIN 版本：`2019.1.0`

优先使用显式解释器路径运行脚本：

```bash
/root/miniconda3/envs/fenics-env/bin/python test.py
```

也可以先激活环境后再运行：

```bash
conda activate fenics-env
python test.py
```

## 开发注意事项

- 当前项目使用经典 FEniCS/DOLFIN，不使用 `dolfinx`。
- 不要使用 base Python 环境执行测试或数据处理脚本，因为该环境不包含 FEniCS。
- 修改 HDF5 数据处理逻辑前，应先检查 `HDF5_Files/` 中示例文件的组、数据集、属性和数组形状。
- 涉及 PDGCN 输入格式时，应优先参考 `PDGCN/` 中已有的数据读取和模型输入约定，避免随意改变字段名、维度顺序或数据类型。