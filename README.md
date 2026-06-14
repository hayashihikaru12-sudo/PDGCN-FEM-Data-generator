# HDF5 FEM Data Generator

本代码库用于为 PDGCN 输入 HDF5 文件生成 FEniCS 温度监督数据。默认流程会复制原始 HDF5 到输出目录，并新增 `mesh/triangles` 与 `fem/*` 字段，不修改 `PDGCN/` 架构代码。

FEM 计算域为 `dynamic/xyz[t,:,:]` 描述的三维曲面网格。每个 HDF5 文件只构建一次 FEniCS 曲面 Mesh 和 FunctionSpace，时间推进时逐帧更新网格坐标；同一输入目录下复用首个文件的三角拓扑，后续文件只校验节点数量一致。

## VS Code 运行入口

推荐直接在 VS Code 中运行：

1. 打开 Run and Debug。
2. 选择 `Run HDF5 FEM Preprocess`。
3. 点击运行。

VS Code 会使用 `.vscode/settings.json` 中配置的解释器：

```text
/root/miniconda3/envs/fenics-env/bin/python
```

运行入口文件为：

```text
run_fem_preprocess.py
```

运行时终端是否输出求解监控信息由运行 JSON 显式控制：

```json
{
  "monitor": {
    "enabled": true,
    "step_stride": 1
  }
}
```

`enabled=false` 时不输出 `[mesh]` 和 `[step]` 监控行；`step_stride` 控制每隔多少步输出一次，其中第 1 步和最后 1 步总会输出。`[mesh]` 行记录曲面网格节点数、三角形数量、时间步数量和建网格耗时；`[step]` 行记录当前步数、物理时间、系数构造耗时、装配耗时、求解耗时、线性残差 `linear_residual_l2` 以及温度范围。

## 运行配置

HDF5 输入目录、修改后 HDF5 输出目录、PVD 可视化输出目录由单独的运行配置管理。程序会批量处理输入目录内匹配的所有 HDF5 文件：

```text
configs/vscode_run_config.example.json
```

示例：

```json
{
  "fem_config_path": "configs/fem_batch_config.example.json",
  "overwrite": true,
  "in_place": false,
  "jobs": [
    {
      "input_hdf5_dir": "HDF5_Files",
      "output_hdf5_dir": "outputs",
      "file_pattern": "*.h5",
      "recursive": false,
      "write_pvd": true,
      "pvd_dir": "outputs/pvd",
      "pvd_stride": 20
    }
  ]
}
```

其中：

- `fem_config_path`：材料参数、温度参数、求解器参数配置文件；可放在顶层作为数据集级默认值，也可在某个 `jobs[]` 内覆盖为目录级配置。
- `input_hdf5_dir`：待处理 HDF5 文件所在目录。
- `output_hdf5_dir`：写入 `mesh/triangles` 和 `fem/*` 后的新 HDF5 输出目录。
- `file_pattern`：匹配文件名，默认建议使用 `"*.h5"`。
- `recursive`：是否递归处理子目录。
- `write_pvd`：是否输出 PVD 可视化文件。
- `pvd_dir`：PVD 输出根目录；每个 HDF5 文件会生成独立子目录。
- `pvd_stride`：PVD 输出帧间隔；HDF5 中仍保存完整帧。

例如输入文件：

```text
HDF5_Files/case1_Q0_V20_dt0p05_F89.h5
```

会输出：

```text
outputs/case1_Q0_V20_dt0p05_F89_fem.h5
outputs/pvd/case1_Q0_V20_dt0p05_F89/temperature.pvd
```

## 可选 CLI

```bash
/root/miniconda3/envs/fenics-env/bin/python -m hdf5_fem.cli process \
  --input "HDF5_Files/*.h5" \
  --config configs/fem_batch_config.example.json \
  --output-dir outputs \
  --overwrite
```

校验输出：

```bash
/root/miniconda3/envs/fenics-env/bin/python -m hdf5_fem.cli validate \
  --input "outputs/*.h5"
```

## PVD 可视化配置

VS Code 入口优先使用 `configs/vscode_run_config.example.json` 中的 PVD 配置。CLI 入口也可以使用 `configs/fem_batch_config.example.json` 中的默认 PVD 配置：

```json
{
  "output": {
    "write_pvd": true,
    "pvd_dir": "outputs/pvd",
    "pvd_stride": 1
  }
}
```

开启后会生成 `outputs/pvd/<case_name>/temperature.pvd`，可用 ParaView 打开查看温度场。
