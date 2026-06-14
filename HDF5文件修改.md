# HDF5 文件修改方案

## 1. 目标

在不改变现有 PDGCN 输入字段的前提下，使用 FEniCS 在同一局部窗口网格上求解温度场，并将 FEM 温度监督数据写入 HDF5 文件。

本代码库只负责对 HDF5 文件进行修改预处理，包括规范化三角网格拓扑、生成 FEM 温度监督数据、写入新增字段和执行数据校验；不负责修改 PDGCN 架构、训练入口、损失函数或数据加载代码。

修改后的 HDF5 文件应同时满足：

- PDGCN 原有读取逻辑仍可正常使用；
- FEM 温度值可按帧、按节点与 `dynamic/xyz` 对齐；
- PDGCN 项目后续可使用 `fem/temperature` 构造监督损失。

PDGCN 当前单层温度更新职责为：

$$
T_{n+1}
=
T_n
+
\Delta T_{\mathrm{source}}
+
\Delta T_{\mathrm{inplane}}
$$

其中 `dynamic/Q` 只用于显式表面热源，不能加入 PDGCN 节点输入特征。PDGCN 节点输入仍保持：

$$
\left[x^*,y^*,z^*,f_x,f_y,f_z,T^*\right]
$$

## 2. 现有 HDF5 结构约束

当前 PDGCN 读取所需字段包括：

```text
dynamic/xyz
dynamic/fiber
dynamic/normal
dynamic/Q
edge_index
boundary_nodes/upwind
boundary_nodes/downwind
boundary_nodes/side
path/heat_center_step_distance
```

HDF5 文件中的拓扑结构来自 Gmsh 离散，约定一定是三角网格。本方案不考虑非三角单元，也不需要重新构建网格。为便于 FEniCS 读取，三角单元连接关系统一整理为：

```text
mesh/triangles    [M, 3]
```

原文件中的 Gmsh 三角单元连接数据应直接复用并规范化到该字段；`mesh/triangles` 中的节点编号必须与 `dynamic/xyz[t, i, :]` 的节点编号完全一致。

## 3. FEniCS 求解模型

FEM 基准数据在局部窗口三角网格上求解嵌入三维空间的二维曲面内对流扩散热传导方程。计算域是由 `dynamic/xyz[t,:,:]` 给出的三维曲面，不再使用 $(x,y)$ 平面投影。HDF5 原始单位为 `mm`、`mm/s`、`W/mm^2`，求解前统一转换为 SI 单位。

### 3.1 参数来源与优先级

FEniCS 求解使用真实物理量。参数来源分为两类：

1. HDF5 文件中已有的工况参数，以 HDF5 文件为准，预处理脚本只读取并做单位转换。
2. HDF5 文件中没有的材料参数和求解控制参数，由外部 JSON 配置文件显式批量提供。

HDF5 优先读取的参数包括：

| 参数 | HDF5 来源 | 原始单位 | 求解单位 |
| --- | --- | --- | --- |
| 节点坐标 | `dynamic/xyz` | `mm` | `m` |
| 纤维方向 | `dynamic/fiber` | 无量纲方向 | 单位方向 |
| 法向方向 | `dynamic/normal` | 无量纲方向 | 单位方向 |
| 表面热流 | `dynamic/Q` | `W/mm^2` | `W/m^2` |
| 扫描速度大小 | 根属性 `velocity_speed` | `mm/s` | `m/s` |
| 扫描速度方向 | 根属性 `velocity_direction_local` | 无量纲方向 | 单位方向 |
| 热源步长距离 | `path/heat_center_step_distance` | `mm` | `m` |
| 边界节点 | `boundary_nodes/*` | 节点编号 | 节点编号 |
| 三角单元 | `mesh/triangles` 或等价 Gmsh 单元字段 | 节点编号 | 节点编号 |

外部 JSON 批量配置至少包含：

```json
{
  "material": {
    "rho": 1575.0,
    "Cp": 1600.0,
    "K_parallel": 5.9,
    "k_ratio": 0.05,
    "heat_source_effective_thickness": 0.00015,
    "heat_source_absorptivity": 1.0
  },
  "temperature": {
    "T_amb": 120.0,
    "unit": "degC"
  },
  "solver": {
    "time_scheme": "backward_euler",
    "linear_solver": "default"
  }
}
```

若同一参数同时出现在 HDF5 和 JSON 中，应按以下规则处理：

- 工况参数以 HDF5 为准，例如速度、热流、坐标、边界节点和时间步来源；
- 材料参数以 JSON 为准，例如 $\rho$、$C_p$、$K_{\parallel}$、$r_K$、$h_{\mathrm{eff}}$ 和 $\eta$；
- 若 JSON 中配置了用于校验的速度或时间步，只能用于一致性检查，不能覆盖 HDF5 原值。

时间步长取：

$$
\Delta t
=
\frac{\Delta s_{\mathrm{heat}}}{v_{\mathrm{scan}}}
$$

其中：

- $\Delta s_{\mathrm{heat}}$ 来自 `path/heat_center_step_distance`，由 `mm` 转为 `m`；
- $v_{\mathrm{scan}}$ 来自根属性 `velocity_speed`，由 `mm/s` 转为 `m/s`。

温度控制方程采用：

$$
\frac{\partial T}{\partial t}
+
\mathbf v_t \cdot \nabla T
=
\nabla \cdot
\left(
\boldsymbol{\alpha}
\nabla T
\right)
+
\frac{\eta q''}{\rho C_p h_{\mathrm{eff}}}
$$

其中：

- $\mathbf v_t$ 为局部曲面切向扫描速度；
- $q''$ 为表面热流，由 `dynamic/Q` 从 `W/mm^2` 转为 `W/m^2`；
- $\eta$ 为热源吸收率；
- $\rho$ 为密度；
- $C_p$ 为比热容；
- $h_{\mathrm{eff}}$ 为热源等效热容量厚度；
- $\boldsymbol{\alpha}$ 为热扩散张量。

若使用物理导热系数 $\mathbf K$，则：

$$
\boldsymbol{\alpha}
=
\frac{\mathbf K}{\rho C_p}
$$

材料各向异性导热张量由纤维方向和曲面法向共同决定。先将单元平均法向归一化为 $\mathbf n$，切平面投影矩阵为：

$$
\mathbf P
=
\mathbf I
-
\mathbf n\mathbf n^{\mathsf T}
$$

将纤维方向投影到切平面后归一化：

$$
\mathbf f_t
=
\frac{\mathbf P\mathbf f}{\|\mathbf P\mathbf f\|}
$$

然后构造三维嵌入空间中的曲面切向导热张量：

$$
\mathbf K
=
K_{\parallel}
\mathbf f_t\mathbf f_t^{\mathsf T}
+
K_{\perp}
\left(
\mathbf P
-
\mathbf f_t\mathbf f_t^{\mathsf T}
\right)
$$

$$
K_{\perp}
=
r_K K_{\parallel}
$$

其中 $r_K$ 对应 PDGCN 配置中的 `k_ratio`。

扫描速度也投影到局部曲面切平面：

$$
\mathbf v_t
=
\mathbf v
-
\left(
\mathbf v\cdot\mathbf n
\right)
\mathbf n
$$

边界条件与 PDGCN 训练保持一致：

$$
T=T_{\mathrm{amb}},
\quad
\mathbf{x}\in\Gamma_{\mathrm{upwind}}\cup\Gamma_{\mathrm{side}}
$$

$$
\boldsymbol{\alpha}\nabla T\cdot\mathbf n=0,
\quad
\mathbf{x}\in\Gamma_{\mathrm{downwind}}
$$

推荐使用后向欧拉时间离散。给定 $T^n$，求 $T^{n+1}$，弱形式为：

$$
\int_{\Omega}
\frac{T^{n+1}-T^n}{\Delta t}
w\,d\Omega
+
\int_{\Omega}
\left(
\mathbf v_t\cdot\nabla T^{n+1}
\right)
w\,d\Omega
+
\int_{\Omega}
\left(
\boldsymbol{\alpha}\nabla T^{n+1}
\right)
\cdot
\nabla w\,d\Omega
=
\int_{\Omega}
\frac{\eta q''}{\rho C_p h_{\mathrm{eff}}}
w\,d\Omega
$$

## 4. 新增 HDF5 字段

保持现有字段不变，只新增 `mesh/` 和 `fem/` 相关字段。

| 字段 | 形状 | 类型 | 说明 |
| --- | --- | --- | --- |
| `mesh/triangles` | `[M, 3]` | `int64` | FEniCS 三角单元拓扑，节点编号与 `dynamic/xyz` 对齐。 |
| `fem/time` | `[T]` | `float64` | FEM 温度对应的物理时间，单位 `s`。 |
| `fem/temperature` | `[T, N, 1]` | `float32` | FEM 求解得到的节点温度。 |
| `fem/temperature_unit` | 标量字符串 | string | 推荐写入 `"degC"`；若写入 `"K"`，必须在元数据中说明。 |
| `fem/valid_mask` | `[T, N, 1]` | `uint8` | 可选字段，`1` 表示该节点该帧温度有效。 |
| `fem/metadata_json` | 标量字符串 | string | FEM 参数、单位、边界条件、求解器和数据来源说明。 |

`fem/temperature` 必须满足：

$$
T_{\mathrm{FEM}}[t,i,0]
\leftrightarrow
dynamic/xyz[t,i,:]
$$

即同一时间帧、同一节点编号表示同一个物理节点。

## 5. 写入流程计划

### 5.1 前置检查

1. 读取 HDF5 文件，确认 `dynamic/xyz`、`dynamic/fiber`、`dynamic/normal`、`dynamic/Q` 形状一致。
2. 读取 `edge_index` 和三类边界节点，确认节点编号范围为 `[0,N-1]`。
3. 检查 HDF5 中的 Gmsh 三角单元连接数据。
4. 若三角单元连接数据的字段名或位置与本方案不同，将其规范化写入 `mesh/triangles`，并在 `fem/metadata_json` 中记录来源。
5. 读取外部 JSON 批量配置，确认材料参数和求解控制参数完整。
6. 若 JSON 中提供了速度、时间步等校验值，只与 HDF5 原值比较，不覆盖 HDF5 数据。

### 5.2 构建 FEniCS 网格

1. 对同一数据集目录，首个 HDF5 文件读取或恢复 `mesh/triangles`，后续 HDF5 文件只检查节点数量一致。
2. 每个 HDF5 文件只构建一次 FEniCS 网格，网格拓扑维度为 2，几何维度为 3。
3. 首帧将 `dynamic/xyz[0, :, :]` 从 `mm` 转为 `m` 后建立三维曲面网格。
4. 时间推进时逐帧用 `dynamic/xyz[n, :, :]` 原地更新 FEniCS `mesh.coordinates()`，不重新构建网格和 `FunctionSpace`。
5. 根据 `boundary_nodes/upwind`、`boundary_nodes/side`、`boundary_nodes/downwind` 标记边界节点；Dirichlet 约束按节点对应自由度施加。

### 5.3 时间推进

推荐约定：

$$
T_{\mathrm{FEM}}[0,i,0]
=
T_{\mathrm{amb}}
$$

然后对 $n=0,\dots,T-2$ 执行：

1. 读取第 $n$ 帧的 `dynamic/Q[n]`，转换为 $q''_n$；
2. 读取第 $n$ 帧的三维坐标、法向和纤维方向，更新曲面几何并构造 $\mathbf v_{t,n}$、$\mathbf K_n$ 和 $\boldsymbol{\alpha}_n$；
3. 使用 $T^n$ 作为初值求解 $T^{n+1}$；
4. 将结果写入 `fem/temperature[n+1, :, 0]`。

时间数组为：

$$
fem/time[n]
=
n\Delta t
$$

若 HDF5 根属性已有 `time_step`，应与上式计算结果做一致性检查。

### 5.4 写入元数据

`fem/metadata_json` 至少记录：

```json
{
  "solver": "fenics-dolfin",
  "equation": "transient convection diffusion heat equation",
  "temperature_unit": "degC",
  "coordinate_unit_input": "mm",
  "coordinate_unit_solver": "m",
  "heat_flux_unit_input": "W/mm^2",
  "heat_flux_unit_solver": "W/m^2",
  "dt_source": "path/heat_center_step_distance / velocity_speed",
  "mesh_geometry": "3d_surface",
  "mesh_build_count": 1,
  "mesh_topology_scope": "dataset_directory",
  "topology_validation": "node_count_only",
  "parameter_priority": {
    "case_parameters": "HDF5",
    "material_parameters": "external_json",
    "solver_parameters": "external_json"
  },
  "external_config": {
    "path": "configs/fem_batch_config.json",
    "material_keys": [
      "rho",
      "Cp",
      "K_parallel",
      "k_ratio",
      "heat_source_effective_thickness",
      "heat_source_absorptivity"
    ]
  },
  "boundary_condition": {
    "upwind": "Dirichlet T_amb",
    "side": "Dirichlet T_amb",
    "downwind": "zero Neumann"
  },
  "node_alignment": "fem/temperature[t,i,0] matches dynamic/xyz[t,i,:]",
  "requires_mesh_triangles": true
}
```

## 6. 与 PDGCN 监督训练的对接说明

本节只说明新增 HDF5 字段如何被 PDGCN 使用，不属于本代码库的实施范围。本代码库的交付物是包含 `fem/temperature` 等字段的 HDF5 文件。

训练时从 `fem/temperature` 读取真实温度，并按 PDGCN 配置转换为无量纲温度：

$$
T^*
=
\frac{T-T_{\mathrm{amb}}}{\Delta T_0}
$$

单步 teacher forcing 阶段使用：

$$
T_{\mathrm{input},n}^{*}
=
T_{\mathrm{FEM},n}^{*}
$$

先施加显式热源：

$$
T_{\mathrm{source},n}^{*}
=
T_{\mathrm{FEM},n}^{*}
+
\Delta T_{\mathrm{source},n}^{*}
$$

再由 PDGCN 预测无源面内温度增量：

$$
\Delta T_{\mathrm{inplane},n}^{*,\mathrm{pred}}
=
\mathrm{PDGCN}
\left(
\mathbf{x}_n,
\mathbf{e}_n,
T_{\mathrm{source},n}^{*}
\right)
$$

最终预测温度：

$$
T_{\mathrm{pred},n+1}^{*}
=
T_{\mathrm{source},n}^{*}
+
\Delta T_{\mathrm{inplane},n}^{*,\mathrm{pred}}
$$

温度监督损失为：

$$
\mathcal L_T
=
\frac{1}{|\Omega|}
\sum_{i\in\Omega}
\left(
T_{\mathrm{pred},n+1,i}^{*}
-
T_{\mathrm{FEM},n+1,i}^{*}
\right)^2
$$

总损失建议为：

$$
\mathcal L
=
\lambda_T\mathcal L_T
+
\lambda_{\mathrm{pde}}\mathcal L_{\mathrm{pde}}
+
\lambda_{\mathrm{out}}\mathcal L_{\mathrm{out}}
+
\lambda_{\mathrm{smooth}}\mathcal L_{\mathrm{smooth}}
$$

初始建议：

```text
lambda_T       = 1.0
lambda_pde     = 0.1 ~ 1.0
lambda_out     = 沿用当前配置
lambda_smooth  = 沿用当前配置
```

## 7. 校验标准

HDF5 写入后需要检查：

1. `fem/temperature.shape == dynamic/xyz.shape[:2] + (1,)`。
2. `fem/time.shape[0] == dynamic/xyz.shape[0]`。
3. `mesh/triangles` 中所有节点编号均在 `[0,N-1]` 范围内。
4. `fem/temperature` 中没有 `NaN` 或 `Inf`。
5. `upwind` 和 `side` 节点满足 Dirichlet 温度约束。
6. `Q=0` 工况下，若初温为 $T_{\mathrm{amb}}$，温度应保持接近环境温度：

$$
\max_{t,i}
\left|
T_{\mathrm{FEM},t,i}
-
T_{\mathrm{amb}}
\right|
\le
\varepsilon_T
$$

7. 有热源工况下，峰值温度、峰值位置和尾迹形态应与物理预期一致。

## 8. 推荐实施顺序

1. 确认或规范化数据集目录首个 HDF5 文件中的 `mesh/triangles`。
2. 编写外部 JSON 批量配置文件，显式给出材料参数和求解控制参数。
3. 编写 FEniCS 三维曲面求解脚本，速度、热流、时间步、坐标和边界节点以 HDF5 为准。
4. 先在 `Q=0` 示例上验证恒温保持。
5. 生成 `fem/time`、`fem/temperature`、`fem/valid_mask` 和 `fem/metadata_json`。
6. 编写 HDF5 结构校验脚本。
7. 批量处理 `HDF5_Files/` 中待修改的 HDF5 文件。
8. 输出处理日志，记录每个文件的新增字段、外部 JSON 材料参数、HDF5 工况参数和校验结果。

PDGCN 架构、HDF5 读取逻辑和训练损失的修改由 PDGCN 项目侧完成，不在本代码库范围内。
