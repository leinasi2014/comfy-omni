# ComfyOmni

> 为 vLLM-Omni 接入 ComfyUI 模型资产与可组合的生成能力。

[English](README.md) · **简体中文**

<!-- README_SYNC: overview -->
## 项目定位

ComfyOmni 是 vLLM-Omni 插件。首期目标是复用已有 ComfyUI/H3 模型与组件文件，在 RAM/VRAM 中加载、
卸载和切换。LoRA 组合、工具、节点工作流及其他模型家族放到后续，不阻塞首期交付。
正常启动和模型切换不应要求用户重新转换权重或组装模型目录。

本项目独立维护，并非 ComfyUI、vLLM 或 MiniMax 的官方项目。

<!-- README_SYNC: naming -->
## 命名

| 角色 | 名称/路径 |
|---|---|
| 产品 | `ComfyOmni` |
| 仓库与发行包 | `comfy-omni` |
| Python 包 | `comfy_omni` |
| 源码 | `src/comfy_omni/` |
| CLI | `comfy-omni` |
| 宿主入口点分组 | `vllm_omni.general_plugins` |

<!-- README_SYNC: status -->
## 当前能力

重构尚未完成。已有单一插件注册、组件目录/API，以及审计过的 legacy H3 curve-cache 兼容路径。
该旧兼容路径已有真实宿主对照证据，继续作为固定可用基线。beta4 DiT 单次 forward 通过，
不能证明完整的原始量化 H3 模型直接加载和生成已经通过。

现有 H3 原文件加载、RAM/VRAM 装卸和已登记模型切换已接入宿主及 API，并有小样本宿主回归。
固定大模型的完整生成与切换仍需真实 GPU 验收；接口和精度边界见
[H3 原文件运行说明](docs/guides/h3-original-files.md)。LoRA 生命周期、工具和节点属于后续工作。
[使用指南](docs/user-guide.zh-CN.md) 区分现有能力与目标。
实时进度和验收证据维护在 [任务 #4](https://github.com/leinasi2014/comfy-omni/issues/4)。

<!-- README_SYNC: goals -->
## 设计目标

- 引用已有 ComfyUI 组件文件，复用一套固定模型环境。
- 在 RAM/VRAM 中管理驻留、请求隔离、加载及失败恢复。
- 明确各格式支持范围，使用原生量化能力或有界的加载阶段内存适配。
- 保留旧实现已经验证的行为，按可用 H3 功能逐步交付。

内存 LoRA 组合、工具和节点工作流属于后续扩展，不是首期验收门槛。

<!-- README_SYNC: architecture -->
## 架构

应用服务协调来源绑定、组件加载器和模型会话；vLLM-Omni 适配层将其接入固定宿主。
已存在的导出包可以继续作为一种资产来源使用；运行时的目标不再以创建模型包为前提。

参见 [运行时架构](docs/architecture/README.md) 和
[H3 优先重构方案](docs/post-merge-refactoring-plan.md)。这些文档定义目标，不代表重构已经实现。

<!-- README_SYNC: milestones -->
## 交付顺序

| ID | 用户可见结果 | 验收条件 |
|---|---|---|
| H1 | 加载已有组件来源 | 固定 H3 加载回归、旧行为不退化、不新增整套模型副本 |
| H2 | 驻留组件复用和模型切换 | 同一控制实例及已有 worker 完成 A→B→A；核对身份、缓存失效、恢复和资源释放 |
| H3 | 真实宿主验证与交付 | 固定资产加载、生成及正常 worker 内切换通过；受影响的软件门禁通过并交付到 main |

这里是依赖顺序，不是实时状态表。[固定模型验证基线](docs/testing/model-validation-baseline.md)
规定资产与验证边界。普通 CI 不下载模型权重。
worker 重建只作为明确报告的恢复或降级路径，不算正常热加载通过。
LoRA 组合、工具和节点工作流不属于本期交付顺序。

<!-- README_SYNC: layout -->
## 仓库结构

```text
src/comfy_omni/    插件、契约、加载器、运行时及应用代码
tests/            单元、契约、集成、打包和宿主检查
docs/             当前设计、使用、测试及代码来源
scripts/          容器执行与仓库检查
.worktrees/       插件内部的开发隔离目录，忽略提交
```

<!-- README_SYNC: development -->
## 开发

`DOCKER_FIRST_POLICY: v1`

先读 [AGENTS.md](AGENTS.md)、[CONTRIBUTING.md](CONTRIBUTING.md) 和
[Docker 执行规范](docs/development/docker-first.md)。项目 Python、测试及模型执行均在 Docker 内完成，
宿主负责文件编辑及 Git、Docker、SSH 操作。

```bash
./scripts/docker.sh docs 3.13
./scripts/docker.sh quality 3.10
./scripts/docker.sh quality 3.13
./scripts/docker.sh package 3.12
```

PowerShell 使用 `scripts/docker.ps1`。真实运行回归在不同分支间复用既有的只读模型挂载；
数值与异常测试用小型样本。相关代码和输入没有变化时沿用有效证据，不为代码测试创建模型副本。

<!-- README_SYNC: contributing -->
## 参与贡献

通过短期分支和聚焦的 PR 交付到受保护的 `main`。中英文 README 同步更新，详见
[CONTRIBUTING.md](CONTRIBUTING.md)。

<!-- README_SYNC: license -->
## 许可证

[Apache License 2.0](LICENSE)。迁入代码及资产保留记录的来源归属和适用许可证。
