# ComfyOmni

> Bring Comfy checkpoints to native Omni runtimes.

[English](README.md) · **简体中文**

<!-- README_SYNC: overview -->
## 项目简介

ComfyOmni 是一个开源桥接项目，用于检查、转换、打包和验证 Comfy 生态 checkpoint，并将其交付给
原生 Omni 运行时。项目坚持离线转换，目标是生成不可变、可验证的运行时包，而不是让推理 worker
在启动时解析任意 Comfy checkpoint。

项目正在基于已经合并的 `h3-forge` 代码库重新构建，最终对外只提供一个 distribution、一个 Python
包、一个 CLI 和一个运行时插件入口。首个计划完成真实验收的集成是固定版本的 vLLM-Omni；其他运行时
必须提供独立 adapter 和验收证据。

ComfyOmni 是独立的开源项目。除非另有明确说明，它不属于 ComfyUI、Comfy.org、vLLM、MiniMax 或
这些项目维护者的官方项目。

<!-- README_SYNC: naming -->
## 命名与源码目录

| 身份 | 名称／路径 |
|---|---|
| 产品与文档标题 | `ComfyOmni` |
| GitHub 仓库及 clone 后的目录 | `comfy-omni` |
| PyPI distribution | `comfy-omni` |
| CLI 命令 | `comfy-omni` |
| Python import package | `comfy_omni` |
| 可导入源码路径 | `src/comfy_omni/` |

因此 clone 后的完整关系是 `comfy-omni/src/comfy_omni`。仓库内部不会再重复创建
`comfy-omni/`：`src/` 用于隔离可导入源码与项目文件；`comfy_omni` 使用下划线，是因为 Python
import package 应当是合法标识符。GitHub 可能把只有一个子目录的连续路径压缩成一行
`src/comfy_omni` 显示。

这个结构遵循 Python Packaging User Guide 的
[src-layout 说明](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
以及
[distribution package 与 import package 的区别](https://packaging.python.org/en/latest/discussions/distribution-package-vs-import-package/)。

<!-- README_SYNC: status -->
## 项目状态

**Pre-alpha 迁移阶段。** 当前仓库包含轻量包／插件注册、严格 checkpoint 检查、固定文本编码器
规范化、不可变源合同、有界离线 ConvRot 导出，以及原生包组装和验证。H3 集成还包含面向明确审计的
`h3-forge@e9cb011` v3 配方的[曲线缓存兼容适配器](docs/migration/h3-cache-runtime-e9cb011.md)，
与现有 native v3 布局并存。适配器保留旧制品和 wire 标识，并区分制品生产者与当前执行的
ComfyOmni wheel 身份。

各项能力分别需要代表性的真实宿主验收。当前验收与发布决定位于
[运行时 Issue](https://github.com/leinasi2014/comfy-omni/issues/11) 和
[项目 Epic](https://github.com/leinasi2014/comfy-omni/issues/4)。HTTP API 迁移、其他运行时 profile、
LoRA 生命周期和完整 DiT 热切换仍是独立工作；当前候选不声明广泛运行时兼容或预览版整体完成。

<!-- README_SYNC: goals -->
## 设计目标

- 在不把模型 payload 加载进 GPU 的情况下检查 checkpoint 结构。
- 通过显式、带版本的 profile 离线转换目标运行时不支持的表示。
- 当目标运行时支持相同原生表示时，保持 tensor 字节不变。
- 使用摘要、溯源、schema 校验和 fail-closed 发布生成不可变包。
- 禁止转换、反量化和映射工作进入推理热路径。
- 把运行时特定代码隔离到 `integrations/vllm_omni` 等 integration 中。
- 提供统一的 Python API、CLI、错误模型、测试基线和发布流程。
- 在修改运行时前证明 LoRA 兼容性，并对不支持的底模／adapter 组合 fail closed。
- 在同一个宿主进程完成完整 DiT 的 A → B → A 热切换，验证模型身份、缓存失效、失败恢复和资源回收。

<!-- README_SYNC: architecture -->
## 架构

```text
CLI / HTTP API / runtime integrations
                 |
                 v
            application
             /        \
     conversion      runtime
             \        /
          artifacts / contracts / domain
                       |
                       v
                      core
```

目标源码结构位于 [`src/comfy_omni`](src/comfy_omni)。[已验证的目标架构图](docs/architecture/README.md)
展示依赖与发布边界，并明确区分已经证明的 M0 能力和计划工作。内部模块必须遵守以上依赖方向；public facade
只服务外部消费者，不能成为仓内模块绕过分层的依赖捷径。

<!-- README_SYNC: milestones -->
## 大方向里程碑

| ID | 里程碑 | 结果 | 状态 |
|---|---|---|---|
| M0 | 仓库基础与公开审计 | 独立仓库、双语文档、消费者清单、许可证／历史审计和冻结基线 | 进行中 |
| M1 | 可信发布门 | pytest 完整收集、Ruff、可复现 sdist/wheel、包资源检查和 clean-install smoke | 计划中 |
| M2 | ComfyOmni 原子迁移 | distribution、import package、CLI、plugin target 和权威文档同步迁移，不擅自修改 wire contract | 计划中 |
| M3 | 单一 bootstrap 与依赖方向 | 一个惰性、幂等 bootstrap；消除插件递归、public facade 反向依赖和 import cycle | 计划中 |
| M4 | 转换模块化 | inspection、contract、mapping、exporter、LoRA conversion、packaging 和 publication 各有明确 owner | 计划中 |
| M5 | 运行时模块化 | runtime service 与离线 conversion、vLLM-Omni 宿主子类彻底分离 | 计划中 |
| M6 | 运行时验收 | 固定 vLLM-Omni adapter 与摘要绑定模型集通过 package、load、request、LoRA preflight／生命周期、完整 DiT A → B → A 热切换、parity 和 fail-closed 验收门 | 计划中 |
| M7 | 开源预览版 | 发布许可证清晰、文档同步、制品可复现的 `0.2.0a1` 或 `0.2.0b1` | 计划中 |

详细顺序和出口条件见
[合并后重构与开源整理方案](docs/post-merge-refactoring-plan.md)。
外部测试资产与证据规则固定在
[模型验证基线](docs/testing/model-validation-baseline.md)；模型 payload 永远不会存入本仓库，也不会由
普通 CI 下载。

<!-- README_SYNC: layout -->
## 仓库结构

```text
.
├── AGENTS.md                 # 架构、编码、测试和 Git 规范
├── CONTRIBUTING.md           # 贡献与 Pull Request 流程
├── Dockerfile                # 开发、质量、打包与 CLI 镜像目标
├── pyproject.toml             # distribution 元数据、CLI、插件入口和工具配置
├── README.md                 # 英文项目简介
├── README.zh-CN.md           # 简体中文项目简介
├── docs/                     # 设计文档、ADR 和公开证据索引
├── scripts/                  # 仓库检查脚本
├── src/comfy_omni/           # 新的模块化 Python 包
└── tests/                    # unit、contract、integration、packaging 和 host 测试层
```

旧仓库和本地证据保留在这个独立 Git 根之外，不属于公开的 ComfyOmni 仓库；只有通过审计迁移后，
相关内容才可以进入本仓库。

<!-- README_SYNC: development -->
## 开发

`DOCKER_FIRST_POLICY: v1`

修改代码前请依次阅读：

1. [`AGENTS.md`](AGENTS.md)
2. [`CONTRIBUTING.md`](CONTRIBUTING.md)
3. [`docs/post-merge-refactoring-plan.md`](docs/post-merge-refactoring-plan.md)
4. [`docs/development/docker-first.md`](docs/development/docker-first.md)

Docker 是唯一权威执行边界。禁止在宿主机安装 ComfyOmni、Python 依赖或模型／运行时依赖。宿主机只
负责文件编辑、Git／GitHub、Docker／Compose、SSH／SCP，以及维持容器边界所需的诊断。仓库门与当前
CLI 统一通过包装脚本执行：

```bash
./scripts/docker.sh docs 3.13
./scripts/docker.sh quality 3.10
./scripts/docker.sh quality 3.13
./scripts/docker.sh package 3.12
./scripts/docker.sh cli 3.13 --help
```

PowerShell 用户使用 `scripts/docker.ps1` 的同名 action。模型／checkpoint 命令必须使用构建好的镜像，
显式只读挂载输入，并单独挂载有边界的输出／证据目录，具体规则见 Docker-first 规范。本地没有 Docker
只代表本地门不可用，不允许回退到宿主 Python；仍必须取得可信 CI 与指定服务器 Docker 证据。

完成所需挂载声明后，以下是**容器内部**命令，绝不是宿主 shell 的安装或执行说明：

```text
comfy-omni inspect CHECKPOINT.safetensors --json
comfy-omni normalize text-encoder SOURCE.safetensors DERIVED.safetensors --json
comfy-omni contract scan SOURCE.safetensors --json
comfy-omni contract draft SOURCE.safetensors --generated-by OPERATOR -o DRAFT.json
comfy-omni contract pin DRAFT.json --name PROFILE --reviewer REVIEWER \
  --evidence REVIEW.md --contract-dir CONTRACTS
comfy-omni contract list --contract-dir CONTRACTS --json
```

目前提供项目身份、严格的 header-only inspection、
[唯一精确规范化 profile](docs/migration/text-encoder-normalization.md)，以及
[已审计的不可变合同工作流](docs/migration/contract-workflows-e9cb011.md)。合同命令会计算源文件摘要，但
不会物化 tensor payload，也不会导入 Torch／vLLM。只有调用方显式传入 store 时外部合同才可见；旧环境
变量仅由 CLI 兼容边界读取。通用旧转换／运行时命令仍不可用。快速、确定性的仓库检查在本地和 CI 的
Docker 中执行；GPU 与运行时验收只在指定服务器的 Docker 中针对摘要绑定资产运行。延后、缺失或目标
不同的检查都不能伪装成通过。

<!-- README_SYNC: contributing -->
## 参与贡献

使用短期分支、Conventional Commits 和职责单一的 Pull Request。公开信息发生变化时必须同步更新
两份 README。完整流程与质量清单见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

<!-- README_SYNC: license -->
## 许可证

ComfyOmni 使用 [Apache License 2.0](LICENSE)。迁移进入项目的第三方代码、fixture 和资产仍须遵守
各自登记的归属与兼容许可证条款。
