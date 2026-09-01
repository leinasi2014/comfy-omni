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

<!-- README_SYNC: status -->
## 项目状态

**早期重构／仓库基础建设阶段。** 当前仓库只包含已审查的架构骨架、开发规范和重构方案，尚未提供
可安装的 `comfy-omni` 版本或可用 CLI。当前骨架不能作为生产运行时或兼容性声明使用。

现有 H3 实现暂时保留在旧工作区；代码只有在完成来源、许可证、合同、测试和模块归属审计后才会迁移。

<!-- README_SYNC: goals -->
## 设计目标

- 在不把模型 payload 加载进 GPU 的情况下检查 checkpoint 结构。
- 通过显式、带版本的 profile 离线转换目标运行时不支持的表示。
- 当目标运行时支持相同原生表示时，保持 tensor 字节不变。
- 使用摘要、溯源、schema 校验和 fail-closed 发布生成不可变包。
- 禁止转换、反量化和映射工作进入推理热路径。
- 把运行时特定代码隔离到 `integrations/vllm_omni` 等 integration 中。
- 提供统一的 Python API、CLI、错误模型、测试基线和发布流程。

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

目标源码结构位于 [`src/comfy_omni`](src/comfy_omni)。内部模块必须遵守以上依赖方向；public facade
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
| M6 | 运行时验收 | 固定 vLLM-Omni adapter 通过 package、load、request、parity 和 fail-closed 验收门 | 计划中 |
| M7 | 开源预览版 | 发布许可证清晰、文档同步、制品可复现的 `0.2.0a1` 或 `0.2.0b1` | 计划中 |

详细顺序和出口条件见
[合并后重构与开源整理方案](docs/post-merge-refactoring-plan.md)。

<!-- README_SYNC: layout -->
## 仓库结构

```text
.
├── AGENTS.md                 # 架构、编码、测试和 Git 规范
├── CONTRIBUTING.md           # 贡献与 Pull Request 流程
├── README.md                 # 英文项目简介
├── README.zh-CN.md           # 简体中文项目简介
├── docs/                     # 设计文档、ADR 和公开证据索引
├── scripts/                  # 仓库检查脚本
└── src/comfy_omni/           # 新的模块化 Python 包骨架
```

骨架旁的旧仓库和本地证据已被明确忽略，不属于公开的 ComfyOmni 仓库。

<!-- README_SYNC: development -->
## 开发

修改代码前请依次阅读：

1. [`AGENTS.md`](AGENTS.md)
2. [`CONTRIBUTING.md`](CONTRIBUTING.md)
3. [`docs/post-merge-refactoring-plan.md`](docs/post-merge-refactoring-plan.md)

仓库当前尚不可安装。在基础建设里程碑期间，可执行的文档检查为：

```bash
python scripts/check_readme_sync.py
```

语言、lint、测试、打包和真实宿主门会随对应里程碑落地后成为强制检查。缺少必需工具属于失败，不能
伪装成通过。

<!-- README_SYNC: contributing -->
## 参与贡献

使用短期分支、Conventional Commits 和职责单一的 Pull Request。公开信息发生变化时必须同步更新
两份 README。完整流程与质量清单见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

<!-- README_SYNC: license -->
## 许可证

ComfyOmni 使用 [Apache License 2.0](LICENSE)。迁移进入项目的第三方代码、fixture 和资产仍须遵守
各自登记的归属与兼容许可证条款。
