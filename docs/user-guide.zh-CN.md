# ComfyOmni 用户指南

> Bring Comfy checkpoints to native Omni runtimes.

本指南描述当前 ComfyOmni 预发布版本（`0.2.0a1`，见 [`pyproject.toml`](../pyproject.toml)）目前已交付的
能力。它严格限定于已合并的代码和记录在 `docs/` 下的证据；它不描述尚未完成的工作。每一项能力声明
都在下方标注其所依据的文档。

<!-- guide-parity -->
> 英文与中文用户指南是一组同步的双语文档。它们携带**相同**的事实——名称、路径、字节数、摘要
> 完全一致，只有正文语言不同。当你修改某一份文件中的事实时，请同步更新另一份。

## ComfyOmni 是什么

ComfyOmni 是一个开源桥接项目，用于检查、转换、打包和验证 Comfy 生态 checkpoint，并将其交付给原生
Omni 运行时。其标语是 "Bring Comfy checkpoints to native Omni runtimes."。项目坚持离线转换，目标是
生成不可变、可验证的运行时包，而不是让推理 worker 在启动时解析任意 Comfy checkpoint。本指南是项目
[README](../README.md) 面向用户侧的补充文档；它聚焦于当前代码中真实存在的命令与 Python API，以及
这些 API 产出的原生包格式。

ComfyOmni 是独立的开源项目。除非另有明确说明，它不属于 ComfyUI、Comfy.org、vLLM、MiniMax 或这些
项目维护者的官方项目。

## 安全模型

ComfyOmni 将 checkpoint 转换与打包视为离线、有界、流式、fail-closed 的制品操作。

- **离线。** inspection、normalization、conversion 和 packaging 不与推理运行时宿主通信。模型工作
  只会在指定服务器的 Docker 容器内、对着只读源挂载执行，具体见
  [`docs/development/docker-first.md`](../docs/development/docker-first.md)。
- **有界。** inspection 只读取 safetensors 头部，从不读取 tensor payload。normalization 每次读取
  最多 8 MiB。ConvRot producer 将每个中间块处理的行数限制在计划的 `max_rows`。原生包链以有界的
  8 MiB 分块复制文件。
- **流式。** 大文件通过持有的、摘要绑定的描述符以有界分块方式处理，而不是加载进内存。包含六个组件的
  61,745,213,741 字节包以有界内存包络进行 stage 和发布（E3 运行在处理该 payload 时峰值 RSS 为
  52,367,360 字节，见
  [`docs/evidence/native-package-assembly-b47d084.md`](../docs/evidence/native-package-assembly-b47d084.md)）。
- **Fail-closed。** 每次拒绝都会在发布前或权重加载前中止。源是只读的：inspection 和 normalization
  从不以写模式打开源 checkpoint，服务器转换也会将源树只读挂载。发布是 staging-first 且 manifest-last：
  文件先复制到私有 staging 树，被独立重新读取并重新哈希，然后通过一次同父目录原子重命名发布，
  且 `h3-comfy-package.json` 作为唯一完成标记最后写入。现有输出路径永不被覆盖；缺少 manifest 的目录
  按定义属于未完成发布，会被保留用于诊断而非递归删除。

## 目前已交付的能力

| 能力 | 状态 | 证据 |
|---|---|---|
| 仅头部 checkpoint inspection（`comfy-omni inspect`） | 已交付 | [`docs/migration/checkpoint-inspection-e9cb011.md`](../docs/migration/checkpoint-inspection-e9cb011.md) |
| 摘要固定的文本编码器规范化（`comfy-omni normalize text-encoder`） | 已交付 | [`docs/migration/text-encoder-normalization.md`](../docs/migration/text-encoder-normalization.md) |
| 不可变原生源合同工作流（`comfy-omni contract` scan / draft / pin / list） | 已交付 | [`docs/migration/contract-workflows-e9cb011.md`](../docs/migration/contract-workflows-e9cb011.md) |
| ConvRot 数值与有界转换链（inverse-ConvRot、QKV 重排、有界 payload producer、不可变事务、manifest-last 发布） | 作为应用层链交付；无 `export-native` CLI | [`docs/migration/convrot-numerics-e9cb011.md`](../docs/migration/convrot-numerics-e9cb011.md) · [`docs/migration/convrot-payload-producers-e9cb011.md`](../docs/migration/convrot-payload-producers-e9cb011.md) · [`docs/migration/convrot-native-export-transaction-e9cb011.md`](../docs/migration/convrot-native-export-transaction-e9cb011.md) |
| 完整 Ref2VA 原生转换（服务器已验证） | 已验收的真实模型切片 | [`docs/evidence/ref2va-full-conversion-25ceccdd5468.md`](../docs/evidence/ref2va-full-conversion-25ceccdd5468.md) |
| 不可变原生包链（receipt → plan → verify → materialize → publish） | 已交付 | [`docs/migration/component-receipt-parsing-e9cb011.md`](../docs/migration/component-receipt-parsing-e9cb011.md) · [`docs/migration/native-package-planning-e9cb011.md`](../docs/migration/native-package-planning-e9cb011.md) · [`docs/migration/native-package-source-verification-e9cb011.md`](../docs/migration/native-package-source-verification-e9cb011.md) · [`docs/migration/native-package-materialization-e9cb011.md`](../docs/migration/native-package-materialization-e9cb011.md) · [`docs/migration/native-package-publication-e9cb011.md`](../docs/migration/native-package-publication-e9cb011.md) |
| 六个组件、61,745,213,741 字节的原生包，在 `srv-00` 上独立验证 | 已验证 | [`docs/evidence/native-package-assembly-b47d084.md`](../docs/evidence/native-package-assembly-b47d084.md) · [`docs/evidence/native-package-assembly-76e2ebb.md`](../docs/evidence/native-package-assembly-76e2ebb.md) |
| 单一惰性幂等 vLLM-Omni bootstrap（`plugin:register`） | 已交付 | [`docs/migration/vllm-omni-bootstrap-e9cb011.md`](../docs/migration/vllm-omni-bootstrap-e9cb011.md) |
| Fail-closed 运行时包合同（`validate_runtime_package`） | 已交付 | [`docs/migration/runtime-package-contract-e9cb011.md`](../docs/migration/runtime-package-contract-e9cb011.md) |

## CLI 命令参考

distribution 暴露 `comfy-omni` CLI（[`pyproject.toml`](../pyproject.toml) 的 `[project.scripts]`）。
以下命令是在 [`src/comfy_omni/cli/`](../src/comfy_omni/cli/) 中实现的命令。这些是容器内部命令；
依据 Docker-first 策略，它们应在构建好的镜像内运行，绝不能在宿主机安装或执行。

```text
comfy-omni --version
comfy-omni inspect CHECKPOINT.safetensors [--json]
comfy-omni normalize text-encoder SOURCE.safetensors DERIVED.safetensors [--json]
comfy-omni contract scan SOURCE.safetensors [--json]
comfy-omni contract draft SOURCE.safetensors -o DRAFT.json --generated-by OPERATOR [--json]
comfy-omni contract pin DRAFT.json --name PROFILE --reviewer REVIEWER \
  --evidence REVIEW.md [--contract-dir CONTRACTS] [--enforce-observed-schema] [--json]
comfy-omni contract list [--contract-dir CONTRACTS] [--json]
```

- `inspect` 接受一个或多个路径并输出仅头部的 inspection。它拒绝非 `.safetensors` 文件，强制 64 MiB
  头部上限与 100,000 tensor 上限，并在出现未索引的尾部字节时以退出码 `2` 和稳定原因
  `safetensors-unindexed-trailing-bytes` 拒绝。它从不读取 tensor payload，也从不导入 Torch 或 vLLM。
- `normalize text-encoder` 应用唯一授权的规范化 profile
  （[`docs/migration/text-encoder-normalization.md`](../docs/migration/text-encoder-normalization.md)）。
  源必须存在且匹配固定的字节数与 SHA-256；目标父目录必须已存在；源与目标必须不同；并且目标及其
  兄弟文件 `DERIVED.safetensors.normalization.json` receipt 都不能存在。它通过 no-overwrite 链接发布
  制品与 receipt。
- `contract scan` 执行只读 census 与精确的三级匹配。仅在恰好有一个 L3 模板匹配时返回退出码 `0`，
  否则返回 `3`。它从不物化 tensor payload。
- `contract draft` 写入一个不可变的待审草稿，绑定源路径、大小与 SHA-256 值、census 摘要、模板身份
  和已安装的 generator 身份。草稿文件以排他方式创建，从不重写。
- `contract pin` 复核并发布一个不可变快照。它要求 `--contract-dir`（或旧式 `H3_FORGE_CONTRACT_DIR`
  兼容变量，仅由本 CLI 边界读取），强制 generator/reviewer 分离，并记录证据文件的字节摘要而非其路径。
  `--enforce-observed-schema` 将原本仅 census 的合同冻结其观测 schema。
- `contract list` 列出编译期以及显式加载的快照。只有当调用方显式传入 store 时外部合同才可见。

合同命令会计算源文件摘要，但不导入 Torch 或 vLLM。通用旧式转换与运行时命令仍不可用。

## 原生包格式

一个已发布的原生包是一个不可变目录树，符合 `h3-comfy-package/v3` 输出 schema（见
[`src/comfy_omni/conversion/packaging/planning.py`](../src/comfy_omni/conversion/packaging/planning.py)）。
它包含恰好六个组件目录，规范地放置在 `Ref2VA/` 下：

| 组件 | 位置 | 典型内容 |
|---|---|---|
| transformer | `Ref2VA/transformer/` | 原生 10-shard DiT checkpoint（10 个 `model-00000N-of-00010.safetensors` 文件，加上 `model.safetensors.index.json`、`config.patch.json`、`export.plan.json`、`manifest.json`） |
| text_encoder | `Ref2VA/text_encoder/` | 文本编码器的严格摘要固定规范化产物 |
| video_vae | `Ref2VA/video_vae/` | 视频 VAE 权重 |
| audio_vae | `Ref2VA/audio_vae/` | 音频 VAE 权重 |
| tokenizer | `Ref2VA/tokenizer/` | 官方 tokenizer 文件，在打包阶段获取 |
| processor | `Ref2VA/processor/` | 官方 processor 文件，在打包阶段获取 |

包根目录携带两个生成文件：

- **`model_index.json`** —— 宿主发现索引。它在独立的 staged 重读之后、manifest 之前写入。它携带
  `_class_name: "MiniMaxH3Pipeline"`、`_diffusers_version: "0.32.2"`、一个 `_minimax_h3` 路由块
  （`partition: "ref2va"`、`sigma_shift_scales: {"audio": 3.0, "video": 12.0}`、`schema_version: 1`，
  以及混合任务 `tasks` 列表 `ref2va|t2va|fl2va`），以及组件分类器对（`transformer` →
  `("diffusers", "MiniMaxH3DiTModel")`、`text_encoder` → `("transformers",
  "MiniMaxH3Qwen3VLHFEncoder")`、`video_vae` → `("diffusers", "MiniMaxH3VideoVAE")`、`audio_vae` →
  `("diffusers", "MiniMaxH3AudioVAE")`、`tokenizer` → `("transformers", "Qwen2TokenizerFast")`、
  `processor` → `("transformers", "Qwen3VLProcessor")`、`scheduler` → `null`）。
- **`h3-comfy-package.json`** —— 包 manifest。其 `package_manifest_sha256` 字段是同一文档恰好排除
  该字段后所得 SHA-256（即 self-digest）。它绑定 `schema`（`h3-comfy-package/v3`）、
  `plan_content_sha256`、`tool`、`host`（`adapter: "vllm-omni"`、`commit`）、`components` 数组、
  `source_files_sha256`、`staged_files_sha256`、`model_index_sha256`、`files` census
  （`path`/`sha256`/`size`）、`file_count`、`total_bytes`，以及 `routing` 块（`manifest`、
  `serving_entrypoint: "Ref2VA/"`、`resident_dit_count: 1`、`supported_tasks`）。

manifest 通过带 fsync 的排他只读创建最后写入。缺少 `h3-comfy-package.json` 的包目录按定义属于未完成
发布。

固定版本的宿主只通过根目录 `model_index.json` 的 `_class_name` 字段把模型目录解析为 pipeline 类
（见
[`docs/migration/native-package-publication-e9cb011.md`](../docs/migration/native-package-publication-e9cb011.md)）。

### 加载前消费者必须验证什么

在运行时加载权重之前，消费者必须用
`comfy_omni.integrations.vllm_omni.package_contract.validate_runtime_package` 验证包根目录。该验证器
host-free，并按如下顺序 fail-closed 拒绝：`package-binding` → `model-index` → `manifest` → `routing` →
`tree-census` → `file-verification` → `components`。它重新推导 manifest self-digest，重新计算
`model_index_sha256` 绑定，确认 model-index 与 manifest 的路由一致，对树做精确 census（拒绝链接与
特殊条目），对照 manifest 重新哈希每个声明的文件，并确认全部六个组件都已存在。任何拒绝都会在权重
加载前中止 pipeline，详见
[`docs/migration/runtime-package-contract-e9cb011.md`](../docs/migration/runtime-package-contract-e9cb011.md)。

## Python API 快速上手

离线包链位于 [`comfy_omni.conversion.packaging`](../src/comfy_omni/conversion/packaging/)，并通过包的
`__init__` 暴露。各步骤的规范意图：

```python
from pathlib import Path

from comfy_omni.artifacts.build_identity import installed_tool_identity
from comfy_omni.conversion.packaging import (
    parse_component_receipt,
    plan_native_package,
    verify_package_sources,
    materialize_package,
    publish_package,
)

tool = installed_tool_identity()                 # 必须报告 distribution 为 "comfy-omni"

receipts = tuple(
    parse_component_receipt(component, source_dir, tool)
    for component, source_dir in [
        ("transformer", "/components/transformer"),
        ("text_encoder", "/components/text_encoder"),
        ("video_vae", "/components/video_vae"),
        ("audio_vae", "/components/audio_vae"),
        ("tokenizer", "/components/tokenizer"),
        ("processor", "/components/processor"),
    ]
)                                                # 六个 receipt，精确树，无写入

plan = plan_native_package(
    receipts,
    vllm_omni_commit="17285c2f55a41bf15772676121814d59a60ace35",
)                                                # AUTHORIZED_PLAN，规范内容 SHA-256

verified = verify_package_sources(plan)          # 重新哈希每个源树，无写入
materialized = materialize_package(plan, Path("/out/native-package"))   # 仅私有 staging
publication = publish_package(plan, materialized)                       # manifest-last 原子发布
```

- `parse_component_receipt(component, source_dir, tool)` 在确定性的树 census 以及对每个文件的
  前后两次 pinned 哈希之后，返回一个不可变 `ComponentReceipt`。它从不写入。
- `plan_native_package(receipts, *, vllm_omni_commit)` 返回一个规范 `NativePackagePlan`。它要求恰好
  六个组件角色、所有组件共用一个相同的 producer 工具身份、固定的 `vllm_omni` 宿主 commit
  `17285c2f55a41bf15772676121814d59a60ace35`，以及规范的 `Ref2VA/` 组件放置。它不读写文件。
- `verify_package_sources(plan)` 重构计划，重新 census 每个源树，并对照计划重新哈希每个已计划的
  文件。
- `materialize_package(plan, output_dir)` 以有界分块将每个已计划的文件复制到一个私有同辈 staging 树，
  拒绝已存在或重叠的输出路径，并返回一个 `STAGED_VERIFIED` 句柄。它不发布任何内容。
- `publish_package(plan, materialization)` 先重新验证 staging 身份与树，再写入 `model_index.json` 与
  （最后写入的）manifest，并通过一次同父目录 `os.rename` 释放该树。现有输出路径永不被覆盖。

### 消费者侧验证

```python
from comfy_omni.integrations.vllm_omni.package_contract import validate_runtime_package

contract = validate_runtime_package("/data/models/comfy-omni/native-package")
print(contract.to_dict())  # status "RUNTIME_VERIFIED"
```

`validate_runtime_package(package_root, *, expected_class_name="MiniMaxH3Pipeline")` 返回一个冻结的
`RuntimePackageContract`，其 `to_dict()` 报告 `status: "RUNTIME_VERIFIED"`。它是消费者在任何权重加载
之前使用的、文档化的 host-free 入口点。

## vLLM-Omni 插件

distribution 注册一个 `vllm_omni.general_plugins` 入口点（[`pyproject.toml`](../pyproject.toml)），解析到
`comfy_omni.plugin:register`，它是
[`comfy_omni.integrations.vllm_omni.bootstrap.register`](../src/comfy_omni/integrations/vllm_omni/bootstrap.py)
之上的薄封装。

`register()` 会做以下事情，详见
[`docs/migration/vllm-omni-bootstrap-e9cb011.md`](../docs/migration/vllm-omni-bootstrap-e9cb011.md)：

- **观察宿主，绝不强制。** 仅当 `vllm_omni` 已驻留于 `sys.modules` 时才执行架构注册；registry
  子模块先从 `sys.modules` 解析，再回退到受保护的 import。缺失或驻留但不完整的宿主会静默延迟——
  无异常、无 latch——因此后续 `register()` 调用会重试。
- **注册声明式惰性字符串。** 它贡献 wire 兼容的架构键 `MiniMaxH3Pipeline` 与 `MiniMaxH3DensePipeline`，
  附带完整限定的 module/class 名与 `get_minimax_h3_post_process_func`。导入 `register()` 不导入任何
  pipeline 模块；宿主在模型加载时才解析它们。
- **每个进程只 latch 一次。** 注册由线程安全的 `NEW → REGISTERING → REGISTERED` 状态机保护，失败时
  重置为 `NEW`，并在宿主的每个进程加载形态下（process0、engine cores、workers）都安全。

它今天**不**注册任何 REST/API-server 钩子、路由、模型或运行时服务；`_is_root_process` 助手是面向未来
仅 API-server 连线的文档化钩子，当前不启动任何东西。

## 当前限制

以下是明确、当前的限制。它们尚未交付，不能从上述能力推断为可用。

- **尚未交付原生宿主加载或生成。** 运行时包合同（`validate_runtime_package`）与
  `H3ComfyMiniMaxH3Pipeline` 子类已经存在，但真实的宿主加载与最小生成仍在推进中（E4-S3，见
  [`docs/migration/runtime-package-contract-e9cb011.md`](../docs/migration/runtime-package-contract-e9cb011.md)）。
- **无 LoRA 生命周期。** LoRA 转换、preflight、激活与去激活尚未交付（issue #12）。
- **无热切换。** 单个宿主进程内的完整 DiT `A → B → A` 热切换尚未交付（issue #13）。
- **无 `export-native` CLI、无宽泛兼容性声明。** 完整 Ref2VA 转换已验收，但旧式 `export-native`
  命令被有意不暴露，且这些离线制品操作不是生产运行时或宽泛兼容性声明
  （[`docs/migration/convrot-native-export-plan-e9cb011.md`](../docs/migration/convrot-native-export-plan-e9cb011.md)）。
- **不重新分发外部资产。** tokenizer 与 processor 组件配置是在打包阶段获取的外部官方资产，本仓库
  不重新分发（[`docs/evidence/native-package-assembly-b47d084.md`](../docs/evidence/native-package-assembly-b47d084.md)）。
  模型权重永不提交到 Git，也永不包含在 wheel 中。

## 许可证与溯源策略

ComfyOmni 使用 [Apache License 2.0](../LICENSE)。迁移进去的模块是 Apache-2.0 `h3-forge` 项目的派生，
并在 `docs/migration/` 中携带 blob 级精确溯源：每份迁移记录都注明旧仓库、不可变源 commit、生产源码
blob、源码许可证与归属。第三方代码、fixture 与资产仍须遵守各自登记的归属与兼容许可证条款。模型
payload、生成的包、服务器证据与未追踪的旧文件永不由本仓库分发。
