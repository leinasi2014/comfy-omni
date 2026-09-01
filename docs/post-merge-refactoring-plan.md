# ComfyOmni（原 h3-forge）合并后重构与开源整理方案

状态：仓库事实审查修订版

日期：2026-09-01
适用范围：M0.5 单仓单插件整合完成后的 `h3-forge` 主仓，以及向 `ComfyOmni` 的公开改名

## 1. 前提与定位

M0.5 已经完成发行边界上的合并：

- 一个 Git 仓库；
- 一个 `pyproject.toml`；
- 一个 Python 发行包 `h3-forge`；
- 一个 vLLM-Omni 插件入口 `h3_forge.plugin:register`；
- 原 `comfy-lora-hotswap` 已成为 `h3_forge.lora_hotswap`；
- 原 `h3-tools` 已成为 `h3_forge.tools`；
- 测试已迁入同一 `tests/` 树。

因此，本方案不再讨论“是否合并插件”，也不重新拆成多个发行包。本期目标是完成合并后的
内部模块治理：收敛历史插件机制、建立单向依赖、拆分超大模块、修复发布门，并把仓库整理成
可维护、可安装、可贡献的开源项目，同时完成从内部开发名 `h3-forge` 到公开产品名
`ComfyOmni` 的迁移。

### 1.1 已确定的公开命名

| 身份 | 目标值 |
|---|---|
| 项目名 | `ComfyOmni` |
| GitHub repository | `comfy-omni` |
| PyPI distribution | `comfy-omni` |
| Python import package | `comfy_omni` |
| CLI | `comfy-omni` |
| 标语 | `Bring Comfy checkpoints to native Omni runtimes.` |

命名含义：项目把 Comfy 生态 checkpoint 经过检查、转换、打包和验证，交付给原生 Omni
运行时。产品标题和文档品牌使用 `ComfyOmni`；GitHub 仓库、发行与命令使用小写 kebab-case
`comfy-omni`；代码标识使用 snake_case `comfy_omni`。

本命名决策只确定项目、distribution、import package、CLI 和标语。现有 HTTP URL、环境变量、
artifact schema、manifest 字段及已发布制品中的 `h3_forge`/`h3-comfy` 标识不自动随之改名；
它们属于独立 public contract，只有经过消费者清单、兼容方案和专项验收后才能变更。

本方案是实施路线图；稳定架构与项目约束仍以以下文档为准：

- 旧仓 `docs/architecture.md`：迁移前的稳定架构和运行时边界；
- 旧仓 `docs/project-binding.md`：迁移前的项目、候选、验证和发布约束；
- 旧仓 `docs/code-management.md`：现行行数、复杂度、规划和质量门；
- 旧仓 `docs/compatibility.md`：现行能力状态与 `SUPPORTED` 定义；
- 旧仓 `docs/runtime-parity.md`：真实运行时验收协议。

旧仓文件是当前私有迁移输入，不属于公开骨架。Phase 0 完成文档权威迁移时，必须把仍有效的内容
提炼进 ComfyOmni 自己的 `docs/`；公开文档不得依赖 sibling 仓的目录布局。

### 1.2 文档权威迁移

当前 `project-binding.md` 与 `architecture.md` 仍以 `h3-forge`、MiniMax H3 和
`h3_forge.plugin.register()` 为稳定事实。这与 ComfyOmni 的新名称及 family-agnostic 方向存在阶段性
冲突，不能通过“本方案引用旧权威、旧权威又否定本方案”的方式长期并存。

迁移规则：

1. Phase 0 生成 public contract inventory，并记录一份命名/范围 ADR；ADR 明确哪些 H3 合同继续有效、
   哪些只是首个 family adapter、哪些名称将在何时切换。
2. 发布门修复期间，现有权威文档继续描述当前可运行的 `h3-forge` 候选。
3. 改名 PR 在同一候选中更新 `project-binding.md`、`architecture.md`、README、包元数据和入口；从该
   commit 起，ComfyOmni 文档成为新事实源，H3 特化内容降为 adapter/compatibility 文档。
4. 在权威切换前，不得对外声称 ComfyOmni 已完成发布；切换后也不得让旧 `h3-forge` 标识重新成为
   Python 内部架构默认值。

标语中的 “native Omni runtimes” 是产品方向，不代表 `0.2.0` 已支持多个宿主。`0.2.0` 的首个且唯一
受验收 adapter 仍是 `UPSTREAM.toml` 固定的 vLLM-Omni；未来 runtime 必须通过新的 integration adapter
和各自真实宿主验收加入，不能复用 vLLM-Omni 的 `SUPPORTED` 结论。

## 2. 目标与非目标

### 2.1 目标

1. 保持单发行包和现有 HTTP wire compatibility，同时消除内部“子插件套子插件”的结构。
2. 建立可自动检查的单向模块依赖，消除 `public` 门面参与的循环依赖。
3. 将离线转换、运行时能力、vLLM-Omni 集成、HTTP API 和 CLI 分成清晰边界。
4. 拆分超大文件和超长函数，使职责、测试和变更影响范围可控。
5. 建立真实执行 pytest、lint、wheel 安装和资源验证的发布门。
6. 清理内部基础设施痕迹、历史流水账和授权不明的交付物，完成开源许可归属。
7. 在不改变现有 fail-closed、离线转换和“零上游源码 fork/patch”边界的前提下完成重构；当前
   对宿主 `_apply_lora` 的进程内 monkeypatch 必须显式登记、版本锁定并测试，不能称为“零宿主补丁”。
8. 将项目、distribution、import package 和 CLI 原子迁移到已确定的 ComfyOmni 命名。

### 2.2 非目标

- 不重新拆回多个仓库或多个 PyPI 发行包；
- 不在结构重构 PR 中增加新模型家族、新量化格式或新运行时功能；
- 不改变 `/v1/h3-forge/*`、`/v1/h3-tools/*`、`/v1/h3-comfy/loras`、`/v1/lora/*`
  的既有协议；
- 不借重构放宽合同、路径、摘要、来源或运行时校验；
- 不在一个 PR 中同时完成目录大迁移、行为修改和协议升级；
- 不以减少文件数或代码行数代替正确性和可验证性。

## 3. 当前基线与主要问题

2026-09-01 审查基线：

- `src/h3_forge` 下约 53,534 行 Python 生产代码；
- 13 个生产模块超过 1,000 行；
- 83 个函数超过现行建议上限 80 行，其中 23 个超过 150 行；
- AST 模块图存在 5 个循环依赖组：一个包含 `package_assembler`、`public` 和多数 LoRA 模块的
  13 模块 SCC，以及 `contracts.registry <-> contracts.snapshots`、
  `component_lora_seam <-> lora_hotswap.stack_loader`、`plugin <-> tools.plugin` 三个双模块 SCC；
  另有 `component_request_seam <-> dense_pipeline` 双模块 SCC（其中一条边是函数内延迟 import，
  结构检查不得漏掉）；
- 环境变量分散在多个模块中，公开配置缺少统一索引；
- `docs/refactor/` 已明确是 M0.5 前历史快照，不能继续作为当前模块事实源。

以上数字是当前 commit 的可复核快照，不是永久基线。Phase 0 必须把 LOC、超标函数、SCC、pytest
收集数和 wheel 内容生成到机器可读报告并绑定 commit；后续结构门比较报告，不手工维护数字。

### 3.1 合并后的历史机制仍然并存

代码已经位于同一个 `h3_forge` 包中，但仍保留：

- `h3_forge._import_hook`；
- `h3_forge.lora_hotswap.import_hook`；
- `h3_forge.tools._import_hook`；
- core、LoRA、tools 各自的注册状态和重试逻辑；
- `tools.plugin -> h3_forge.plugin -> tools.plugin` 的受 latch 保护回调；
- `standalone`、`companion`、`sibling plugin` 等旧模块语义。

这些机制在归并时用于保持行为等价，但不应成为合并后的长期架构。

### 3.2 公共门面参与内部依赖

当前 `h3_forge.public` 同时承担外部门面、内部共享入口和高层 package API，并导出若干私有
`_validate_*` 实现。LoRA/tools 模块从该门面获取底层能力，而 package assembler 又反向依赖
LoRA public，形成跨层循环。

长期规则应是：public facade 只供外部消费者使用，仓内模块必须直接依赖所属低层合同或端口，
不得通过 public facade 绕回高层实现。

### 3.3 发布门不能完整代表仓库状态

当前 `scripts/check.py` 使用 `unittest.defaultTestLoader.discover()`，但合并进来的 tools、LoRA
以及部分 fsops 测试使用模块级 pytest 函数。审查时至少 18 个测试文件中的 483 个模块级 pytest
测试函数不会被 unittest discovery 收集，权威检查因此不能代表完整 pytest 基线。

同时，缺少 Ruff 或 build frontend 时会报告 `NOT_CONFIGURED` 并继续成功；wheel 检查只验证
CLI/plugin 两个模块和 entry point，没有验证 mapping packs、`target_matrix.json`、LICENSE/NOTICE
以及安装后的实际命令。

### 3.4 仓库内容与公开发行内容混杂

当前仓库同时包含：

- 可发行源码和测试；
- 内部验证主机专用部署脚本；
- 内部绝对路径、主机命名和端口；
- `issues/`、`prs/`、`docs/research/` 中的过程记录；
- 历史吸收快照；
- 验证截图和本地构建产物。

这些内容必须按“产品源码、贡献者文档、可复现证据、私有运维资料”重新分类。

## 4. 目标架构

### 4.1 目标目录

以下目录是演进终态，不要求一次性移动完成：

```text
src/comfy_omni/
├── __init__.py
├── plugin.py                         # 稳定轻量 entry-point shim，仅委托 integration bootstrap
├── public.py                         # 唯一外部 Python 门面，仅 re-export 稳定 API
├── core/                             # 标准库叶子：错误、基础类型、端口协议
│   ├── errors.py
│   ├── models.py
│   └── typing.py
├── domain/                           # 纯领域：component、tensor、plan、coverage，不接触 I/O/宿主
│   ├── components.py
│   ├── tensors.py
│   ├── plans.py
│   └── coverage.py
├── artifacts/                        # 文件、摘要、严格 JSON、safetensors、溯源
│   ├── fs.py
│   ├── hashing.py
│   ├── canonical_json.py
│   ├── safetensors.py
│   ├── provenance.py
│   └── contract_store.py             # snapshot 的读取、O_EXCL 发布与路径安全
├── contracts/                        # 纯合同模型/注册表/schema，不读取环境或文件系统
│   ├── model.py
│   ├── registry.py
│   ├── snapshot_model.py
│   ├── schemas.py
│   └── templates/
├── conversion/                       # 只负责离线观察、计划、转换和验证
│   ├── inspection/
│   ├── contract_workflows/            # scan/draft/pin/activate 用例实现
│   ├── profiles/                      # converter profile model、registry、builtins
│   ├── oracle/
│   ├── mapping/
│   ├── exporters/
│   ├── packaging/
│   └── lora/
├── runtime/                          # 服务期能力，不包含 FastAPI 和 CLI
│   ├── h3/
│   ├── components/
│   ├── loras/
│   ├── tools/
│   ├── hotel/
│   ├── budget/
│   └── acceleration/
├── application/                      # CLI/API/integration 共享的用例编排，不含界面代码
│   ├── conversion.py
│   ├── contracts.py
│   ├── packages.py
│   ├── components.py
│   └── hotel.py
├── integrations/
│   └── vllm_omni/                    # 唯一宿主集成边界
│       ├── bootstrap.py
│       ├── registry.py
│       ├── import_hook.py
│       ├── host_adapters.py
│       └── pipelines/
├── api/                              # FastAPI 路由、wire schema、错误映射
│   ├── router.py
│   ├── errors.py
│   └── routes/
│       ├── components.py
│       ├── hotel.py
│       ├── loras.py
│       └── tools.py
├── validation/                       # 支持声明、preflight、parity 和发布验收
│   ├── compatibility.py
│   ├── preflight/
│   └── parity/
├── cli/
│   ├── __init__.py
│   ├── main.py
│   ├── parser.py
│   ├── output.py
│   └── commands/
└── resources/
    ├── __init__.py
    └── mapping_packs/
        └── __init__.py
```

`plugin.py` 必须保持薄、幂等、无模型 I/O，并只委托
`integrations.vllm_omni.bootstrap.register()`。console script 直接指向
`comfy_omni.cli.main:main`，不依赖 `cli/__init__.py` 的隐式 re-export。未来新增宿主放到新的
`integrations/<runtime>/`，不得在 `conversion` 中加入宿主判断。

### 4.2 模块迁移映射

| 当前模块 | 目标归属 | 说明 |
|---|---|---|
| `inspection.py` | `conversion/inspection/` + `artifacts/safetensors.py` + `domain/tensors.py` | I/O、用例编排和纯 descriptor/判定必须拆开 |
| `fsops.py`、`provenance.py`、`qkv.py` | `artifacts/` 或 `domain/` | 文件/摘要进入 artifacts；纯布局算法进入 domain |
| `h3/contracts/registry.py`、`templates.py` | `contracts/` | 只保留纯模型、模板与 compile-time registry |
| `h3/contracts/snapshots.py`、`contract_auto/` | `artifacts/contract_store.py` + `conversion/contract_workflows/` + `application/contracts.py` | snapshot schema、文件 I/O、scan/draft/pin/activate 分开，消除现有 registry SCC |
| `oracle/`、`mapping_packs/`、`converter/` | `conversion/` | 通用离线转换主线 |
| 根 `registry.py`、`h3/profiles.py` | `conversion/profiles/` + `contracts/schemas.py` + runtime adapter contribution | converter registry、artifact schema 和宿主 arch contribution 按职责拆开 |
| `native_export.py`、`vae_export.py` | `conversion/exporters/` | 按 component/export route 拆分 |
| `package_assembler.py`、`converter/package_v6.py` | `conversion/packaging/` | schema、plan、writer、verifier、publication 分开 |
| `lora_hotswap/bake_*`、`comfy_oracle.py`、overlay/migration | `conversion/lora/` | 离线 LoRA 产品链 |
| `lora_hotswap/catalog.py`、stack/cache/loader | `runtime/loras/` | 请求期 LoRA 生命周期 |
| `tools/contracts.py`、CAS/catalog/runtime | `runtime/tools/` | 工具合同、目录、驻留和执行 |
| `runtime_hotel/` | `runtime/hotel/` | 保持独立 bounded context |
| `parity/`、`preflight/`、根 `check.py` | `validation/` | 支持状态、宿主前检和 A/B 证据不混入 converter/domain |
| `component_catalog/`、`h3/component_*` | `runtime/components/` | 请求 schema、catalog、协调和生命周期 |
| `h3/runtime_*`、`h3/dense_pipeline.py` | `integrations/vllm_omni/pipelines/` + `runtime/h3/` | 纯领域逻辑与宿主子类分开 |
| `h3/vram_budget.py`、各 component ledger | `runtime/budget/` | 纯预算估算与 loader/adapter 分离 |
| `acceleration.py` 及 profile 实现 | `runtime/acceleration/` + `integrations/vllm_omni/` | 策略/计划与宿主执行分开 |
| 三套 plugin/import hook | `integrations/vllm_omni/bootstrap.py` + `import_hook.py` | 全项目唯一注册协调器 |
| 各域 API 模块 | `api/routes/` + `application/` | 路由只做 wire/error 映射；用例编排不留在 FastAPI 模块 |

### 4.3 强制依赖方向

```text
CLI ───────────────┐
HTTP API ──────────┼──> application ──> conversion/runtime ──> domain
runtime adapter ───┘           │                 │                 │
                               └─────────────────┼──> contracts ───┤
                                                 └──> artifacts ───┴──> core
```

依赖规则：

1. `core` 只依赖标准库。
2. `domain` 只依赖 `core`，不得依赖路径、Torch、FastAPI 或宿主类型。
3. `contracts` 可以依赖 `core/domain`，不得读取文件系统、环境变量或导入 Torch/FastAPI/vLLM。
4. `artifacts` 可以依赖 `core/domain/contracts`，统一承担受控 I/O。
5. `conversion` 可以依赖 `core/domain/contracts/artifacts`，不得依赖 `runtime/api/cli/integrations`。
6. `runtime` 可以依赖 `core/domain/contracts/artifacts`，不得依赖 FastAPI、CLI 或 conversion。
7. `application` 编排 conversion/runtime 用例，不反向被领域层依赖，也不携带 FastAPI/argparse 类型。
8. `api` 和 `cli` 只能调用 application service，不直接实现文件发布、模型转换或 GPU 协调。
9. `integrations/vllm_omni` 是 `vllm`、`vllm_omni` import 的唯一所有者；确有必要的 host protocol
   类型隔离在 adapter 内。现存分散 import 必须随迁移逐个清零。
10. 迁移前，仓内模块禁止 import `h3_forge.public`、`h3_forge.lora_hotswap.public` 或
   `h3_forge.tools.public`；迁移后同样禁止通过 `comfy_omni.public` 进行内部反向依赖。
11. 任何低层模块不得通过延迟 import 反向调用高层模块来规避依赖检查。

依赖方向应通过仓库自有 AST 检查或 import-linter 类工具进入 CI，不再只依赖人工评审。

## 5. 单一插件注册模型

### 5.1 声明式 contribution

core、LoRA、tools、hotel 不再各自扮演插件。它们提供纯声明或惰性工厂，由唯一 bootstrap 组合：

```python
@dataclass(frozen=True)
class PluginContribution:
    profiles: tuple[ProfileSpec, ...] = ()
    runtime_models: tuple[RuntimeModelSpec, ...] = ()
    router_factories: tuple[LazyRef, ...] = ()

@dataclass(frozen=True)
class LazyRef:
    module: str
    attribute: str
```

各领域只回答“我贡献什么”，不负责“何时向宿主注册”。`LazyRef` 只保存 dotted reference；收集
contribution 时不得 import FastAPI、Torch、vLLM pipeline 或模型代码，只有对应 bootstrap 阶段确认处于
正确进程后才能解析引用。

### 5.2 唯一 bootstrap 流程

当前入口 `h3_forge.plugin:register` 迁移为目标入口 `comfy_omni.plugin:register`，目标根级
`plugin.py` 内部只委托
`integrations.vllm_omni.bootstrap.register()`：

1. 收集编译期 contributions；
2. 幂等注册 converter profiles；
3. 宿主 registry 可用时注册/覆盖字符串形式的 runtime model；
4. 仅 root/API process 安装唯一 `after_import` hook；
5. API server 模块初始化完成后，构建一次总 router 并挂载所有子路由；
6. 重入只重试尚未完成的阶段，不递归调用另一个 plugin。

建议以显式阶段状态替代三组布尔量：

```text
PROFILES_REGISTERED
RUNTIME_REGISTERED
IMPORT_HOOK_ARMED
API_MOUNTED
```

每个阶段必须幂等、可测试、可观察，并保持注册路径无模型文件 I/O。

bootstrap 还必须定义以下状态机规则：

- 单进程内用一把锁保护“检查—执行—提交状态”，并允许同线程安全重入或明确拒绝递归；
- contribution 先按稳定 key 排序并检查重复。两个模块声明同一 profile、runtime arch 或 route 时，若
  不是显式 `replaces=<owner>` 合同则启动失败，不采用“最后注册者获胜”；
- 宿主 arch override、`_apply_lora` patch 和 API mount 各是独立阶段，记录 `pending/applied/failed` 与
  最后错误；一次部分失败只能重试未完成阶段，不重复已经提交的注册；
- patch 安装只发生在已确认的 API-server 阶段，安装前校验宿主版本和目标 callable shape；worker
  进程不得解析 router factory 或安装 API patch；
- 并发、两次调用、首次宿主未就绪、route 冲突、patch shape drift 和“第三阶段失败后重试”都有测试。

### 5.3 命名迁移与兼容策略

- `pyproject.toml` 的 project name 改为 `comfy-omni`；
- 源码根包由 `src/h3_forge/` 改为 `src/comfy_omni/`；
- console script 改为 `comfy-omni = "comfy_omni.cli.main:main"`；
- `vllm_omni.general_plugins` target 改为 `comfy_omni.plugin:register`；entry-point key 是否保留
  `h3_forge` 或改为 `comfy_omni` 由 Phase 0 的宿主 selector 消费审计决定；
- 项目名、distribution、import package、CLI 和 plugin target 在一个候选中原子迁移，禁止出现
  README 已改名但 wheel/entry point 仍为旧名的中间发布物；
- 保留既有 runtime arch key、REST URL、环境变量和 artifact schema，除非另有专项契约变更；
- `h3_forge` 以及 `h3_forge.lora_hotswap.plugin`、`h3_forge.tools.plugin` 在迁移期只允许作为
  内部兼容 shim；
- 若项目尚未对外发布这些 Python 子模块，可在 `0.2.0` 直接移除 shim；
- 若已有外部消费者，shim 仅 re-export 并发出明确 deprecation warning，一个 minor 周期后删除；
- 不为从未公开的内部函数建立兼容层。

若公开发布前没有真实 `h3_forge` 外部消费者，优先直接改名，不为假想消费者增加双包安装、
namespace alias 或双 entry point。若已经存在消费者，必须先列出 import、CLI、entry point、环境变量、
HTTP 和 artifact 六类消费面，再决定最小兼容周期。

这个消费者清单和决策必须在 Phase 0 完成，不能等改名 PR 开始后再决定。至少采用以下矩阵：

| 消费面 | 当前值 | `0.2.0` 目标 | 默认兼容策略 |
|---|---|---|---|
| 项目标题 | `h3-forge` | `ComfyOmni` | 直接切换并保留迁移说明 |
| GitHub repository | 内部旧仓 | `comfy-omni` | 独立公开 Git 根，小写 kebab-case |
| distribution | `h3-forge` | `comfy-omni` | 新 distribution；发布前核验并保留 PyPI/GitHub 名称 |
| Python import | `h3_forge` | `comfy_omni` | 无真实消费者则直接切换；否则单独 shim PR |
| console script | `h3-forge` | `comfy-omni` | 默认直接切换；是否保留旧 alias 由消费清单决定 |
| entry-point group | `vllm_omni.general_plugins` | 不变 | 宿主合同，不改 group |
| entry-point key/target | `h3_forge` / `h3_forge.plugin:register` | key：Phase 0 决策；target：`comfy_omni.plugin:register` | `VLLM_PLUGINS=h3_forge` 等 selector 存在时优先保留 key；同一 wheel 只保留一个有效入口 |
| runtime arch key | `MiniMaxH3*` 等 | 不变 | 视为宿主 wire contract |
| HTTP URL | `/v1/h3-forge/*` 等 | `0.2.0` 不变 | 后续版本另立 URL 迁移 ADR |
| 环境变量 | `H3_FORGE_*` | `0.2.0` 不变 | 集中登记为 legacy-stable contract |
| JSON/artifact schema | `h3_forge.error/v1`、`h3-comfy-*` 等 | `0.2.0` 不变 | schema version 与品牌名分离，禁止无版本替换 |

当前 `install_serving_video_lora_bridge()` 会 monkeypatch 宿主 `_apply_lora`。Phase 0 必须把它加入
host integration patch registry，记录 owner、目标符号、宿主版本/shape guard、安装进程、幂等性、
失败行为和移除条件。长期目标优先改用正式宿主扩展接口；在移除前，它是受控兼容补丁而不是“零补丁”。

## 6. 公共 API 与合同边界

### 6.1 public facade

目标 `comfy_omni.public` facade 只导出经过承诺的 SDK 能力：

- 只读 inspection；
- 稳定合同模型；
- conversion application service；
- package verify；
- 必要的错误类型和结果类型。

不得从 public facade 导出：

- `_validate_*`、`_reject_*` 等私有实现；
- FastAPI router；
- vLLM pipeline 子类；
- 可变的全局 registry；
- 仅供 CLI 调度的 helper；
- 仅供测试 monkeypatch 的内部 seam。

### 6.2 provider 化解除反向依赖

package assembler 不应 import LoRA public 来识别产品变体。改为由 application layer 传入明确的
扩展计划或 provider：

```python
class PackageExtension(Protocol):
    def validate_inputs(self, context: AssemblyContext) -> ExtensionPlan: ...
    def contribute_manifest(self, plan: ExtensionPlan) -> Mapping[str, object]: ...
    def materialize(self, plan: ExtensionPlan, staging: Path) -> None: ...
```

assembler 只处理统一的 extension contract；LoRA overlay、curve cache 和未来扩展各自实现 provider。
这样可移除 `package_assembler <-> lora_hotswap.public` 循环，同时保留现有产品语义。

### 6.3 schema 与版本

- Python package 版本只有一个来源，运行时通过 `importlib.metadata.version("comfy-omni")` 读取；
- artifact schema 版本与 Python package 版本分离；
- schema 常量集中登记，但各领域保持自己的解析器和错误类型；
- schema 变更必须带正向、旧版兼容或明确拒绝测试；
- 对外 JSON 字段、错误 kind、环境变量和 URL 视为 public contract。

## 7. 共享基础能力收拢

以下能力当前存在多个实现，但不能直接机械替换：

- canonical JSON；
- SHA-256 和 descriptor-bound hash；
- symlink/reparse/junction 拒绝；
- 原子写、`O_EXCL`、fsync 和不可覆盖发布；
- strict JSON duplicate-key 拒绝；
- staging directory 创建和原子提交。

迁移方法：

1. 为每套现有实现补齐 characterization tests；
2. 明确其错误类型、Windows/POSIX 差异和 TOCTOU 语义；
3. 在 `artifacts/` 提供低层 typed primitive；
4. 每次只迁移一个消费者；
5. 在领域边界把低层错误转换回原有 public error；
6. 完成全部迁移后删除重复实现和临时 re-export。

收拢的目标是单一语义所有者，不是为了减少几行重复而弱化安全边界。

## 8. 超大模块拆分方案

拆分顺序以解除耦合和降低变更风险为准，不按行数机械排序。

### 8.1 CLI

当前 `cli.py` 同时拥有 parser、所有命令参数、外部合同激活、命令执行和错误渲染。

目标：

```text
cli/
├── main.py                 # 入口和统一退出码
├── parser.py               # 根 parser 与子命令挂载
├── output.py               # JSON/text 输出和错误 envelope
└── commands/
    ├── inspect.py
    ├── convert.py
    ├── contracts.py
    ├── exports.py
    ├── lora.py
    ├── packages.py
    ├── parity.py
    ├── preflight.py
    └── hotel.py
```

每个 command 模块只定义 `configure_parser()` 和 `run()`，业务逻辑进入 application service。

### 8.2 package 与 exporter

`package_assembler.py`、`converter/package_v6.py`、`native_export.py`、`vae_export.py` 按以下职责拆分：

```text
conversion/packaging/
├── manifest.py
├── plan.py
├── writer.py
├── verifier.py
└── publication.py

conversion/exporters/
├── common.py
├── native.py
├── dense_bf16.py
└── vae.py
```

计划构建必须是纯函数或只读操作；writer 只消费冻结计划；verifier 不复用 writer 的内部状态；
publication 是唯一允许提交最终目录的模块。

### 8.3 dense pipeline

`h3/dense_pipeline.py` 同时包含宿主 import、合同识别、QKV 布局、模型构造、权重加载、forward、
TE patch 和 pipeline wrapper。目标拆分：

```text
runtime/h3/hybrid8/
├── contracts.py
├── geometry.py
├── qkv.py
├── modules.py
├── loader.py
├── forward.py
└── text_encoder.py

integrations/vllm_omni/pipelines/
└── dense.py
```

纯合同、几何和 QKV 算法不 import vLLM；只有最终 pipeline adapter 继承宿主类。

### 8.4 component runtime 与 VRAM

```text
runtime/components/
├── specs.py
├── sources.py
├── collective.py
├── text_encoder.py
├── vae.py
├── lora.py
└── service.py

runtime/budget/
├── common.py
├── hybrid8.py
├── text_encoder.py
└── vae.py
```

预算模块只计算和解释，不执行模型加载；加载服务消费预算决定。collective 模块只负责投票、广播、
一致性和补偿协议，不解析 API 请求或环境变量。

### 8.5 tools runtime 与 LoRA oracle

`tools/runtime.py` 拆为 coordinator、loader、residency、activation、engine；
`lora_hotswap/comfy_oracle.py` 拆为 reference binding、candidate I/O、receipt schema、receipt validator、runner。

拆分前必须锁定现有异常文本、退出码、receipt schema 和故障清理行为。

## 9. 配置与错误模型

### 9.1 配置

不建立一个全局巨型 Settings。按 bounded context 建立不可变设置：

- `LoraSettings`；
- `ToolSettings`；
- `HotelSettings`；
- `ComponentSettings`；
- `VramSettings`；
- `ContractStoreSettings`。

规则：

- `os.environ`/`os.getenv` 只允许出现在设置构造器或明确的宿主 adapter；
- application/runtime 函数接收 settings 对象，不在执行中重复读取环境；
- 默认值、单位、范围、敏感性和作用进程写入 `docs/configuration.md`；
- 测试直接构造 settings，不依赖进程全局环境；
- 兼容环境变量名保持不变，内部字段可以规范化。

### 9.2 错误

建议三层：

1. `core` 基础错误：输入、合同、资源、完整性、状态冲突；
2. 领域错误：`ConversionError`、`LoraError`、`ToolError`、`ComponentError`、`HotelError`；
3. 边界映射：CLI 退出码和 HTTP error envelope。

领域服务不得抛 FastAPI `HTTPException`；API 层不得依赖异常字符串猜测状态码。公开错误必须包含稳定
`kind/code`，人类可读 message 可以改进但不能替代机器字段。

### 9.3 代码规范

- Ruff formatter 是唯一格式化结果，Ruff lint 负责 import 排序和已启用的 `E/F/I/UP/B` 规则；禁止
  在 PR 中混入与目标无关的全仓格式化。
- 新增 public API、application service、Protocol 和 dataclass 必须完整标注参数与返回类型；边界 JSON
  先解析为 typed model，禁止把无约束 `dict[str, Any]` 传播到 domain/runtime。
- domain value object、plan 和 receipt 优先使用 frozen dataclass/不可变容器；可变 registry 的 owner、
  生命周期和锁必须明确。
- 顶层模块写职责/允许依赖/禁止依赖 docstring；public API 写合同、异常和副作用，注释解释“为什么”，
  不复述代码。
- 除 CLI renderer 外，生产代码不使用 `print`；使用模块 logger，日志不得包含 token、完整敏感路径、
  checkpoint 内容或未脱敏请求。跨进程阶段日志必须带 operation/candidate 标识。
- 不在领域层捕获宽泛 `Exception`；边界必须捕获时要保留 cause、映射稳定错误并测试失败路径。禁止
  `except: pass`、静默 fallback 和未经合同允许的 best effort。
- 文件路径使用 `pathlib.Path` 和 artifacts 安全 primitive；最终输出只能由 publication owner 提交，
  任何 helper 不得绕过 staging、摘要、链接拒绝和不可覆盖语义。
- 测试命名描述可观察行为；bug fix 必须先有失败回归，涉及 wire/schema/publication 的测试同时覆盖
  正例、篡改/缺失/重复项和中断清理。

## 10. 测试与质量门

### 10.1 测试分层

```text
tests/
├── unit/                   # 无网络、无宿主、无 GPU
├── contract/               # schema、manifest、mapping pack、receipt
├── integration/            # CLI、API、plugin hoststub、跨模块流程
├── packaging/              # sdist/wheel 安装和资源验证
├── host/                   # 真实 vLLM-Omni 或冻结 host mirror
└── fixtures/
```

pytest markers：

- `slow`：CPU 慢测试；
- `gpu`：需要 CUDA；
- `host`：需要固定 vLLM-Omni；
- `integration`：跨进程或 HTTP；
- `linux`：依赖 POSIX publication 语义。

当前测试尚未登记这些 markers，不能先用 `-m "not gpu and not host"` 过滤并假定未标记测试都是安全的。
迁移顺序是：先让现有 `python -m pytest -q` 完整收集并通过，再为确有外部条件的测试补 marker 和 skip
reason，并用 `--strict-markers` 防止拼写错误。PR CPU lane 运行全部 unit/contract/packaging 以及无宿主
integration；GPU/host/linux lane 由受控环境运行并绑定候选 commit。

CI 安装与平台矩阵固定如下，避免“完整 CPU 测试”含义随机器变化：

| lane | 安装 | 平台/Python | 责任 |
|---|---|---|---|
| lint/build/base smoke | `.[dev]` + build 后的 base wheel | Ubuntu；最低/最高支持 Python | Ruff、build、无 extras import/CLI |
| CPU full | `.[dev,serve,audit,vae]` | Ubuntu Python 3.11/3.12 | 包含 CPU Torch 的全量非 host/GPU 测试 |
| filesystem | `.[dev,serve,audit,vae]` | Windows + Ubuntu Python 3.12 | link/reparse、原子发布、路径边界 |
| packaging extras | 分别安装 base、`[serve]`、`[audit]`、`[vae]` | Ubuntu Python 3.12 | §11.2 能力矩阵 smoke |
| pinned host | 固定 container/wheelhouse | 受控 Linux/GPU | `[runtime]`、真实 vLLM-Omni、TP/媒体/parity |

若 CPU full 的 Torch wheel 不支持某个 Python 版本，不缩短项目声明范围来掩盖问题；base lane 仍覆盖
该版本，Torch 能力矩阵单独记录受支持的 Python 交集。dev extra 必须包含测试/构建工具，但不假装
自动包含 Torch；CPU full 明确通过 `[audit,vae]` 安装 Torch。

### 10.2 PR 必过门

```bash
python -m ruff format --check src tests scripts deploy
python -m ruff check src tests scripts deploy
python -m pytest -q --strict-markers
python scripts/check_release.py
```

`check_release.py` 必须使用新建的空临时目录构建 sdist/wheel、执行 twine check、从 sdist 重建 wheel、
创建 clean venv 并运行安装态 smoke；它不得读取或删除仓库现有 `dist/`、`build/` 或历史 wheel，避免
`twine check dist/*` 把旧产物误计入当前候选。

类型检查采用渐进式 no-regression：Phase 0 记录现有诊断基线，新增 `core/domain/contracts/application`
模块必须 strict，通过后再逐目录收紧。具体选择 Pyright 或 mypy 必须写入 `pyproject.toml` 并固定在 dev
依赖中；未确定工具前，不把“类型检查”写成虚假的发布通过项。

Phase 1 在干净虚拟环境中安装 wheel，并验证当前身份：

```bash
h3-forge --help
h3-forge profiles --json
h3-forge contract list --json
python -c "import h3_forge; import h3_forge.plugin"
```

Phase 2 改名候选把同一组 smoke 原子切换为：

```bash
comfy-omni --help
comfy-omni profiles --json
comfy-omni contract list --json
python -c "import comfy_omni; import comfy_omni.plugin"
```

packaging smoke 必须断言以下资源存在且可读取：

- `mapping_packs/target_matrix.json`；
- 每个生产 mapping pack 的 `pack.json`；
- wheel `.dist-info/licenses/` 下的 LICENSE，以及需要随分发携带的 NOTICE/归属文件；
- console script 和唯一 `vllm_omni.general_plugins` entry point。

资源验证必须在非源码目录、无 editable install 的全新虚拟环境中执行。Phase 1 通过
`importlib.resources.files("h3_forge.mapping_packs")` 枚举，Phase 2 改为
`comfy_omni.mapping_packs`；只有后续资源目录迁移完成后才使用
`comfy_omni.resources.mapping_packs`。每一步都要验证所有生产 pack；只检查 zip 中存在模糊的
`LICENSE` 文件名或只 import 两个模块都不算通过。sdist 解包后还要重复构建 wheel，避免 sdist 和
wheel 内容不一致。

缺少 pytest、Ruff、build 或 twine 时，发布门必须失败，不再以 `NOT_CONFIGURED` 计为成功。

### 10.3 结构门

- 新生产 `.py` 文件不超过 600 行；
- 已超过 1,300 行的文件只减不增；
- 新增或修改函数不超过 80 行；
- 圈复杂度不超过 15；
- 新增循环依赖数为零，最终消除全部已知 SCC；
- 新增 direct `os.getenv`、跨层 import、内部 public-facade import 必须被 CI 拒绝；
- 结构门先采用 no-regression baseline，再随拆分逐步降低基线，避免为一次性清零制造巨型 PR。

## 11. 打包与依赖

### 11.1 package data

在 `pyproject.toml` 中显式声明 runtime resources，不依赖本地 editable install 或偶然的构建缓存。
Phase 1 仍使用当前包名和当前目录：

```toml
[tool.setuptools.package-data]
h3_forge = [
  "mapping_packs/*.json",
  "mapping_packs/*.md",
  "mapping_packs/*/*.json",
  "mapping_packs/*/*.md",
]
```

Phase 2 只做 namespace 改名时，先等价改成 `comfy_omni.mapping_packs`，不同时移动资源目录。后续将
mapping packs 移入目标 `resources/` 的 PR 再改为：

```toml
[tool.setuptools.package-data]
comfy_omni = [
  "resources/mapping_packs/*.json",
  "resources/mapping_packs/*.md",
  "resources/mapping_packs/*/*.json",
  "resources/mapping_packs/*/*.md",
]
```

代码使用 `importlib.resources` 访问随包资源；需要真实目录的调用在受控临时目录中 materialize，
不得假设 wheel 安装后资源一定对应普通源码路径。

### 11.2 依赖分组

建议按实际 import 边界整理：

- base：离线 inspection、contracts、mapping 和基础转换所需最小依赖；
- `serve`：FastAPI/Pydantic 及 API server 集成；
- `runtime`：Torch/vLLM-Omni 宿主能力，若不能由 PyPI 解析则文档化固定镜像/commit；
- `audit`：数值审计；
- `vae`：VAE 转换；
- `dev`：pytest、Ruff、build、twine、类型检查和结构检查；
- `all`：面向开发/验证镜像的聚合 extra。

单发行包不等于所有依赖都必须进入基础安装。离线 CLI、API server、GPU runtime 应通过真实 import
边界决定依赖归属。

在移动依赖前，必须生成“能力—extra—入口”矩阵并做 clean-install 测试：

| 安装形式 | 必须可用 | 缺少可选依赖时的行为 |
|---|---|---|
| `comfy-omni` | inspection、合同、基础 mapping/convert、`--help` | 不得因导入 FastAPI/Torch/vLLM 失败 |
| `comfy-omni[serve]` | HTTP schema/router 构建 | 明确提示缺 runtime adapter，不在 import 时崩溃 |
| `comfy-omni[runtime]` | vLLM-Omni plugin 注册与宿主 adapter | 宿主版本/符号不匹配时 fail closed 并给稳定错误 |
| `comfy-omni[audit]` | Torch 数值审计命令 | 仅调用命令时检查 Torch |
| `comfy-omni[vae]` | VAE 转换命令 | 仅调用命令时检查 Torch/相关 backend |

当前默认 plugin 回调最终会导入 FastAPI 路由，因此若从 base 移走 FastAPI/Pydantic，必须同时保证：

1. entry-point import 和 worker `register()` 不导入 FastAPI；
2. 只有 API-server 进程触发路由工厂时才检查 `[serve]`；
3. 基础安装被宿主发现但未安装 `[serve]` 时，行为符合明确的部署合同（可选择“跳过 API 并告警”或
   “宿主启动 fail closed”，不得得到偶然的 `ModuleNotFoundError`）；
4. 上述每种安装形式都有独立 wheel smoke。

### 11.3 版本与发布

- 先发布 `0.2.0a1`/`0.2.0b1`，明确 pre-alpha/beta 状态；
- sdist 和 wheel 必须来自同一干净 commit；
- 发布物不得来自仓库内保留的旧 `dist/`；
- release workflow 生成临时产物，验证后上传，不把构建产物提交回 Git；
- changelog 记录用户可见行为，不继续承载完整 QA 日志。

## 12. 开源仓库整理

### 12.1 公开根目录

公开仓使用独立 Git 根 `plugins/comfy-omni/`，GitHub slug 为 `comfy-omni`，项目标题为
`ComfyOmni`。旧 `h3-forge` 及其他 sibling 仓只作为审计后的迁移输入；外层 `plugins/`、worktree、
Codex 临时目录、本地参考克隆和证据目录均不属于开源项目。

当前 Git remote 指向 RFC1918 地址/本地 sibling 仓，且提交历史来自 subtree 吸收。开源时默认创建
新的 public mirror/export repo，在审计后的 commit 上演练 clone/build/test，再设置独立的公开 remote；
不得把现有私有 `origin` 原地改成 GitHub 后直接推送。是否保留完整历史由 history secret/license scan
决定；若需过滤，保留旧私有仓只读归档，并用映射表记录公开首 commit 与内部候选 commit，不对现有
私有 remote 做 history rewrite 或 force-push。

### 12.2 必备文件

```text
LICENSE
NOTICE                       # 按实际派生/捆绑内容决定并核验
THIRD_PARTY.md               # 来源、commit、license、reuse 方式
CONTRIBUTING.md
CODE_OF_CONDUCT.md
SECURITY.md
SUPPORT.md
CHANGELOG.md
README.md
.github/workflows/*.yml
.github/ISSUE_TEMPLATE/*
.github/pull_request_template.md
```

### 12.3 第三方来源与许可证

开源前逐项核对：

- 标记为 `verbatim`、`official`、`mechanically copied` 的代码来源；
- 原文件的 copyright、SPDX 和 NOTICE；
- Apache/MIT/BSD 派生代码所需归属；
- GPL 或无许可证仓库是否仅用于行为观察，是否有实现片段进入当前代码；
- 测试 fixture、截图、提示词、生成媒体和模型派生物的公开权利；
- 模型权重许可证与代码许可证必须分别说明。

`ref/sources.json` 可以继续记录研究来源，但 `THIRD_PARTY.md` 只记录真正进入源码、fixture 或发行物的
内容，避免把“参考过”与“分发了”混为一谈。

### 12.4 内部信息清理

公开树和完整 Git 历史扫描以下内容：

- RFC1918 IP、内部 SSH remote、用户名和绝对路径；
- 私有验证主机假设、容器名、端口和挂载路径；
- API token、私钥、密码、临时签名 URL；
- 本地模型名、未公开 artifact 标识和内部日志；
- Codex 会话链接、绝对代码链接和临时审查转储。

通用部署脚本保留到 `examples/deployment/`；只适用于内部验证主机的 runbook 移到私有运维仓。

### 12.5 文档分类

建议公开文档结构：

```text
docs/
├── getting-started.md
├── architecture.md
├── support-matrix.md
├── configuration.md
├── cli.md
├── api.md
├── development.md
├── security-model.md
├── adr/
└── evidence/               # 仅保留可复核、已脱敏、可公开的证据索引
```

处理原则：

- `issues/`、`prs/`：完成的决策提炼为 ADR，其余由公开 issue tracker 承担；
- `docs/research/`：保留有长期价值的设计依据，删除重复 QA 转储和本地绝对链接；
- `docs/absorbed/`：可压缩为一份迁移说明和来源 commit 表，历史细节由 Git 保留；
- `artifacts/`：小型、授权明确、能支撑公开测试的 fixture 可保留；媒体交付物移到 release asset；
- README：聚焦安装、五分钟示例、支持矩阵、状态和贡献入口；完整证明链进入专项文档。

## 13. 分阶段实施计划

每阶段使用独立 PR；每个 PR 在 issue 中列出改动文件、公开接口、复用点、拆分点和验收命令，遵守
`code-management.md` 的规划先行约束。

### Phase 0：公开冻结与基线

目标：确认公开边界，建立可重复基线。

工作：

- 确认公开内容只进入独立 `plugins/comfy-omni/` Git 根，公开仓库 slug 为 `comfy-omni`，旧
  `h3-forge` 只作为逐文件审计的迁移来源；
- 核验并预留 GitHub/PyPI 名称，核对 ComfyUI/相关项目的名称与商标使用要求，并在 README 明确项目
  归属；输出 import、CLI、entry point、环境变量、HTTP、artifact 六类消费面清单，签字决定是否
  需要任何 `h3_forge`/旧 CLI shim；
- 写入命名/范围 ADR，明确稳定文档从 H3 特化权威迁移为 ComfyOmni core + 首个 H3/vLLM adapter；
- 冻结一个 clean candidate commit；
- 生成绑定 commit 的 pytest 收集数、Ruff、wheel 内容、LOC/超长函数、模块 SCC 基线报告；
- 生成依赖 import census 与“能力—extra—入口”矩阵，确认 base/serve/runtime 的失败行为；
- 登记所有宿主 monkeypatch，尤其 `_apply_lora` bridge 的版本 guard 与移除条件；
- 对当前树和完整历史执行 secret/license/internal-path 扫描，并生成逐文件 origin/license/disposition
  清单；对带 `verbatim`、`mechanically copied`、`heritage` 标记的实现逐一给出处置，至少包括
  `tools/archs/latent_resizer_3d.py`、`h3/dense_pipeline.py` 的官方数学镜像和 contracts 模板来源；
- 明确未跟踪文件和本地构建产物的 disposition。

出口条件：有可复核的发布前基线、命名/兼容决策、依赖能力矩阵、host patch registry、逐文件许可
处置和公开历史结论；任何来源/许可证未确认的文件都已从公开候选排除或被干净实现替换；没有直接
推送当前私有 remote/history 的未决项。

### Phase 1：发布门修复

目标：让“检查通过”真正代表测试和安装通过。

工作：

- `scripts/check.py` 委托 pytest 而不是 unittest discovery；
- 新增 `scripts/check_release.py`，所有构建与 clean-install 验证只使用临时目录；
- 工具缺失 fail closed；
- 显式 package data；
- wheel/sdist clean-install smoke；
- GitHub Actions CPU CI；
- 开源基础文件和许可清单骨架。

此阶段保持 `h3-forge`/`h3_forge` 名称不动，只修复“测试与发布门不可信”的问题，确保下一阶段的
机械改名有可靠保护。

出口条件：干净环境构建、安装、资源发现、CLI、plugin import、完整 CPU 测试全部通过；sdist 可重建
出同内容 wheel，缺少发布工具会失败。

### Phase 2：ComfyOmni 原子改名

目标：只改变公开身份和内部 import namespace，不改变 wire、artifact 或运行时行为。

工作：

- 将 distribution、源码包、CLI、entry-point target 原子迁移为
  `comfy-omni`、`comfy_omni`、`comfy-omni`、`comfy_omni.plugin:register`；entry-point key 严格执行
  Phase 0 决策，不因品牌改名自动改变；
- 全量修改源码、测试、脚本、deploy、文档、字符串式 plugin/runtime module path 和类型路径；
- 删除硬编码 `__version__`，统一通过 `importlib.metadata.version("comfy-omni")` 读取；
- 按 Phase 0 决策增加或不增加独立兼容 shim；shim 不与主改名逻辑混写；
- 保持 URL、`H3_FORGE_*`、runtime arch key、错误/artifact schema 不变，并加回归测试；
- 同一候选更新稳定权威文档、README、CHANGELOG 和发布元数据。

出口条件：源码树只有一个业务实现 namespace；wheel 只注册一个有效 vLLM-Omni entry point；旧标识
只出现在兼容矩阵允许的位置；新旧行为差异仅为 Phase 0 批准的身份变化。

### Phase 3：单一 bootstrap

目标：完成逻辑层面的单插件收敛。

工作：

- 引入 contribution；
- 建立唯一 bootstrap 和 import hook；
- 合并 API router 挂载；
- 删除 plugin 间递归；
- 原子迁移 profile/runtime/API 注册测试；
- 保留现有 entry point、arch key、路由和幂等/重试行为。

出口条件：三个旧注册状态机收敛为一个阶段状态机；worker import 仍无 FastAPI、模型 I/O 和目录扫描。

### Phase 4：依赖方向与共享基础能力

目标：先解除循环，再开始大规模拆分。

工作：

- 禁止内部 import public facade；
- assembler extension provider 化；
- 收拢 canonical JSON/hash/path/publication primitives；
- 建立依赖方向 CI；
- 将环境读取迁入领域 settings。

出口条件：已知 import SCC 清零；共享 I/O 消费者逐一通过 characterization tests。

### Phase 5：CLI、package、exporter 拆分

目标：缩小离线主路径的模块和函数职责。

工作：

- CLI command 模块化；
- package schema/plan/write/verify/publication 分离；
- native/dense/vae exporter 分离；
- public facade 缩面；
- wheel 安装态执行全部离线命令 smoke。

出口条件：相关超标文件显著下降；外部 CLI 参数、退出码和 artifact schema 不变。

### Phase 6：runtime 拆分

目标：隔离纯运行时逻辑与 vLLM 宿主适配。

工作：

- dense pipeline 拆分；
- component runtime/collective/source/budget 拆分；
- tools coordinator/residency/loader/engine 拆分；
- LoRA runtime 与离线 LoRA 分离；
- hoststub、TP、补偿和 fail-closed 测试按新边界迁移。

出口条件：只有 integrations 层继承/导入宿主实现；真实 host 候选保持原加载和请求行为。

### Phase 7：公开候选

目标：形成可发布的 `0.2.0` 预发布版本。

工作：

- README、support matrix、configuration、CLI/API 文档定稿；
- 清理或迁出内部 runbook、研究转储和未授权 artifact；
- 许可与 NOTICE/THIRD_PARTY 复核；
- clean history 或过滤后的公开镜像演练；
- sdist/wheel、容器、真实 host 候选绑定到同一 commit；
- 以 `ComfyOmni` 名义发布 `comfy-omni` 的 `0.2.0a1` 或 `0.2.0b1`。

出口条件：公开 clone 可按文档完成安装、CPU 检查和最小命令；发布物内容与许可清单一致。

## 14. 推荐 PR 切片

建议按以下顺序创建有界 PR：

1. `oss-baseline-and-contract-inventory`：基线报告、消费面、依赖能力矩阵、host patch registry、ADR；
2. `oss-release-gate`：保持旧名，修复 pytest、CI、package data、sdist/wheel clean-install smoke；
3. `comfy-omni-rename`：纯目录/引用/元数据/权威文档改名，不夹带行为重构；
4. `legacy-name-shim`：仅当 Phase 0 证明有消费者时创建，否则明确跳过；
5. `single-bootstrap`：唯一 contribution/bootstrap/import hook；
6. `dependency-direction`：内部 public import 禁止、application use cases、provider、模块图门；
7. `shared-artifact-primitives`：逐消费者迁移 fs/hash/canonical-json/publication/contract-store；
8. `cli-command-split`；
9. `package-export-split`；
10. `component-runtime-split`；
11. `dense-pipeline-split`；
12. `tools-and-lora-runtime-split`；
13. `oss-docs-and-release-candidate`。

每个 PR 只承担一个主要结构目标。若一个 PR 同时修改 public contract、artifact schema 和 runtime
行为，应继续拆分，除非三者不可分且 issue 中给出完整消费者与迁移证明。

## 15. 迁移纪律

1. 先测试和 characterization，后移动实现。
2. 纯移动、依赖反转、行为修改分成不同提交。
3. 每次迁移一个消费者，不做全仓搜索替换式的安全 helper 合并。
4. public wire、artifact schema、错误 kind、环境变量和 entry point 保持回归测试。
5. 旧模块 shim 不能包含业务逻辑，只能 re-export 或委托新实现。
6. 新旧路径并存期间，唯一实现必须在新模块；不得双写修复。
7. 每个阶段结束后更新 `architecture.md` 和 `CHANGELOG.md`，不把动态任务状态写入稳定架构文档。
8. 真实受控验证主机的运行结果属于候选证据，不直接写成通用默认配置。
9. 未通过结构门的重构不得以“测试全绿”为由合并；未通过测试门也不得以“只是移动”为由合并。

## 16. 完成定义

本轮“合并后重构与开源整理”完成需要同时满足：

- 单仓、单发行包、单 entry point 保持不变；
- 项目名、PyPI distribution、Python package、CLI 和标语分别为 `ComfyOmni`、`comfy-omni`、
  `comfy_omni`、`comfy-omni` 和 `Bring Comfy checkpoints to native Omni runtimes.`；
- Phase 0 的消费面兼容矩阵已有结论，旧 import/CLI shim 要么按证明实现，要么明确不存在；URL、环境变量、
  runtime arch key 和 schema 的 legacy-stable 标识都有自动化回归测试；
- core/LoRA/tools 不再拥有独立 plugin/import hook；
- 模块依赖图无循环；
- 仓内没有 public-facade 反向依赖；
- CLI/API/integration 通过明确的 application use case 编排；离线 conversion、runtime 与
  vLLM-Omni integration 分层明确；
- `0.2.0` 的支持声明明确限定为已固定并验收的 vLLM-Omni adapter，不因标语宣称未验收宿主；
- 所有宿主 monkeypatch 已移除或位于受测试的 patch registry，具有版本 guard、幂等性和退出计划；
- 所有新模块满足行数和复杂度门，历史超标文件完成计划内拆分；
- `python -m pytest -q`、Ruff、sdist/wheel、clean install smoke 进入 CI 且 fail closed；
- mapping packs 和 runtime resources 在 wheel 中可读取；
- README 的能力声明与 support matrix 一致；
- 公开树和公开历史没有未处置的秘密、内部基础设施信息或授权不明内容；
- LICENSE、NOTICE/归属、THIRD_PARTY 与实际发布物内容一致；
- 一个公开候选 commit 完成 CPU 发布门和适用的真实 host 验收。

达到这些条件后，当前 `h3-forge` 才完成向 `ComfyOmni` 的公开身份迁移，并从“完成插件物理归并”
进入“完成单产品架构收敛”的状态。
