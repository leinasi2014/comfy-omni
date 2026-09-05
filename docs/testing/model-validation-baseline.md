# ComfyOmni 固定模型验证基线

- 状态：固定既有资产身份；复用同一只读环境验收代码
- 修订日期：2026-09-05
- 机器可读资产记录：[model-baseline.v1.json](model-baseline.v1.json)

## 1. 用途

本基线固定一组既有 ComfyUI/H3 模型、LoRA 和配置，供代码修改前后反复验证。目标是直接复用这套
只读资产，热加载和卸载管理 RAM/VRAM 驻留，不以新建磁盘转换副本或完整模型包为启动前置。
已有可用的 legacy 包可以原址复用；原始组件的直接加载能力仍须由具体代码和测试证明。
首期验收关注：

1. 不加载 payload 即可检查 safetensors 结构与来源；
2. 按明确的组件路径和表示合同加载既有权重，必要的适配在有界内存中完成；
3. 在运行时修改之前判定模型、组件和量化表示是否兼容；
4. 在指定 GPU 验证主机加载完整 H3 组件并生成视频与音频；
5. 在同一控制实例及已有 worker 内完成完整 DiT 的 `A → B → A` 热切换，复用未变化的组件，
   不复用错误缓存并回收资源。

首期顺序为 H1 既有来源直接加载、H2 驻留与正常 worker 内切换、H3 真实宿主验证及交付。
完整 LoRA 生命周期、工具及节点工作流放到后续，不阻塞首期。下面的 LoRA 资产与场景保留为后续
测试数据，不要求为了首期验收读取它们或运行 LoRA 场景。

当前 [H3 来源路由](../../src/comfy_omni/integrations/vllm_omni/pipelines/runtime_pipeline.py) 已通过
`additional_config.comfy_omni_h3` 的 `active` 和 `sources` 选择固定 A、B 原始 ConvRot DiT，绕过
导出包验证入口；共享组件仍引用已有的宿主可读目录。[驻留协调器](../../src/comfy_omni/integrations/vllm_omni/residency_control.py)
和 [运行时 API](../../src/comfy_omni/api/routes/h3_runtime.py) 接通已有 worker 内的 DiT 装卸与切换。
接口及精度范围见 [H3 原文件运行说明](../guides/h3-original-files.md)。小型样本宿主回归证明这些
代码入口，不替代固定大模型完整生成、A→B→A 和资源回收的验收，也不证明表中所有原始 TE/VAE
格式均已支持直载。

这是一份验收合同，不是支持声明。模型存在或摘要正确，只能证明资产身份；只有对应行为的代码、
自动化检查和真实宿主证据同时通过，才能提升能力状态。

## 2. 固定资产

| ID | 角色 | 固定文件 | 字节数 | SHA256 |
|---|---|---|---:|---|
| `primary-dit` | 主 DiT | `10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors` | 20,967,637,320 | `54d56b15c65923b54c9ca16b494dae641bfe9455cfcb1c19c49b1008e270bbc1` |
| `text-encoder` | 文本编码器 | `qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors` | 15,683,129,659 | `47babbb3e4b7e43c097351ca39cfb7f326d014ae53a584f8559dc8121abca94c` |
| `audio-vae` | 音频 VAE | `model.safetensors` | 605,429,308 | `37dddc2f3e6d5d5139d823d5ea283bbf304dadcb885b1ccda818aa13dade5ea2` |
| `video-vae` | 视频 VAE 原始来源 | `minimax_h3_video_vae_fp16.safetensors` | 5,207,808,496 | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| `spatial-physics-lora` | LoRA 候选 | `wushu_spatial_physics_clean_3000_pruned.safetensors` | 155,109,672 | `7d14f3701560068e7004159c8b2a7278bd2dbfc9e5e3b60d0bc9aef6c049919d` |
| `realism-people-lora` | LoRA 候选 | `h3-realism-people-t2v-i2v-r2v.safetensors` | 131,229,656 | `acc529601d2da117fb81179e76c56e488a3beab1171659d305f04fa3655b787e` |
| `hot-swap-dit` | 完整 DiT 热切换候选 | `minimax_h3_ref2va_pruned_zs05_int8_convrot.safetensors` | 20,970,379,680 | `71b8085ac4221ee036708c230a007d617dccca1b0028b95bb4ee106cb2a385c5` |
| `tokenizer-config` | tokenizer 配置目录 | `Ref2VA/tokenizer/`（4 文件） | 11,492,078 | 逐文件以 JSON 为准 |
| `processor-config` | processor 配置目录 | `Ref2VA/processor/`（7 文件） | 11,498,352 | 逐文件以 JSON 为准 |

仓库、revision、下载页面、许可证状态和测试参数以 JSON 为准。固定文件名中的 `int8_convrot` 只是
来源声明，不替代 ComfyOmni 对 safetensors header、tensor schema 和量化 metadata 的独立检查。

### 2.1 文本编码器源文件的格式例外

指定验证主机对 `text-encoder` 的完整字节数和 SHA256 校验后发现：源文件的 safetensors
header 索引覆盖到数据区末尾之前 72 字节，末尾存在一个来源侧附加标记。该源文件因此不满足
[官方 safetensors 格式规范](https://github.com/huggingface/safetensors#format)中“整个数据区必须被索引
且不能有空洞”的要求。ComfyOmni 的 strict reader 必须继续以
`safetensors-unindexed-trailing-bytes` 拒绝它，不能用文件名、来源可信度或通用“忽略尾部”开关绕过。

JSON 中的 `staging_policy` 仅保留现有合同测试使用的派生身份数据：`15,683,129,587` 字节，SHA256
为 `a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f`，以及对应的 header、索引和
尾部摘要。这些数据不安排模型处理任务，也不要求生成副本。直接读取原文件的适配器仍需证明
这 72 字节的处理边界，不能放宽通用 strict reader 或据此宣称已经支持直载。

本发现绑定 ComfyOmni commit `71488775c17888ba81210b2cf1ba5bc4e52eb52d`、表中源文件字节数与
SHA256，以及 JSON 中的 header/索引/尾部摘要。strict reader 在结构检查阶段即拒绝，因此这次运行
没有产生 text-encoder 组件识别、NVFP4 识别或运行时可加载的结论。

### 2.2 视频 VAE 来源与配置身份

表中视频 VAE 摘要始终指向原始下载文件。其 562 个张量包含 560 个模型参数和
`latents_mean`、`latents_std` 两个统计张量；这是宿主参数布局差异，不是 safetensors 格式损坏。
既有测试记录中的 560 参数派生物为 5,207,806,104 字节，SHA256 为
`5a624684fad53d4acd0762aa7b07de4204de0bbb90f92c479605e326ccceb148`。
该配对保留上游包装配置中的 24 维高精度统计值，不把原始权重内的 FP16 统计快照当作它的精确来源。
派生配置为 2,906 字节，SHA256 为
`5d1163e8fb4030f3c927714611335840a6e500071cdf5d75ea9c13fccf9f5abc`。
`e4-component-configs.v1.json` 的 `files` 保留上游文件身份，`runtime_derivations`
单独绑定这份派生配置及其两个原始来源：1,807 字节的上游包装配置，以及 1,164 字节、SHA256 为
`66c68f541e6578ce613ce7a0fc985eb59097038829e49f7535e6d08e6d95ab12` 的 `source/config.json`。
该配置的字节来源是架构与包装配置的合并、清单中的三个字段覆盖，以及 `indent=2` 加末尾换行的
序列化。保留这一关系是为了准确归属参数和统计，不能把高精度统计误称为源 FP16 值的提升。
`staging_policy` 和 `runtime_conformance` 仅为现有合同测试保留原始与派生身份数据，不驱动运行
流程。直接读取原始 VAE 时须证明参数和统计的正确对应，不能把派生摘要写回源文件身份，
也不能忽略额外参数后声称加载成功。

### 2.3 官方 Ref2VA tokenizer/processor 组件目录

`tokenizer-config` 与 `processor-config` 来自官方 `MiniMaxAI/MiniMax-H3` 仓库 revision
`42ed227ee7df40d41602854ae760620d6eb651fe`（该 revision 与旧 `h3-forge` 对官方模板代码
hash-lock 的 revision 一致）下的 `Ref2VA/tokenizer/`（4 文件）与 `Ref2VA/processor/`（7 文件）。
两目录为多文件资产：字节数、SHA256 与 git blob SHA-1 逐文件固定在 JSON 中。

| 组件 | 文件 | 字节数 | SHA256 |
|---|---|---:|---|
| `tokenizer-config` | `merges.txt` | 1,671,839 | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `tokenizer-config` | `tokenizer.json` | 7,032,403 | `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7` |
| `tokenizer-config` | `tokenizer_config.json` | 11,003 | `a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68` |
| `tokenizer-config` | `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `processor-config` | `chat_template.json` | 5,499 | `5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce` |
| `processor-config` | `merges.txt` | 1,671,839 | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `processor-config` | `preprocessor_config.json` | 390 | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `processor-config` | `tokenizer.json` | 7,032,403 | `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7` |
| `processor-config` | `tokenizer_config.json` | 11,003 | `a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68` |
| `processor-config` | `video_preprocessor_config.json` | 385 | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |
| `processor-config` | `vocab.json` | 2,776,833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

历史获取在指定验证主机的 Docker 容器内完成；每个文件通过字节数与 git blob SHA-1 双重校验
（对 pinned revision 字节精确）后记录 SHA256。后续验证直接引用这些既有目录，无需重新下载、
复制或组包。两个目录在 `tokenizer.json`、`tokenizer_config.json`、`merges.txt`、`vocab.json` 上
内容一致，属官方布局事实，不是下载缺陷。

### 2.4 组件运行时配置与代码

固定宿主的该加载路径需要非权重文件：`text_encoder/config.json`（Qwen3VLConfig）、两个 VAE
的 `config.json` 与 `auto_map` 引用的 Python 代码文件。这些文件取自魔塔社区官方镜像
`MiniMax/MiniMax-H3@master` 的 `Ref2VA/` 分区（与 Hugging Face `MiniMaxAI/MiniMax-H3` 同源），
逐文件 SHA256 记录于 [e4-component-configs.v1.json](e4-component-configs.v1.json)（29 文件、
178,766 字节：text_encoder 1、video_vae 16、audio_vae 12）。VAE `config.json` 的
潜变量通道和统计由所选配置来源及精确派生记录绑定，不能在加载时猜测或替换；视频 VAE 的
架构合并、路径调整和统计来源见第 2.2 节。这些配置及代码可由来源路径引用；记录其身份不要求
每次测试重新下载或创建包含全部组件的新包。

## 3. 验收场景

### 3.1 资产身份

首次建立资产的字节身份时，对既有文件验证逻辑大小与完整 SHA256，并记录来源、文件身份及配置。
结构检查可以只读取 header；它本身不声称证明完整 payload 摘要。已由可信记录绑定且保持只读的
同一套资产用于后续代码验收，不要求每次检查、加载或切换都重新读取全部模型计算 SHA256。

后续加载执行轻量身份检测：核对路径所指文件、大小、文件标识和变更时间，以及有界 header、配置
与原记录的对应关系；加载过程中检查实际消费的 tensor 名称、shape 和 dtype。轻量检测是对既有
验证记录的复用，不是新的完整内容校验。文件缺失、替换或变化时停止受影响的加载，查明原因并
按当前任务重新确认相关资产；不得静默接受变化，也不默认重扫所有未受影响模型。

运行证据记录 ComfyOmni commit、宿主 adapter、容器或 wheel 身份，并引用来源 SHA256 的原始
验证记录及本次轻量检测结果。没有已验证适配路径的格式差异应明确拒绝，不能通过盲目复制或
转换来绕过。资产缺失时将依赖它的实际验收标为不可执行，不默认下载或生成替代物。

### 3.2 主运行时冒烟

`primary-dit + text-encoder + audio-vae + video-vae` 是首个 H3/vLLM-Omni 正向组件集。验收至少包括：

- 预检与预算计算不修改运行时；
- 组件加载、一次安全的确定性提示词生成、视频解码和音频解码；
- 输出媒体、请求参数和日志均经过脱敏并以摘要绑定；
- 失败不得静默回退到其他 checkpoint、精度或组件。

反复验收使用相同只读来源路径和明确的宿主环境，以代码候选为受测变量。不要求先产生新的 dense
权重、规范化副本或完整包；结果还应说明是否新增了模型 payload，默认不得新增。已有旧版可用路径
继续作为独立基线，不因新入口尚未完成而被删除或改写。

`primary-dit` 的上游仓库标记为敏感内容。公开测试只允许安全、中性的提示词；权重、未审核输出和
原始请求不得进入 Git 或公开证据目录。

### 3.3 LoRA 兼容性（后续范围）

此场景不是首期门槛。既有场景 ID 和资产身份保留；只有领取后续 LoRA 任务时才执行相关验收。

两个 LoRA 都先经过只读检查和兼容性 oracle。当前主 DiT 为 INT8 ConvRot，普通 H3 LoRA 是否可以
直接激活不能靠文件名或上游示例推断。验收规则是：

1. oracle 输出绑定底模与 LoRA 摘要、tensor/key 布局、任务族、量化表示和稳定 reason code；
2. 只有 `SUPPORTED` 且证明当前加载表示下的映射和数值路径时，才允许修改运行时；
3. 不兼容时在 mutation 前明确拒绝，这属于 fail-closed 通过，不得尝试“看起来能跑”的 fallback；
4. Spatial Physics 先测 `0.3`，再测 `0.5`；上游仍将其标记为实验阶段；
5. Realism People 使用触发词 `r34l1sm`，基准 scale 为 `1.0`；`0.6`、`0.8` 仅作为较弱效果观察点；
6. 若进入激活测试，必须完成 off → on → off，并验证缓存、模型摘要与输出 receipt 没有串扰。

### 3.4 完整 DiT 热切换

`primary-dit` 为 A，`hot-swap-dit` 为 B。B 是已经把 Z-Image 空间细节 graft 烘焙进权重的完整 DiT，
不是 LoRA。正常热切换必须在同一控制服务实例及已有 worker 中完成，并记录控制服务和各 worker
的身份。工作进程重建只算明确报告的恢复或降级，不能作为正常热加载通过：

1. 加载 A，报告 A 的摘要身份并生成；
2. 预检 B，通过后切换到 B，报告 B 的摘要身份并生成；
3. 清理 B，切回 A，再次报告 A 的摘要身份并生成；
4. 每一步验证 tensor schema、任务模式、量化 metadata、设备驻留、缓存 generation 和资源预算，
   并复用未变化的 TE、VAE 和 tokenizer 等组件；
5. 切换失败时保持或恢复最后一个完整状态，不留下半发布 registry、悬挂 cache 或错误 active identity；
6. 稳定窗口后显存与主存回到已声明预算，媒体与日志 receipt 能证明每次生成使用了正确模型。

这里的加载、清理和回切指 RAM/VRAM 的驻留与状态管理。A、B 和共享组件继续引用同一套既有
磁盘资产，不通过每次转换、创建新模型包或删除原模型完成热切换。

若 A/B 预检发现任务族或 schema 不兼容，必须在加载前拒绝并保留证据；这不等于热切换能力通过。
基线只能通过独立 PR 更改，不能在测试脚本中偷偷换用另一个权重。

## 4. 证据范围

证据按实际观察的行为归属，不把资产准备过程排成运行能力的晋级路线。

| 范围 | 证明内容 | 不证明的内容 |
|---|---|---|
| 资产身份 | 首次字节数与 SHA256 匹配；后续记录明确的轻量检测 | ComfyOmni 能正确加载或生成 |
| 结构与映射 | header、tensor 与兼容性检查通过 | GPU 能加载或生成 |
| 加载与生成 | 指定 GPU 主机复用既有资产完成加载与生成 | 所有运行时都受支持 |
| 首期生命周期 | 已有 worker 内 DiT `A → B → A`、组件复用及资源回收通过 | LoRA、工具或节点已交付 |
| 后续 LoRA | 被明确选择的 LoRA 兼容性或激活场景通过 | 首期必须包含 LoRA，或未测试组合受支持 |
| 软件交付 | 同一候选通过受影响的 CPU、软件发行包、结构、宿主和许可门并合入 `main` | 未来版本自动继承结论 |

源码、运行时、配置或资产变化时，重新验证实际受影响的行为。未改变的来源身份和独立数值证据
可复用；重测代码不意味着重新下载、全量哈希或重建模型。保留失败和成功的原始记录，准确说明
它们对应的候选与路径；文档更新不能替代当前直接加载和生成的验收。

## 5. 资产与许可证边界

- 权重由维护者在受控环境外部获取，普通 CI 不联网下载模型。
- ComfyOmni 仓库、sdist、wheel、源码归档和测试 fixture 都不得携带这些权重。
- 每项资产遵守其来源仓库及底模许可；`text-encoder` 在公开证据前仍需完成许可证复核。
- 生成媒体只有在提示词、输入、人物/商标、模型条款和公开权利都审查完成后才能进入公开证据。
- 任何模型替换都必须同时更新 JSON、本文、合同测试和受影响的验收 issue。

## 6. 当前目标与测试数据

当前目标、实施顺序和验收决定位于 [项目目标](https://github.com/leinasi2014/comfy-omni/issues/4)
及其关联任务。本文不维护另一套进度路线。JSON 中的来源、原始/派生身份、`staging_policy`
和既有场景 ID 仅为来源追溯与现有合同测试保留数据兼容性，不安排模型处理任务，也不构成运行
前置。文档和代码尚未一致的部分应明确列为待完成工作，不能用文档更新宣称功能已经迁移完成。
