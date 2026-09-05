# ComfyOmni 固定模型验证基线

- 状态：`v1` 已冻结，等待逐阶段实现与真实宿主验收
- 日期：2026-09-02
- 机器可读权威：[model-baseline.v1.json](model-baseline.v1.json)

## 1. 用途

本基线把重构目标绑定到一组确定的 Comfy 单文件组件。它用于验证 ComfyOmni 是否真正完成：

1. 不加载 payload 即可检查 safetensors 结构与来源；
2. 以明确 profile 离线转换、打包并验证不可变产物；
3. 在运行时修改之前判定底模、量化表示和 LoRA 是否兼容；
4. 在指定 GPU 验证主机加载完整 H3 组件并生成视频与音频；
5. 在同一宿主进程完成完整 DiT 的 `A → B → A` 热切换，不复用错误缓存并回收资源。

这是一份验收合同，不是支持声明。模型已下载或摘要正确，只能证明资产身份；只有对应阶段的代码、
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
| `tokenizer-config` | 包组件 tokenizer 配置目录 | `Ref2VA/tokenizer/`（4 文件） | 11,492,078 | 逐文件以 JSON 为准 |
| `processor-config` | 包组件 processor 配置目录 | `Ref2VA/processor/`（7 文件） | 11,498,352 | 逐文件以 JSON 为准 |

仓库、revision、下载页面、许可证状态和测试参数以 JSON 为准。固定文件名中的 `int8_convrot` 只是
来源声明，不替代 ComfyOmni 对 safetensors header、tensor schema 和量化 metadata 的独立检查。

### 2.1 文本编码器源文件的格式例外

指定验证主机对 `text-encoder` 的完整字节数和 SHA256 做 E1 校验后发现：源文件的 safetensors
header 索引覆盖到数据区末尾之前 72 字节，末尾存在一个来源侧附加标记。该源文件因此不满足
[官方 safetensors 格式规范](https://github.com/huggingface/safetensors#format)中“整个数据区必须被索引
且不能有空洞”的要求。ComfyOmni 的 strict reader 必须继续以
`safetensors-unindexed-trailing-bytes` 拒绝它，不能用文件名、来源可信度或通用“忽略尾部”开关绕过。

后续 staging 切片必须保留原文件和原 SHA256，只能通过 digest-pinned profile 原子生成新的规范化
副本。receipt 至少绑定原 SHA256、原字节数、被移除尾部的字节数与 SHA256、派生文件字节数、派生
SHA256、工具 commit 与 wheel SHA256；派生文件必须重新通过同一个 strict reader。指定验证主机已用
只读、流式摘要发现流程冻结派生文件身份：`15,683,129,587` 字节，SHA256 为
`a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f`。该发现仅冻结预期身份；只有
ComfyOmni 的正式命令从固定源文件生成独立副本、发布 receipt 并完成严格重读后，才构成 E3 验收。
此例外不是运行时支持声明，也不能扩展成对任意尾部数据的容忍。

本发现绑定 ComfyOmni commit `71488775c17888ba81210b2cf1ba5bc4e52eb52d`、表中源文件字节数与
SHA256，以及 JSON 中的 header/索引/尾部摘要。strict reader 在结构检查阶段即拒绝，因此这次运行
没有产生 text-encoder 组件识别、NVFP4 识别或运行时可加载的结论。

### 2.2 视频 VAE 来源与运行时派生物

表中视频 VAE 摘要始终指向原始下载文件。其 562 个张量包含 560 个模型参数和
`latents_mean`、`latents_std` 两个统计张量；这是宿主参数布局差异，不是 safetensors 格式损坏。
运行时选用单独的 560 参数派生物：5,207,806,104 字节，SHA256 为
`5a624684fad53d4acd0762aa7b07de4204de0bbb90f92c479605e326ccceb148`。
统计值由对应配置承载：2,906 字节，SHA256 为
`5d1163e8fb4030f3c927714611335840a6e500071cdf5d75ea9c13fccf9f5abc`。
`e4-component-configs.v1.json` 的 `files` 保留上游文件身份，`runtime_derivations`
单独绑定这份派生配置及其原始来源；组包使用派生配置与原有动态代码组成的精确名单。
`staging_policy` 记录原始与派生身份；组包验收只接受派生 payload 及其精确配套文件，
不能把派生摘要写回上游源文件身份，也不能忽略额外参数后声称加载成功。

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

下载只允许在指定验证主机的 Docker 容器内经维护者 loopback-only 隧道进行，不修改系统代理；
每个文件必须同时通过字节数与 git blob SHA-1 双重校验（对 pinned revision 字节精确）后才记录
SHA256。两个目录在 `tokenizer.json`、`tokenizer_config.json`、`merges.txt`、`vocab.json` 上
内容一致，属官方布局事实，不是下载缺陷。

E3 组件树中，`transformer` 使用本仓库自己的 Ref2VA 全量转换输出（绑定 `hot-swap-dit` 来源），
`text_encoder` 使用 digest-pinned 规范化 strict 副本。视频 VAE 使用上述 Comfy 源文件的派生
payload，音频 VAE 使用表中摘要固定的 ModelScope 官方 `Ref2VA/audio_vae/model.safetensors`。
不引入官方 transformer 或 text_encoder 权重。

### 2.4 组件运行时配置与代码（E4 包 v3）

真实宿主加载要求组件目录携带非权重文件：`text_encoder/config.json`（Qwen3VLConfig）、两个 VAE
的 `config.json` 与 `auto_map` 引用的 Python 代码文件。这些文件取自魔塔社区官方镜像
`MiniMax/MiniMax-H3@master` 的 `Ref2VA/` 分区（与 Hugging Face `MiniMaxAI/MiniMax-H3` 同源），
逐文件 SHA256 记录于 [e4-component-configs.v1.json](e4-component-configs.v1.json)（29 文件、
178,766 字节：text_encoder 1、video_vae 16、audio_vae 12）。VAE `config.json` 的
`latent_channels/latents_mean/latents_std` 后续按转换后负载实测派生修订（legacy 先例：从实际
payload 张量派生统计）。下载仅在指定验证主机的 Docker 容器内经魔塔国内直连执行。

## 3. 验收场景

### 3.1 资产身份

任何检查、转换或加载前都必须验证逻辑大小与 SHA256。下载使用 `.partial` 和原子改名；缺失、超长、
摘要不符或来源未绑定时立即失败。源文件通过 E1 后仍须独立通过格式检查；非规范源文件必须隔离，
不得直接加载。运行证据必须记录 ComfyOmni commit、宿主 adapter 版本、容器或 wheel 身份，以及
全部参与源资产与派生资产的 SHA256。

### 3.2 主运行时冒烟

`primary-dit + text-encoder + audio-vae + video-vae` 是首个 H3/vLLM-Omni 正向组件集。验收至少包括：

- 预检与预算计算不修改运行时；
- 组件加载、一次安全的确定性提示词生成、视频解码和音频解码；
- 输出媒体、请求参数和日志均经过脱敏并以摘要绑定；
- 失败不得静默回退到其他 checkpoint、精度或组件。

`primary-dit` 的上游仓库标记为敏感内容。公开测试只允许安全、中性的提示词；权重、未审核输出和
原始请求不得进入 Git 或公开证据目录。

### 3.3 LoRA 兼容性

两个 LoRA 都先经过只读检查和兼容性 oracle。当前主 DiT 为 INT8 ConvRot，普通 H3 LoRA 是否可以
直接激活不能靠文件名或上游示例推断。验收规则是：

1. oracle 输出绑定底模与 LoRA 摘要、tensor/key 布局、任务族、量化表示和稳定 reason code；
2. 只有 `SUPPORTED` 且证明映射或离线 bake 路径时，才允许修改运行时；
3. 不兼容时在 mutation 前明确拒绝，这属于 fail-closed 通过，不得尝试“看起来能跑”的 fallback；
4. Spatial Physics 先测 `0.3`，再测 `0.5`；上游仍将其标记为实验阶段；
5. Realism People 使用触发词 `r34l1sm`，基准 scale 为 `1.0`；`0.6`、`0.8` 仅作为较弱效果观察点；
6. 若进入激活测试，必须完成 off → on → off，并验证缓存、模型摘要与输出 receipt 没有串扰。

### 3.4 完整 DiT 热切换

`primary-dit` 为 A，`hot-swap-dit` 为 B。B 是已经把 Z-Image 空间细节 graft 烘焙进权重的完整 DiT，
不是 LoRA。一次可接受的热切换证据必须在同一宿主进程中完成：

1. 加载 A，报告 A 的摘要身份并生成；
2. 预检 B，通过后切换到 B，报告 B 的摘要身份并生成；
3. 清理 B，切回 A，再次报告 A 的摘要身份并生成；
4. 每一步验证 tensor schema、任务模式、量化 metadata、设备驻留、缓存 generation 和资源预算；
5. 切换失败时保持或恢复最后一个完整状态，不留下半发布 registry、悬挂 cache 或错误 active identity；
6. 稳定窗口后显存与主存回到已声明预算，媒体与日志 receipt 能证明每次生成使用了正确模型。

若 A/B 预检发现任务族或 schema 不兼容，必须在加载前拒绝并保留证据；这不等于热切换能力通过。
基线只能通过独立 PR 更改，不能在测试脚本中偷偷换用另一个权重。

### 3.5 固定六组件真实组包（E3）

`hot-swap-dit`（经 Ref2VA 全量转换输出）、`text-encoder`（规范化 strict 副本）、`audio-vae`、
`video-vae`、`tokenizer-config`、`processor-config` 构成首个固定组件集。可接受的组包证据必须在
指定验证主机的 Docker 内完成：

1. 六个组件目录 census 恰好等于冻结名单（无 sidecar、锁文件或多余条目），payload 以只读硬链接
   进入独立组件树，不修改既有 ComfyUI 模型树；
2. 安装 wheel 的真实工具身份驱动 `parse_component_receipt → plan_native_package →
   verify_package_sources → materialize_package → publish_package` 完整链；
3. 每一步的不可变结果（receipt 摘要、plan 自摘要、源验证摘要、staged census 摘要、manifest
   自摘要）都通过 pinned 期望值或自摘要重算复核；
4. 发布后的包由独立 verifier 容器从头重读：树 census、逐文件 pinned 哈希、manifest 自摘要与
   内容一致性，任何不一致即整体失败；
5. 证据记录 ComfyOmni commit、wheel SHA256、容器镜像 ID、全部组件来源摘要与耗时/峰值 RSS。

组包成功不构成运行时声明：native load、最小生成、LoRA 与热切换仍属 E4/E5。

## 4. 证据等级

| 等级 | 证明内容 | 不证明的内容 |
|---|---|---|
| E0 | URL/revision 已记录 | 文件已经下载 |
| E1 | 字节数与 SHA256 匹配 | ComfyOmni 能理解文件 |
| E2 | header、tensor 与兼容性检查通过 | GPU 能加载或生成 |
| E3 | 离线转换、打包、重读和摘要验证通过 | 真实宿主兼容 |
| E4 | 指定 GPU 主机完成加载与生成 | 所有运行时都受支持 |
| E5 | LoRA 生命周期或 DiT `A → B → A` 场景通过 | 未测试模型与并发模式受支持 |
| E6 | 同一候选通过 CPU、打包、结构、宿主和许可门并合入 `main` | 未来版本自动继承结论 |

任何后续源码、运行时、配置、模型摘要或宿主环境变化，都会使受影响等级的证据失效，必须重新运行。

## 5. 资产与许可证边界

- 权重由维护者在受控环境外部获取，普通 CI 不联网下载模型。
- ComfyOmni 仓库、sdist、wheel、源码归档和测试 fixture 都不得携带这些权重。
- 每项资产遵守其来源仓库及底模许可；`text-encoder` 在公开证据前仍需完成许可证复核。
- 生成媒体只有在提示词、输入、人物/商标、模型条款和公开权利都审查完成后才能进入公开证据。
- 任何模型替换都必须同时更新 JSON、本文、合同测试和受影响的验收 issue。

## 6. 与里程碑的关系

- M0 冻结 JSON、许可状态和宿主证据格式；
- M1 让 manifest 合同检查进入 CPU CI，但不下载模型；
- M4 用固定文件驱动 inspection、oracle、conversion、package 与 publication；
- M5/M6 用固定组件集驱动运行时、LoRA 和完整 DiT 热切换；
- M7 只接受绑定同一源码候选、同一 manifest 版本和完整证据链的预发布版本。
