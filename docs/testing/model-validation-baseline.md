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
| `audio-vae` | 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| `video-vae` | 视频 VAE | `minimax_h3_video_vae_fp16.safetensors` | 5,207,808,496 | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| `spatial-physics-lora` | LoRA 候选 | `wushu_spatial_physics_clean_3000_pruned.safetensors` | 155,109,672 | `7d14f3701560068e7004159c8b2a7278bd2dbfc9e5e3b60d0bc9aef6c049919d` |
| `realism-people-lora` | LoRA 候选 | `h3-realism-people-t2v-i2v-r2v.safetensors` | 131,229,656 | `acc529601d2da117fb81179e76c56e488a3beab1171659d305f04fa3655b787e` |
| `hot-swap-dit` | 完整 DiT 热切换候选 | `minimax_h3_ref2va_pruned_zs05_int8_convrot.safetensors` | 20,970,379,680 | `71b8085ac4221ee036708c230a007d617dccca1b0028b95bb4ee106cb2a385c5` |

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
