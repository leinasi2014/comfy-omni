# ComfyOmni 开发交接

更新时间：2026-09-03（Asia/Shanghai）

## 1. 当前目标

把最新可运行的 `h3-forge` 能力按清晰依赖方向重构到独立开源仓库 ComfyOmni，并最终在
`srv-00` 的 Docker 环境完成固定模型的原生包加载、最小生成、LoRA 生命周期和完整 DiT
`A → B → A` 热切换验收。

项目固定身份：

- 项目名：`ComfyOmni`
- GitHub 仓库：`leinasi2014/comfy-omni`
- PyPI distribution：`comfy-omni`
- Python package：`comfy_omni`
- CLI：`comfy-omni`
- 标语：`Bring Comfy checkpoints to native Omni runtimes.`

## 2. 仓库与权威来源

- 本地新仓库：`D:\Source\vllm-omni\plugins\comfy-omni`
- GitHub：<https://github.com/leinasi2014/comfy-omni>
- 最新旧项目权威：`D:\Source\vllm-omni\plugins\h3-forge`
- 只允许参考旧项目提交：`e9cb011d00b028c149db3978de246c54f6e34acc`
- 旧项目许可证：Apache-2.0
- 旧项目有未跟踪文件 `docs/h3-api-reference.md`；不要修改、删除或提交它。
- 迁移代码进入 PR 前必须记录旧文件的精确 commit/blob、许可证、归属、保留行为、特征测试证据和
  distribution disposition。不要参考旧项目中比上述提交更新、未提交或其他分支的实现。

## 3. 不可违反的执行规则

- 项目 Python、pytest、Ruff、构建、打包、依赖安装、下载、模型解析、转换和推理全部在 Docker 中运行。
- 本机只允许编辑/读取文件、Git/GitHub、Docker 编排、SSH/SCP 和只读诊断；本机 Docker 当前不可用，
  因此普通门禁以 GitHub Actions 的 Docker job 为准（quality/documentation 仅在 PR 与 main push 上触发，
  分支单独 push 不触发——先建 draft PR 再等门禁）。
- `srv-00` 上同样 Docker-first；不得直接向宿主安装 Python 包、修改系统代理或污染系统环境。
- 模型源以只读 mount 进入容器；输出 staging-first、独立验证、manifest-last，禁止覆盖既有产物。
- 不得把模型、wheel、缓存、服务器证据、生成媒体、私有基础设施或密钥提交到 Git。
- 不得在文档中记录完整 VLESS URI、UUID 或其他凭据。本项目只记录“使用维护者提供的 VLESS 节点”。
- 依赖方向固定为：
  `core -> domain/contracts/artifacts -> conversion/runtime -> application -> CLI/API/integrations`。
- import `comfy_omni` 或插件入口不得加载 Torch、FastAPI、vLLM、模型或 checkpoint payload。
- Git 流程：单 WIP；冻结 READY；先提交 RED；观察 Docker 红灯；GREEN；REFACTOR；短分支；PR；
  全绿后 squash merge；最后必须等待 `main` push 自身的 Docker 回读。
- ruff format 会把能放进 120 列的推导式/调用合并成单行；多行括号导入与单行导入混排会被
  isort 要求特定顺序——写测试时预先合并，红灯前先看是否为格式噪声。
- 开发依据 `D:\skills\manage-agile-software-development\SKILL.md`；服务器操作依据适用的 SSH/srv skill。

## 4. 已完成并进入主线（本交接新增部分）

### 4.1 基础与转换（此前已合并）

仓库规范、门禁、README/AGENTS/CONTRIBUTING、Docker-first、quality 3.10/3.13、package 3.12、
documentation 合同、Ref2VA 完整转换（PR #25）、原生包规划（PR #26）、包源验证（PR #27）。

### 4.2 原生包私有暂存（PR #28，Issue #9 slice 3）

- 合并提交：`ad4770432a867c71e57ed3a8cac83e051040ef1b`
- `artifacts.fileops.copy_file_pinned_exclusive` + `conversion.packaging.materialization.materialize_package`。
- RED `1471ba5` / run `33679298001`；GREEN `dd369fe` / run `33680349335`（220 tests）。
- provenance：`docs/migration/native-package-materialization-e9cb011.md`。
- 主线回读：quality `33681696086`、documentation `33681696067`。

### 4.3 manifest-last 原子发布（PR #29，Issue #9 slice 4）

- 合并提交：`0d111a553f40a86981566119efe37df3671ce4ed`
- `conversion.packaging.publication.publish_package`：独立重读 staged 树、canonical manifest
  （`package_manifest_sha256` 自摘要，排除字段域）、manifest-last 独占写、同父目录原子 rename、
  `PackagePublication`/`PUBLISHED` 结果；失败保留 staging、绝不覆盖输出。
- RED `28eef7c` / run `33683554019`（`1 failed, 220 passed`，模块缺失；此前三次 push 为格式噪声）。
- GREEN `debdbfc` / run `33686167540`（227 tests）；documentation run `33686167394`。
- provenance：`docs/migration/native-package-publication-e9cb011.md`。
- 主线回读：quality `33686494301`、documentation `33686494167`。
- 已知刻意偏离：不再生成 legacy 每分区 `model_index.json`；routing 索引并入 manifest；
  同父目录原子 rename 为新增强化。

### 4.4 组件目录 receipt 解析（PR #30，Issue #9 slice 5）

- 分支：`codex/feat-package-receipts`；READY：
  <https://github.com/leinasi2014/comfy-omni/issues/9#issuecomment-5516855121>
- `conversion.packaging.receipts.parse_component_receipt`：确定性树 census、拒绝 link/special/
  空树、每文件 pinned 哈希、复核遍（同尺寸改写也拒绝）、`receipt_sha256` canonical 摘要、
  直接产出 `plan_native_package` 可用的 `ComponentReceipt`。
- RED `788306c` / run `33687433419`（`1 failed, 227 passed`；首推 `55ff5e5`/`33687245144` 为格式噪声）。
- GREEN `745a1da` / run `33688797338`（233 tests）；documentation run `33688797414`。
- provenance：`docs/migration/component-receipt-parsing-e9cb011.md`。

完整发布链已闭合：`parse_component_receipt -> plan_native_package -> verify_package_sources ->
materialize_package -> publish_package`。

### 4.5 E3 前置与固定六组件真实组包（PR #31 + 服务器执行，Issue #9）

- 决策冻结：<https://github.com/leinasi2014/comfy-omni/issues/9#issuecomment-5517264707>
- 基线扩展（PR #31，合并 `b47d084`）：官方 `MiniMaxAI/MiniMax-H3@42ed227e` 的
  `Ref2VA/tokenizer`（4 文件）与 `Ref2VA/processor`（7 文件）入库；srv-00 Docker 内经维护者
  loopback 隧道下载，每文件字节数 + git blob SHA-1 双验证。基线合同测试同步演进
  （RED `33690994503` → GREEN `33691438938`，233 tests）。main 回读 `33691593802`/`33691593808`。
- E3 真实组包已在 srv-00 完成并独立复核（2026-09-03）：候选 `b47d084`，wheel
  `578665973c…cde483`，镜像 `sha256:89cd460e…3d27`；六组件（transformer=Ref2VA 全量转换输出
  40,226,030,420B/14 文件、text_encoder=规范化 strict 副本、双 VAE、tokenizer/processor）receipt
  → plan → verify → materialize → publish 全链 `ASSEMBLED_PUBLISHED`，28 文件
  61,745,213,741B，manifest `e3577642…21bd`，plan `bf514bba…6d20`，1017.5s，峰值 RSS 52MB；
  独立 verifier 容器 `VERIFIED`（139.7s）。证据：
  `docs/evidence/native-package-assembly-b47d084.md` 与
  `/home/hyl/comfy-omni-e3/run-b47d084-attempt1`。
- 发布包位于 `srv-00:/data/models/comfy-omni/e3-output/native-package`（`Ref2VA/` +
  `h3-comfy-package.json`）。
- srv-00 操作要点：编排必须以 transient systemd 单元（`sudo systemd-run --uid=1000`）运行，
  普通 ssh 后台任务会被会话回收；`umask 077` 下建出的组件/输出目录需 `chmod a+rx` 供容器
  uid 65532 读取；跨 `/home` 与 `/data` 不能硬链接，组件目录按同盘硬链接 + 容器内 bind mount
  组装统一 `/components`。

Issue #9 保持打开，直到其验收条款全部满足并评审关闭。

## 5. 当前唯一 WIP：E3 证据文档 PR

- 分支：`codex/docs-e3-assembly-evidence`（PR #32）；`docs/evidence/native-package-assembly-b47d084.md` +
  HANDOFF 更新。

## 6. 下一位执行者的精确续接步骤

1. 等 E3 证据 PR 的 quality/documentation Docker checks；全绿后按模板转 ready、squash merge、
   等 main 回读，并把证据要点评论到 Issue #9（引用 4.5 的 digest）。
2. 下一切片（建议顺序）：
   - 单一、lazy、idempotent 的 vLLM-Omni plugin bootstrap（`vllm_omni.general_plugins` 入口，
     不得加载 Torch/vLLM/模型；多进程安全）；先在 Issue #9 或新 issue 冻结 READY。
   - 随后 native load 与最小安全提示词视频/音频生成 E4（使用已发布包 + pinned host
     `17285c2f`）；再 LoRA 只读 oracle 与 off/on/off；再 `A → B → A` E5。
3. 服务器 E4 注意：pinned vllm-omni host commit 与本仓库 bootstrap 的对接方式尚未冻结，
   进入前先在 issue 固化 host 侧加载路径与证据格式。

常用只读/编排命令：

```text
git status --short
git log --oneline --decorate -8
gh pr view 30 --json headRefOid,isDraft,mergeStateStatus,statusCheckRollup,url
gh pr checks 30 --watch
gh run list --branch codex/feat-package-receipts --limit 6
```

## 7. 固定测试资产与下载事实

机器可读权威是 `docs/testing/model-baseline.v1.json`，说明文档是
`docs/testing/model-validation-baseline.md`。7 项资产均已下载到 `srv-00` 的
`/data/models/comfy-omni/` 管理树中，已观察字节数和 SHA256 与基线一致；这只达到 E1，不代表均可加载。

| ID | 文件 | bytes | SHA256 |
| --- | --- | ---: | --- |
| primary-dit | `10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors` | 20,967,637,320 | `54d56b15...bbc1` |
| text-encoder | `qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors` | 15,683,129,659 | `47babbb3...94c` |
| audio-vae | `minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 | `8e505d95...b48` |
| video-vae | `minimax_h3_video_vae_fp16.safetensors` | 5,207,808,496 | `7c1f1314...522` |
| spatial-physics-lora | `wushu_spatial_physics_clean_3000_pruned.safetensors` | 155,109,672 | `7d14f370...19d` |
| realism-people-lora | `h3-realism-people-t2v-i2v-r2v.safetensors` | 131,229,656 | `acc52960...87e` |
| hot-swap-dit | `minimax_h3_ref2va_pruned_zs05_int8_convrot.safetensors` | 20,970,379,680 | `71b8085a...c5` |

完整摘要必须从 JSON 复制，不得使用表中缩写执行验证。Hugging Face 资产下载时通过 loopback-only
tunnel 使用维护者提供的 VLESS 节点；ModelScope 文本编码器通过国内 ModelScope CDN 直连下载。
没有把代理配置写入系统或仓库。

重要例外：text encoder 原文件尾部有 72 个未索引字节，strict reader 必须拒绝；只能按固定 profile
生成独立规范化副本。预期派生身份是 15,683,129,587 bytes、
`a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f`。不得原地修改源文件或把
“忽略尾部”变成通用开关。

六组件目录已于 E3 固化（见 4.5）：tokenizer/processor 来自官方
`MiniMaxAI/MiniMax-H3@42ed227e` 的 `Ref2VA/` 子目录（基线新增 `tokenizer-config`/
`processor-config` 条目），payload 组件见 §7 表；transformer 使用 PR #25 的 Ref2VA 全量转换
输出树。

## 8. 仍未完成

- E3 证据 PR 的门禁、合并与 Issue #9 评论。
- 单一、lazy、idempotent vLLM-Omni plugin bootstrap 与 CLI/runtime load。
- primary package 的 native load、最小安全提示词视频/音频生成 E4（使用已发布包）。
- Spatial Physics 与 Realism People 的只读 compatibility oracle、支持时的 off/on/off 生命周期。
- primary DiT A、Z-Image-native DiT B 的同进程 `A → B → A` E5。
- 双语用户文档、公开许可证复核、可复现预发布和最终 M7 开源预览版。

当前没有理由宣称整个重构完成。下载完成不等于兼容，转换完成不等于 package/load/generation 完成，
PR 分支绿灯也不等于主线完成。
