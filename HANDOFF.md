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
- 当前受保护主线：`main@e088740f3d647db819329240cdac863f82dda356`

迁移代码进入 PR 前必须记录旧文件的精确 commit/blob、许可证、归属、保留行为、特征测试证据和
distribution disposition。不要参考旧项目中比上述提交更新、未提交或其他分支的实现。

## 3. 不可违反的执行规则

- 项目 Python、pytest、Ruff、构建、打包、依赖安装、下载、模型解析、转换和推理全部在 Docker 中运行。
- 本机只允许编辑/读取文件、Git/GitHub、Docker 编排、SSH/SCP 和只读诊断；本机 Docker 当前不可用，
  因此普通门禁以 GitHub Actions 的 Docker job 为准。
- `srv-00` 上同样 Docker-first；不得直接向宿主安装 Python 包、修改系统代理或污染系统环境。
- 模型源以只读 mount 进入容器；输出 staging-first、独立验证、manifest-last，禁止覆盖既有产物。
- 不得把模型、wheel、缓存、服务器证据、生成媒体、私有基础设施或密钥提交到 Git。
- 不得在文档中记录完整 VLESS URI、UUID 或其他凭据。本项目只记录“使用维护者提供的 VLESS 节点”。
- 依赖方向固定为：
  `core -> domain/contracts/artifacts -> conversion/runtime -> application -> CLI/API/integrations`。
- import `comfy_omni` 或插件入口不得加载 Torch、FastAPI、vLLM、模型或 checkpoint payload。
- Git 流程：单 WIP；冻结 READY；先提交 RED；观察 Docker 红灯；GREEN；REFACTOR；短分支；PR；
  全绿后 squash merge；最后必须等待 `main` push 自身的 Docker 回读。
- 开发依据 `D:\skills\manage-agile-software-development\SKILL.md`；服务器操作依据适用的 SSH/srv skill。

## 4. 已完成并进入主线

### 4.1 仓库、规范与门禁

独立仓库、`src/comfy_omni` 布局、双语 README、AGENTS/CONTRIBUTING、Docker-first 规范、GitHub
protected-main 工作流、Python 3.10/3.13 quality、3.12 package/install smoke 和 documentation contract
均已建立。README 路线图中的状态文字仍较保守，不得仅根据表格推断实际完成度；GitHub Issues、合并
提交和证据文档才是当前交付状态权威。

### 4.2 Ref2VA 完整转换（PR #25，Issue #8）

- 合并提交：`27c462f243eff4748a9c6d584cabe0af713af959`
- 服务器证据：`docs/evidence/ref2va-full-conversion-25ceccdd5468.md`
- 候选代码：`25ceccdd54680a5e32ea1574974c138d39d08bd6`
- 输入：`minimax_h3_ref2va_pruned_zs05_int8_convrot.safetensors`
- 输入身份：20,970,379,680 bytes；
  `71b8085ac4221ee036708c230a007d617dccca1b0028b95bb4ee106cb2a385c5`
- 结果：932 actions、532 tensors、10 shards、40,225,668,192 target bytes。
- 完整转换耗时 636.593 秒，峰值 RSS 1,249,308,672 bytes。
- 独立 verifier：`VERIFIED`；14 个输出文件；330 raw-copy、2 QKV reorder、200 ConvRot tensors；
  600 个抽样行、4,838,400 elements；最大 BF16 绝对误差 0.0060174393，在冻结容差内。
- GPU：NVIDIA CMP 170HX，compute capability 8.0，68,212,293,632 bytes，CUDA 13.0。
- Docker 策略审计：5 个容器通过；仅 conversion 容器获得 GPU 0；无 Docker socket/宽泛 mount；
  结束后残留容器为 0。
- 主线回读：quality/package run `33676167336`，documentation run `33676167338`。

服务器证据仍保留在：
`/home/hyl/comfy-omni-acceptance/ref2va-full-conversion-25ceccdd5468-attempt3`。

### 4.3 原生包规划（PR #26，Issue #9 slice 1）

- 合并提交：`9d25ff16e91aff5f36da1e21c9e3ecf6948c3170`
- 文档：`docs/migration/native-package-planning-e9cb011.md`
- 固定六组件：`transformer`、`text_encoder`、`video_vae`、`audio_vae`、`tokenizer`、`processor`。
- 固定 host commit：`17285c2f55a41bf15772676121814d59a60ace35`。
- 输出合同：`h3-comfy-package/v3`、`h3-comfy-package.json`、`Ref2VA/`、单驻留 DiT、
  `ref2va|t2va|fl2va`。
- RED run `33676906712`；GREEN runs `33677384662` / `33677384784`。
- 主线回读：`33677637592` / `33677637532`。

### 4.4 原生包源验证（PR #27，Issue #9 slice 2）

- 合并提交：`e088740f3d647db819329240cdac863f82dda356`
- 文档：`docs/migration/native-package-source-verification-e9cb011.md`
- 已实现完整 plan 重建、自摘要复核、精确树 census、link/reparse/special entry 拒绝、
  pinned-descriptor 流式 SHA256 和 immutable verification result。
- RED run `33678019449`：1 failed / 204 passed。
- GREEN runs `33678567807` / `33678567901`：211 passed。
- 文档提交回读：`33678832949` / `33678832936`。
- 主线回读：`33678931432` / `33678931474`。

Issue #9 必须保持打开，直到 package writing、独立 output verification、atomic publication 和固定组件
真实组包完成。

## 5. 当前唯一 WIP：PR #28

- 分支：`codex/feat-package-materialization`
- PR：<https://github.com/leinasi2014/comfy-omni/pull/28>（当前仍是 draft）
- READY：<https://github.com/leinasi2014/comfy-omni/issues/9#issuecomment-5515909283>
- RED commit：`1471ba5b296e94892c951155782bf8157bd60ae8`
- RED quality run：`33679298001`，两条 Python 线均为唯一缺失模块失败，`1 failed, 211 passed`；
  package/install smoke 通过。
- GREEN implementation commit：`dd369fe5d241fa70ce6c5e536981d7fb0e62922d`
- GREEN quality run：`33680349335`，Python 3.10/3.13、220 tests、package/install smoke 全绿。
- GREEN documentation run：`33680349346`，通过。

当前实现内容：

- `artifacts.fileops.copy_file_pinned_exclusive`：8 MiB 有界复制、源 descriptor 身份前后检查、
  exclusive target、fsync、目标路径身份检查和独立回读。
- `conversion.packaging.materialization.materialize_package`：先复核 plan/source，拒绝输出覆盖和
  source/output overlap，在输出父目录建立私有 sibling staging，复制精确目标，复核 staging census，
  返回 `STAGED_VERIFIED` 不可变结果。
- 失败时最终 output 不出现；私有 staging 保留作诊断，不执行不安全的递归清理。
- 测试覆盖正常暂存、既有输出、路径重叠、预验证后源漂移、linked source、复制中断、意外 staging
  文件、目标碰撞和同尺寸源改写。

本交接文档提交后 PR head 会变化，`dd369fe` 的绿灯仍是行为证据，但新 head 必须重新通过 Docker
门禁后才能继续。

## 6. 下一位执行者的精确续接步骤

1. 确认位于 `codex/feat-package-materialization`，工作树只包含预期的交接提交。
2. 等待 PR #28 最新 head 的 quality/documentation Docker checks；禁止用旧 head 绿灯代替。
3. 对失败只做最小修复。若全绿，新增
   `docs/migration/native-package-materialization-e9cb011.md`，记录：
   - `h3-forge@e9cb011...`；
   - `package_assembler.py` blob `e64558f1d3bb6e1ee6f714b70e783d9df907f9ce`；
   - `fsops.py` blob `ae40e46eef808f979ee085e806f2380e50b6c01d`；
   - RED/GREEN runs、保留行为、许可证/归属、distribution disposition。
4. 更新 PR #28 body，列出 READY、RED、GREEN、REFACTOR、非目标和回滚；转 ready。
5. 等文档 head 的 Docker checks 全绿后 squash merge PR #28。
6. 等 `main` push 自身 quality/package/documentation 回读；把合并 commit 和 run URL 评论到 Issue #9。
7. 新建下一个短分支并先在 Issue #9 冻结 READY。推荐下一切片仅做：生成 package manifest/model index、
   独立重读 staged output、manifest-last 写入和同父目录原子 rename；不要同时加入 CLI 或真实模型。
8. 上述完整 publication 合并后，再用固定六组件在 `srv-00` Docker 中做真实 package E3；随后才进入
   单一 vLLM-Omni bootstrap、native load/minimal generation、LoRA、`A → B → A`。

不得在本机运行 `python`、`pip`、`pytest` 或 `ruff`。常用只读/编排命令：

```text
git status --short
git log --oneline --decorate -8
gh pr view 28 --json headRefOid,isDraft,mergeStateStatus,statusCheckRollup,url
gh pr checks 28 --watch
gh run list --branch codex/feat-package-materialization --limit 6
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

## 8. 仍未完成

- PR #28 的 provenance 文档、ready、合并和主线回读。
- package manifest/model index、独立 output verifier、manifest-last atomic publication。
- 固定六组件真实组包 E3。
- 单一、lazy、idempotent vLLM-Omni plugin bootstrap 与 CLI/runtime load。
- primary package 的 native load、最小安全提示词视频/音频生成 E4。
- Spatial Physics 与 Realism People 的只读 compatibility oracle、支持时的 off/on/off 生命周期。
- primary DiT A、Z-Image-native DiT B 的同进程 `A → B → A` E5。
- 双语用户文档、公开许可证复核、可复现预发布和最终 M7 开源预览版。

当前没有理由宣称整个重构完成。下载完成不等于兼容，转换完成不等于 package/load/generation 完成，
PR 分支绿灯也不等于主线完成。
