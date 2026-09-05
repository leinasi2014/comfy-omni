# ComfyOmni 用户指南

> 用于使用既有 ComfyUI H3 资产的 vLLM-Omni 插件。

<!-- guide-parity -->
> 英文与中文用户指南保持同步。两者陈述相同的产品事实和链接，只有正文语言不同。

## 它是什么

ComfyOmni 是一个 `vllm_omni.general_plugins` 单插件。首期目标是加载、卸载和切换已有 ComfyUI 安装中的 H3 模型与组件文件，在 RAM/VRAM 中管理它们，不要求离线转换、新建 BF16 模型副本或完整包组装。LoRA 组合、工具和节点工作流放到后续，不阻塞首期交付。

它是独立的 Apache-2.0 项目，不是 ComfyUI、vLLM 或 MiniMax 的官方项目。

## 当前使用

插件入口为 `comfy_omni.plugin:register`。它只会在 vLLM-Omni 已加载时进行惰性注册；导入时不会启动服务、下载资产、创建模型副本或加载权重。参见 [bootstrap 记录](migration/vllm-omni-bootstrap-e9cb011.md)。

当前代码提供一个插件集成、用于描述既有 H3 资产的组件目录路径，以及经过审计的旧版 H3 v3 curve-cache recipe 入口。旧入口是针对其已记录输入布局的已验证兼容路径；它不使转换或打包成为日常 serving 的组成部分。直接加载既有 H3 原始量化 A 格式权重是另一条尚未验证的路径。参见 [H3 cache-runtime 记录](migration/h3-cache-runtime-e9cb011.md)。

目标使用已配置 ComfyUI 安装中已有的模型与组件文件，在部署验证期间保持它们只读。加载和切换在同一控制服务实例及其已有 worker 中管理 RAM/VRAM，复用未变化的组件。worker 重建只作为明确报告的恢复或降级路径，不计作正常热加载验收通过。

## 首期范围

下列内容是 H3-first 产品目标，但目前还不是面向用户的命令或受支持部署合同：

| 范围 | 状态 |
|---|---|
| H1：既有 H3 原始文件直接加载 | 实现及验收尚未完成 |
| H2：RAM/VRAM 驻留、组件复用及已有 worker 内 A → B → A | 尚未验收 |
| H3：真实宿主加载、生成、切换与交付 | 旧兼容路径已有证据；新直接加载路径尚未验收 |

请勿从此表推断 HTTP 路由、CLI 命令、环境变量、模型格式或切换工作流。正常流程仍仅限于已经完成集成并针对所选宿主验证的能力。

## 后续扩展

完整内存 LoRA 组合、H3 工具和有类型节点工作流放到后续，不是首期验收要求。本指南不宣称任意 ComfyUI 工作流或第三方节点已经受支持。

## 边界

仓库不重新分发模型权重、LoRA payload、生成包或服务器证据。保留的代码归属与许可记录见 [source attribution](migration/source-attribution.md)，它们不是模型准备教程。

开发与宿主验证规则见 [Docker-first development](development/docker-first.md) 和 [model-validation baseline](testing/model-validation-baseline.md)。
