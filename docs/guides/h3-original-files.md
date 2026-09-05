# H3 原文件运行（首期）

ComfyOmni 是 vLLM-Omni 插件。本首期直接使用已有的 H3 共享组件目录和
两份只读原始 ConvRot DiT 文件，在进程内加载与切换；不创建 BF16 导出包、
不复制模型，也不需要持久化 LoRA 合并缓存。

当前支持一个 diffusion stage（`stage_id=0`）和一个 stage replica，DiT 内部
使用 TP2。节点图、LoRA 和工具调用属于后续能力，不是本说明的运行接口。

## 准备已有文件

组件根必须是已有的 `Ref2VA` 目录，含 `model_index.json`、text encoder、
VAE、tokenizer 和 processor。它继续提供共享组件；原始 DiT 不从该目录的
`transformer` 分片加载。

示例使用以下已存在的挂载路径：

```text
/models/shared/Ref2VA/
/models/comfy/diffusion_models/10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors
/models/comfy/diffusion_models/minimax_h3_ref2va_pruned_zs05_int8_convrot.safetensors
```

第一个文件是 A（`h3-beta4-convrot`），第二个是 B（`h3-pruned-convrot`）。
运行时会在首次建立 binding 时认证固定来源；后续读取检查文件和 header
身份。请把它们作为只读文件挂入运行容器。

## 启动配置

将下列字段传给 pinned vLLM-Omni 的 `AsyncOmni`。`additional_config` 的键和
`format` 值是严格契约，未知字段或格式会被拒绝。

```python
from vllm_omni.entrypoints.async_omni import AsyncOmni

engine = AsyncOmni(
    model="/models/shared/Ref2VA",
    num_gpus=2,
    tensor_parallel_size=2,
    text_encoder_tp_size=2,
    data_parallel_size=1,
    worker_extension_cls=(
        "comfy_omni.integrations.vllm_omni.residency.H3ResidencyWorkerExtension"
    ),
    diffusion_quantization_config={
        "default": None,
        "text_encoder": {"method": "int8"},
    },
    additional_config={
        "comfy_omni_h3": {
            "active": "a",
            "sources": {
                "a": {
                    "path": "/models/comfy/diffusion_models/"
                            "10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors",
                    "format": "h3-beta4-convrot",
                },
                "b": {
                    "path": "/models/comfy/diffusion_models/"
                            "minimax_h3_ref2va_pruned_zs05_int8_convrot.safetensors",
                    "format": "h3-pruned-convrot",
                },
            },
        },
    },
)
```

`worker_extension_cls` 必须是全限定字符串，避免将 Python class 对象放进要
序列化到 worker 的配置。首期只对 text encoder 启用 INT8；不要沿用旧路径的
transformer INT8 配置。

运行时骨干槽按宿主契约使用 BF16，时间表和 AdaLN 敏感槽使用 FP32。B 原始
来源中的普通 F16/F32 copy 保持原生 dtype，直到存在明确的宿主槽 dtype 适配；
这不是“任意模型自动读入”或通用转换承诺。

原文件路径的音频解码使用 cuDNN 确定性算法，并保留原有 dtype、TF32 和
cuDNN 启用设置；调用结束或异常后恢复后端设置。这使同一输入的回切比较具有
稳定的执行条件，但不承诺与未限制算法的旧路径产生逐字节相同的音频。
设置 `COMFY_OMNI_H3_AUDIO_TRACE=1` 可在诊断日志中记录解码输入和输出摘要；
默认关闭，不保存音频或模型 payload。

## 运行与切换

在已挂载的容器中，可先用验收 runner 只检查入口。输出目录必须是新的、独立
目录，不能位于组件或原文件树内：

```bash
python3 scripts/acceptance/h3_raw_runtime.py \
  --stage load --out /results/h3-raw-load \
  --component-root /models/shared/Ref2VA \
  --source-a /models/comfy/diffusion_models/10Eros_Max_h3_TURBO-hybrid_beta4_int8_convrot.safetensors \
  --source-b /models/comfy/diffusion_models/minimax_h3_ref2va_pruned_zs05_int8_convrot.safetensors \
  --prompt /inputs/prompt.txt --reference /inputs/reference.png
```

`--stage forward` 运行 A 的单次请求；`--stage aba` 在同一 `AsyncOmni` 进程
中执行 A→B→A；`--stage full` 再执行 release unload/load 和一次 A 请求。
这些阶段是验收工具，不表示 GPU 验收已经通过。

宿主 API 挂载后提供五个运行时路由：

```text
GET  /v1/comfy-omni/h3/runtime?stage_id=0
POST /v1/comfy-omni/h3/runtime/switch  {"stage_id":0,"selection":"b","cpu_cache_budget_bytes":0}
POST /v1/comfy-omni/h3/runtime/unload  {"stage_id":0,"mode":"release","cpu_budget_bytes":0}
POST /v1/comfy-omni/h3/runtime/load    {"stage_id":0}
POST /v1/comfy-omni/h3/runtime/resume  {"stage_id":0,"expected_active_selection":"a"}
```

`unload` 默认 `release`，CPU cache 预算为 0；`mode:"cpu"` 时必须显式提供
足以容纳当前 rank 权重的 `cpu_budget_bytes`。切换的 `cpu_cache_budget_bytes`
则控制完整逻辑输入张量缓存，两者默认都为 0。状态中的字节数是每个 replica 的
reporting rank，不是 TP 总量，不能自行相加后当作全局显存结论。

切换会暂停新生成请求、等待所有 rank 空闲，再在所有 rank 上 prepare、commit
和 finalize。任何阶段失败都会保留暂停状态且不回退到另一条模型路径；修复后
先检查 `GET` 状态，确认目标 selection 已加载且健康，再用 `resume` 明确恢复。
