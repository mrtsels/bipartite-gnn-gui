# VLM 推理指南

## 概述

本指南说明如何生成供训练管线使用的 VLM 预测。

## 生成预测

当您拥有某组截图的 VLM 预测时,训练管线可将其作为输入。脚本 `scripts/generate_vlm_predictions.py` 负责整个生成流程。

### 试运行 — 查看脚本将处理哪些文件

使用 `--dry-run` 标志预览文件:

```bash
python scripts/generate_vlm_predictions.py --dry-run
```

### 输出格式

脚本将每个预测 JSON 保存到输出目录。输出格式遵循 `docs/requirements/vlm_format.md` 中的约定。

### 追踪

管线会记录其使用的真实 VLM 文件,确保实验日志可复现。

### 错误处理

- API 调用失败时,脚本最多重试 3 次。
- 已有输出文件的图片会被跳过。
- 所有重试后仍失败的 API 调用进入单独的错误日志。
