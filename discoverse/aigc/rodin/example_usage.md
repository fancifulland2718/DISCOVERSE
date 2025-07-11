# Rodin 3D模型生成工具使用示例

## 📋 功能概览

这个工具支持三种运行模式，可以灵活地管理3D模型生成任务。

## 🚀 典型工作流程

### 方案1: 分离式工作流（推荐）

适合批量生成和异步处理的场景。

```bash
# 步骤1: 批量提交生成任务
export RODIN_API_KEY=your_api_key_here
python workflow_text-to-3d.py --mode generate

# 步骤2: 稍后检查并下载完成的模型
python workflow_text-to-3d.py --mode download

# 步骤3: 重复步骤2直到所有任务完成
python workflow_text-to-3d.py --mode download
```

**优点**：
- 可以批量提交多个任务
- 不需要长时间等待
- 可以随时中断和恢复
- 支持增量下载

### 方案2: 一体式工作流

```bash
# 提交任务并等待下载
export RODIN_API_KEY=your_api_key_here
python workflow_text-to-3d.py --mode both
```