#!/bin/bash

#SBATCH --job-name=material_agent-p100

#SBATCH --partition=gpu_p100

#SBATCH --nodes=1

#SBATCH --gres=gpu:1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --time=04:00:00

#SBATCH --output=log/material_agent_p100.out

# 加载正确的模块层级（非常关键）
module load cluster/genius/gpu_p100

# 加载 ollama
module load ollama/0.6.0-GCCcore-13.3.0

# 启动 Ollama 服务
ollama serve > ollama_server.log 2>&1 &

sleep 8

# 下载模型
ollama pull llama3.1

# 启用Python
module load Python/3.10.8-GCCcore-12.2.0

source PythonEnv/bin/activate

# 运行程序
python invoke.py
