unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset no_proxy

#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_dsv4_flash_8k1k_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Writing log to: ${LOG_FILE}"

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32
export INF_NAN_MODE_FORCE_DISABLE=1
export SGLANG_SET_CPU_AFFINITY=1
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV

export HCCL_BUFFSIZE=1500
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

# skip gpu branch
export SGLANG_OPT_FP8_WO_A_GEMM=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export FORCE_DRAFT_MODEL_NON_QUANT=1
export SGLANG_DSV4_FP4_EXPERTS=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False

# mtp
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1

# profiling 
# export SGLANG_NPU_PROFILING=False
# export SGLANG_NPU_PROFILING_BS=10
# export SGLANG_NPU_PROFILING_STEP=10
# export SGLANG_NPU_PROFILING_STAGE="decode"

#export HCCL_BUFFSIZE=8
#unset PYTORCH_NPU_ALLOC_CONF
#export SGLANG_ZBAL_LOCAL_MEM_SIZE=61184
#export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
#export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
#export SGLANG_ZBAL_BOOTSTRAP_URL="tcp://192.168.41.147:14699"
#export ZBAL_ENABLE_GRAPH=1

# path
cd ~
export PYTHONPATH=/home/t00937989/sglang-pd/python:$PYTHONPATH
MODEL_PATH=/home/weights/DeepSeek-V4-Flash-w8a8-mtp

python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
    --page-size 128 \
    --tp-size 8 \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host 0.0.0.0 --port 30000 \
    --mem-fraction-static 0.85 \
    --prefill-max-requests 1 \
    --disable-radix-cache --chunked-prefill-size 8192 \
    --max-running-requests 32 \
    --dp-size 8 --enable-dp-attention \
    --moe-a2a-backend deepep --deepep-mode auto \
    --quantization modelslim --enable-dp-lm-head \
    --kv-cache-dtype bfloat16 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 2 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 3

exit 1

python3 -m sglang.bench_serving \
    --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --dataset-name random \
    --backend sglang \
    --host 0.0.0.0 \
    --port 30000 \
    --max-concurrency 160 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --num-prompts 160 \
    --disable-ignore-eos \
    --random-range-ratio 1 \
    --warmup-requests 0
