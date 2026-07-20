unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset no_proxy

#!/bin/bash
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

# path
export PYTHONPATH=/home/kelon/code/dsv4/sglang/python:$PYTHONPATH
MODEL_PATH=/home/weights/DeepSeek-V4-Flash-w8a8-mtp-ms

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1

P_IP=('192.168.25.209')
D_IP=('192.168.25.212')

LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

export ASCEND_MF_STORE_URL="tcp://192.168.25.209:24669"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=60


for i in "${!P_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]];
    then
        echo "Prefill -> ${P_IP[@]}"

        export HCCL_BUFFSIZE=8
        unset PYTORCH_NPU_ALLOC_CONF
        export SGLANG_ZBAL_LOCAL_MEM_SIZE=62084
        export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
        # zbccl if use mix alloc
        export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
        export ZBAL_NPU_ALLOC_CONF=use_vmm_for_static_memory:True
        export SGLANG_ZBAL_BOOTSTRAP_URL="tcp://192.168.25.209:14699"
        # zbccl if support graph   [m~Hneed custom pta   [m~I
        export ZBAL_ENABLE_GRAPH=1
	# 强制负载均衡
	#export SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS=1

        python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
            --page-size 128 \
            --tp-size 16 \
            --trust-remote-code \
            --device npu \
            --attention-backend dsv4 \
            --watchdog-timeout 9000 \
            --host ${P_IP[$i]} --port 30000 \
            --mem-fraction-static 0.62 \
            --prefill-max-requests 7 \
            --max-prefill-tokens 80000 \
            --chunked-prefill-size -1 \
            --max-running-requests 112 \
            --dp-size 16 --enable-dp-attention \
            --moe-a2a-backend deepep --deepep-mode normal \
            --quantization modelslim --enable-dp-lm-head \
            --kv-cache-dtype bfloat16 \
            --disable-cuda-graph \
            --disable-radix-cache

        exit 1
    fi
done


