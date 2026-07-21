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

export HCCL_BUFFSIZE=2000
export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
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

# path
# export PYTHONPATH=/home/zkk/sglang/python:$PYTHONPATH
MODEL_PATH=/home/weights/DeepSeek-V4-Pro-w4a8-mtp

D_IP=(192.168.25.209 192.168.25.212)
LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "${LOCAL_HOST1}"
echo "${LOCAL_HOST2}"

for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "Decode -> ${D_IP[$i]}"

        export HCCL_SOCKET_IFNAME=enp196s0f0
        export GLOO_SOCKET_IFNAME=enp196s0f0

        python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
            --page-size 128 \
            --tp-size 32 \
            --trust-remote-code \
            --device npu \
            --attention-backend dsv4 \
            --watchdog-timeout 9000 \
            --host ${D_IP[$i]} --port 30000 \
            --dist-init-addr ${D_IP[0]}:5000 --nnodes 2 --node-rank $i \
            --mem-fraction-static 0.7 \
            --prefill-max-requests 1 \
            --disable-radix-cache --chunked-prefill-size -1 \
            --max-running-requests 4 \
            --disable-overlap-schedule \
            --dp-size 4 --enable-dp-attention \
            --moe-a2a-backend deepep --deepep-mode auto \
            --quantization modelslim --enable-dp-lm-head \
            --kv-cache-dtype bfloat16 \
            --cuda-graph-bs 1 2 4 8 10

        exit 1
    fi
done

exit 1

python3 -m sglang.bench_serving \
    --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --dataset-name random \
    --backend sglang \
    --host 192.168.25.209 \
    --port 30000 \
    --max-concurrency 4 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --num-prompts 4 \
    --disable-ignore-eos \
    --random-range-ratio 1 \
    --warmup-requests 0
