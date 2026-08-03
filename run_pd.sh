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

        python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
            --page-size 128 \
            --tp-size 16 \
            --trust-remote-code \
            --device npu \
            --attention-backend dsv4 \
            --watchdog-timeout 9000 \
            --host ${P_IP[$i]} --port 30000 \
            --disaggregation-mode prefill --disaggregation-transfer-backend ascend \
            --disaggregation-bootstrap-port $((8998+$i)) \
            --mem-fraction-static 0.62 \
            --prefill-max-requests 6 \
            --max-prefill-tokens 70000 \
            --chunked-prefill-size -1 \
            --max-running-requests 112 \
            --dp-size 16 --enable-dp-attention \
            --moe-a2a-backend deepep --deepep-mode normal \
            --quantization modelslim --enable-dp-lm-head \
            --kv-cache-dtype bfloat16 \
            --disable-cuda-graph \
            --disable-radix-cache \
	        --load-balance-method round_robin \
            --ep-dispatch-algorithm static --init-expert-location /home/cjr/eplb_prefill_heat/pd_prefill_0720.pt \


        exit 1
    fi
done

for i in "${!D_IP[@]}";
do
    if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]];
    then
        echo "Decode -> ${D_IP[$i]}"

        export HCCL_BUFFSIZE=1200
        export DEEPEP_NORMAL_LONG_SEQ_ROUND=8
        export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
        export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
        export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

        python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
            --page-size 128 \
            --tp-size 16 \
            --trust-remote-code \
            --device npu \
            --attention-backend dsv4 \
            --watchdog-timeout 9000 \
            --host 0.0.0.0 --port 30000 \
            --mem-fraction-static 0.8 \
            --prefill-max-requests 1 \
            --disable-radix-cache --chunked-prefill-size 32768 \
            --disaggregation-mode decode --disaggregation-transfer-backend ascend \
            --max-running-requests 896 \
            --dp-size 16 --enable-dp-attention \
            --moe-a2a-backend deepep --deepep-mode auto \
            --quantization modelslim --enable-dp-lm-head \
            --kv-cache-dtype bfloat16 \
            --cuda-graph-bs 1 2 4 8 16 24 36 40 48 56\
            --speculative-algorithm EAGLE \
            --speculative-num-steps 2 \
            --speculative-eagle-topk 1 \
            --speculative-num-draft-tokens 3

        exit 1
    fi
done

exit 1

python -m sglang_router.launch_router \
    --pd-disaggregation --policy cache_aware \
    --prefill http://192.168.25.209:30000 8998 \
    --decode http://192.168.25.212:30000 \
    --host 0.0.0.0 --port 6688

python3 -m sglang.bench_serving \
    --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --dataset-name random \
    --backend sglang \
    --host 192.168.25.209 \
    --port 6688 \
    --max-concurrency 128 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --num-prompts 128 \
    --disable-ignore-eos \
    --random-range-ratio 1 \
    --warmup-requests 0

python3 -m sglang.bench_serving \
    --dataset-path /home/zkk/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --dataset-name random \
    --backend sglang \
    --host 192.168.25.209 \
    --port 6688 \
    --max-concurrency 768 \
    --random-input-len 8000 \
    --random-output-len 1000 \
    --num-prompts 3072 \
    --disable-ignore-eos \
    --random-range-ratio 1 \
    --warmup-requests 0

#--ep-dispatch-algorithm static --init-expert-location /data/cjr/eplb_prefill_heat/expert_distribution_recorder_1784515249.206882.pt \
#--expert-distribution-recorder-buffer-size 2048 \
#--eplb-rebalance-num-iterations 2048 \
#--expert-distribution-recorder-mode stat
