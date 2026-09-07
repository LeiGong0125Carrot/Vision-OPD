#!/bin/bash

set -euo pipefail

# =============================================================================
# 视觉域 OPSA baseline — Qwen3.5-4B, 6K armA 高清 2459 题 (arXiv 2608.31046)
#
# 零监督零特权: 无 teacher 模块 (完全不加载) / 无 reward / 无标签 / 不读 hide 列。
# 数据只消费 prompt + 原始高清全图。advantage = 熵自适应负值 ([-1,-0.5]) 只作用于
# batch 级 lowest-20% 学生 logp token, PPO clipped loss。
# 对齐官方 run_opsa.sh: n=1, batch 64, lr 1e-6 constant, 无 IS 权重, 无 KL。
# 高清配置对齐 Arm A 先例 (62_v06k_resume.sbatch): prompt 8192 / ulysses 2 /
# activation offload / micro_bs=1。
#
# 用法 (interactive 双卡):
#   bash opsa/run_opsa_4b.sh
# 干跑冒烟 (2 步):
#   TOTAL_TRAIN_STEPS=2 EXPERIMENT_SUFFIX=dryrun bash opsa/run_opsa_4b.sh
# fixed 模式消融:
#   OPSA_MODE=fixed OPSA_FIXED_ADV=-0.5 EXPERIMENT_SUFFIX=fixed bash opsa/run_opsa_4b.sh
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_NAME="vopd"
MODEL_PATH="Qwen/Qwen3.5-4B"

# --- 官方 OPSA 配方 ---
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}   # = train_batch_size -> on-policy, ratio==1
ROLLOUT_N=${ROLLOUT_N:-1}                        # OPSA 不需要组内比较
LR=${LR:-1e-6}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}            # 官方 constant lr
OPSA_MODE=${OPSA_MODE:-entropy}
OPSA_TOKEN_FRACTION=${OPSA_TOKEN_FRACTION:-0.2}
OPSA_ADV_MIN=${OPSA_ADV_MIN:--1.0}
OPSA_ADV_MAX=${OPSA_ADV_MAX:--0.5}
OPSA_FIXED_ADV=${OPSA_FIXED_ADV:-null}

ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
# --- 6K armA 高清配置 (Arm A 先例) ---
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048} # 给 response 变长机制留空间 (原 1024)
TRAIN_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$TRAIN_MAX_MODEL_LEN}"
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}
ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-False}
ACTOR_PPO_MICRO_BS=${ACTOR_PPO_MICRO_BS:-1}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$TRAIN_MAX_MODEL_LEN}
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
ACTOR_PARAM_OFFLOAD=True
ACTIVATION_OFFLOAD=${ACTIVATION_OFFLOAD:-True}
ULYSSES_SP=${ULYSSES_SP:-2}
ACTOR_OPTIMIZER_OFFLOAD=True
REF_PARAM_OFFLOAD=True
TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE:-2}
TRAINER_NNODES=${WORLD_SIZE:-1}
TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-10}
TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-3}  # 2459 行 / 64 ≈ 38 步/epoch, 3 epoch ≈ 115 步
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-null}     # 干跑冒烟用: 设 2
TRAINER_MAX_ACTOR_CKPT_TO_KEEP=null
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","tensorboard"]'}
ROLLOUT_AGENT_NUM_WORKERS=8
DATA_DATALOADER_NUM_WORKERS=8
CUSTOM_CHAT_TEMPLATE_FILE="${PROJECT_ROOT}/chat_templates/perception_chat_template_qwen35.jinja"

TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-${PROJECT_ROOT}/data/TreeVGR-RL-37K/train_6k_armA_hide.parquet}"

MODEL_NAME=$(basename "$MODEL_PATH")
EXPERIMENT_NAME="OPSA-${MODEL_NAME}"
if [[ "$OPSA_MODE" == "fixed" ]]; then
    EXPERIMENT_NAME="OPSAfix-${MODEL_NAME}"
fi
if [[ -n "${EXPERIMENT_SUFFIX:-}" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-${EXPERIMENT_SUFFIX}"
fi
PROJECT_NAME="Vision-OPD-StateAdaptive"
TRAINER_DEFAULT_LOCAL_DIR="${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}"
TRAINER_ROLLOUT_DATA_DIR="${PROJECT_ROOT}/rollouts/${EXPERIMENT_NAME}"
mkdir -p "$TRAINER_ROLLOUT_DATA_DIR"

EXTRA_ARGS=("$@")

# =============================================================================
# ENVIRONMENT
# =============================================================================
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
export USER="${USER:-$(id -un 2>/dev/null || echo root)}"
ulimit -c 0

CHAT_TEMPLATE_ARGS=()
if [[ -n "${CUSTOM_CHAT_TEMPLATE_FILE}" ]]; then
    if [[ ! -f "${CUSTOM_CHAT_TEMPLATE_FILE}" ]]; then
        echo "Custom chat template file not found: ${CUSTOM_CHAT_TEMPLATE_FILE}" >&2
        exit 1
    fi
    CHAT_TEMPLATE_ARGS+=(actor_rollout_ref.model.custom_chat_template_file="$CUSTOM_CHAT_TEMPLATE_FILE")
fi

echo "Running: $EXPERIMENT_NAME  (mode=$OPSA_MODE frac=$OPSA_TOKEN_FRACTION adv=[$OPSA_ADV_MIN,$OPSA_ADV_MAX])"

python3 -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" \
    data.train_files="[\"$TASK_TRAIN_FILE\"]" \
    data.val_files="[]" \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.truncation=error \
    data.shuffle=True \
    data.trust_remote_code=True \
    data.return_multi_modal_inputs=True \
    data.image_key=images \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.dataloader_num_workers=$DATA_DATALOADER_NUM_WORKERS \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
    "actor_rollout_ref.actor.optim.betas=[0.9,0.98]" \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_PPO_MICRO_BS \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.model.enable_activation_offload=$ACTIVATION_OFFLOAD \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES_SP \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ULYSSES_SP \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
    actor_rollout_ref.actor.calculate_entropy=True \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=False \
    actor_rollout_ref.actor.self_distillation.distillation_topk=null \
    actor_rollout_ref.actor.self_distillation.opsa_enable=True \
    actor_rollout_ref.actor.self_distillation.opsa_mode=$OPSA_MODE \
    actor_rollout_ref.actor.self_distillation.opsa_token_fraction=$OPSA_TOKEN_FRACTION \
    actor_rollout_ref.actor.self_distillation.opsa_advantage_min=$OPSA_ADV_MIN \
    actor_rollout_ref.actor.self_distillation.opsa_advantage_max=$OPSA_ADV_MAX \
    actor_rollout_ref.actor.self_distillation.opsa_fixed_advantage=$OPSA_FIXED_ADV \
    actor_rollout_ref.actor.self_distillation.evt_enable=False \
    actor_rollout_ref.actor.self_distillation.state_adaptive=False \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
    actor_rollout_ref.actor.self_distillation.is_clip=null \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=False \
    actor_rollout_ref.actor.self_distillation.teacher_image_key=null \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_AGENT_NUM_WORKERS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    reward_model.enable=False \
    critic.model.path=$MODEL_PATH \
    reward_model.use_reward_loop=False \
    custom_reward_function.path=null \
    trainer.project_name=$PROJECT_NAME \
    trainer.group_name=$EXPERIMENT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.n_gpus_per_node=$TRAINER_N_GPUS_PER_NODE \
    trainer.nnodes=$TRAINER_NNODES \
    trainer.save_freq=$TRAINER_SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.max_actor_ckpt_to_keep=$TRAINER_MAX_ACTOR_CKPT_TO_KEEP \
    trainer.total_epochs=$TRAINER_TOTAL_EPOCHS \
    trainer.total_training_steps=$TOTAL_TRAIN_STEPS \
    trainer.val_before_train=False \
    trainer.default_local_dir=$TRAINER_DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir="$TRAINER_ROLLOUT_DATA_DIR" \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
