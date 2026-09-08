#!/bin/bash

set -euo pipefail

# =============================================================================
# OPD-Aha 自研复刻 (revision_opd Eq.6-14) + plausibility floor — 4B 两臂
#
# Student  : Qwen3.5-4B (trainable), full image
# Teacher  : 同一 4B 初始权重的 frozen copy, 两次 no_grad 前向:
#              p⁺ = teacher(hide 特权图, bbox_images + teacher_prompt)
#              p⁰ = teacher(mean-RGB 同尺寸空白图, null_images + 同一 teacher_prompt)
# 重建     : u = log p⁺ − log p⁰ (学生 top-k+尾桶支持上);
#            log q = log_softmax(log p⁺ + β·u_gated), β=4 (论文最优)
# 损失     : q 喂进与 V0 完全相同的广义 JSD 路径 (alpha=0.5 = 论文 Eq.13)
# floor 臂 : AHA_FLOOR_ALPHA=0.1 → 谷区 (p⁺ < α·max p⁺) 负 u 截 0, 抗 Ren 挤压
# repro 臂 : AHA_FLOOR_ALPHA 不设 (null) → 忠实复刻
#
# 用法:
#   bash opsa/aha/run_aha_4b.sh                       # Aha-repro (β=4, 无 floor)
#   AHA_FLOOR_ALPHA=0.1 bash opsa/aha/run_aha_4b.sh   # Aha+floor
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_NAME="vopd"
MODEL_PATH="Qwen/Qwen3.5-4B"
TEACHER_MODEL_SOURCE="legacy"
TEACHER_REGULARIZATION="ema"
TEACHER_UPDATE_RATE=0.0            # frozen init-copy teacher

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-96}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
LR=2e-6
ALPHA=0.5                          # 广义 JSD = 论文 Eq.13 (½KL(pS‖m)+½KL(q‖m))
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4608}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
TRAIN_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$TRAIN_MAX_MODEL_LEN}"
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}
ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-True}
ACTOR_PPO_MICRO_BS=${ACTOR_PPO_MICRO_BS:-null}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$TRAIN_MAX_MODEL_LEN}
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
ACTOR_PARAM_OFFLOAD=True
ACTIVATION_OFFLOAD=${ACTIVATION_OFFLOAD:-False}
ULYSSES_SP=${ULYSSES_SP:-1}
ACTOR_OPTIMIZER_OFFLOAD=True
REF_PARAM_OFFLOAD=True
TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE:-8}
TRAINER_NNODES=${WORLD_SIZE:-1}
TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-5}
TRAINER_TOTAL_EPOCHS=1
TRAINER_MAX_ACTOR_CKPT_TO_KEEP=null
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","tensorboard","wandb"]'}
ROLLOUT_AGENT_NUM_WORKERS=8
DATA_DATALOADER_NUM_WORKERS=8
CUSTOM_CHAT_TEMPLATE_FILE="${PROJECT_ROOT}/chat_templates/perception_chat_template_qwen35.jinja"

# --- Aha knobs ---
AHA_BETA=${AHA_BETA:-4.0}
AHA_FLOOR_ALPHA=${AHA_FLOOR_ALPHA:-null}   # null=repro 臂; 0.1=floor 臂
FULL_LOGIT=True; TOPK=100                  # aha 硬性要求 (top-k+尾桶支持)

# --- Data ---
TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-${PROJECT_ROOT}/data/TreeVGR-RL-37K/train_sa4k_aha.parquet}"

MODEL_NAME=$(basename "$MODEL_PATH")
if [[ "$AHA_FLOOR_ALPHA" != "null" ]]; then
    EXPERIMENT_NAME="Aha-floor-${MODEL_NAME}"
else
    EXPERIMENT_NAME="Aha-${MODEL_NAME}"
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

echo "Running: $EXPERIMENT_NAME  (aha_beta=$AHA_BETA, floor_alpha=$AHA_FLOOR_ALPHA)"
echo "Teacher: frozen init copy (source=$TEACHER_MODEL_SOURCE, update_rate=$TEACHER_UPDATE_RATE)"

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
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=$FULL_LOGIT \
    actor_rollout_ref.actor.self_distillation.distillation_topk=$TOPK \
    actor_rollout_ref.actor.self_distillation.state_adaptive=False \
    actor_rollout_ref.actor.self_distillation.evt_enable=False \
    actor_rollout_ref.actor.self_distillation.aha_enable=True \
    actor_rollout_ref.actor.self_distillation.aha_beta=$AHA_BETA \
    actor_rollout_ref.actor.self_distillation.aha_floor_alpha=$AHA_FLOOR_ALPHA \
    actor_rollout_ref.actor.self_distillation.null_image_key=null_images \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
    actor_rollout_ref.actor.self_distillation.teacher_model_source=$TEACHER_MODEL_SOURCE \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=$TEACHER_REGULARIZATION \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE \
    actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
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
    trainer.val_before_train=False \
    trainer.default_local_dir=$TRAINER_DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir="$TRAINER_ROLLOUT_DATA_DIR" \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
