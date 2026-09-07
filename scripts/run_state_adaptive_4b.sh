#!/bin/bash

set -euo pipefail

# =============================================================================
# State-Adaptive OPD-Aha — 4B pilot  (TreeVGR/09_01_plan.md, v1: 无 sharpening)
#
# Student  : Qwen3.5-4B (trainable), full image
# Teacher  : 同一 4B 初始权重的 frozen copy (legacy + ema + update_rate=0 → ref module)
#            两次前向: H = region 特权视图 (bbox_images + teacher_prompt),
#                      F = 学生自己的 full-image 输入 (loss 内部自动做, 无需数据准备)
# 数据     : train_sa4k.parquet (RL-37K 任务加权 4k, 文本两侧全裸, counting 已排除)
# 消融基线 : self_distillation.state_adaptive=false → 标准特权 OPD (V0)
# =============================================================================
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
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
ALPHA=0.5                          # 广义 JSD (与 09_01_plan §9 一致)
# 全量 4k 数据实测: 学生 prompt(含视觉token)p99≈830, 最大≈4016(32条大图行), teacher侧最大≈900
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4608}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
TRAIN_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$TRAIN_MAX_MODEL_LEN}"
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}
ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-True}
ACTOR_PPO_MICRO_BS=${ACTOR_PPO_MICRO_BS:-null}    # dynamic_bsz=False 时必设 (Arm A 用 1)
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$TRAIN_MAX_MODEL_LEN}
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
ACTOR_PARAM_OFFLOAD=True
ACTIVATION_OFFLOAD=${ACTIVATION_OFFLOAD:-False}   # 高清数据(Arm A)显存救星: 激活下放 CPU
ULYSSES_SP=${ULYSSES_SP:-1}                       # 序列并行: 高清长序列按卡切分 (Arm A 用 2)
ACTOR_OPTIMIZER_OFFLOAD=True
REF_PARAM_OFFLOAD=True
TRAINER_N_GPUS_PER_NODE=${TRAINER_N_GPUS_PER_NODE:-8}
TRAINER_NNODES=${WORLD_SIZE:-1}
TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-5}
TRAINER_TOTAL_EPOCHS=1
TRAINER_MAX_ACTOR_CKPT_TO_KEEP=null
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","tensorboard","wandb"]'}   # wandb 默认开; 断网节点用 WANDB_MODE=offline
ROLLOUT_AGENT_NUM_WORKERS=8
DATA_DATALOADER_NUM_WORKERS=8
CUSTOM_CHAT_TEMPLATE_FILE="${PROJECT_ROOT}/chat_templates/perception_chat_template_qwen35.jinja"

# --- State-Adaptive knobs (09_01_plan §18; sharpening 已按附录 §C 移除) ---
SA_ENABLE=${SA_ENABLE:-True}
SA_LAMBDA=${SA_LAMBDA:-1.0}
SA_TAU_E=${SA_TAU_E:-0.03}
SA_TEMP_E=${SA_TEMP_E:-0.015}
SA_RHO=${SA_RHO:-0.9}
SA_EPS_MAX=${SA_EPS_MAX:-0.5}
SA_BETA_MAX=${SA_BETA_MAX:-8.0}
SA_BINARY_ITERS=8
# v2 修复 (u 分解探针 2026-09-03): 关门归零 + p^H 加权逐词去均值, 默认开
SA_ZERO_FLOOR=${SA_ZERO_FLOOR:-True}
SA_CENTER_U=${SA_CENTER_U:-True}
SA_U_CLIP=${SA_U_CLIP:-4.0}

# --- EVT (09_03_method_v2 §4B): EVT_ENABLE=True 时覆盖 SA, 走优势形式尾部更新 ---
EVT_ENABLE=${EVT_ENABLE:-False}
EVT_FRACTION=${EVT_FRACTION:-0.2}
EVT_CLIP_NEG=${EVT_CLIP_NEG:-2.0}
EVT_CLIP_POS=${EVT_CLIP_POS:-1.0}
EVT_EMA_ETA=${EVT_EMA_ETA:-0.05}
EVT_NEUTRAL_BASE=${EVT_NEUTRAL_BASE:-0.0}
EVT_UBAR_INIT=${EVT_UBAR_INIT:-${PROJECT_ROOT}/data/TreeVGR-RL-37K/evt_ubar_init.json}
EVT_NEGATIVE_ONLY=${EVT_NEGATIVE_ONLY:-False}
if [[ "$EVT_ENABLE" == "True" ]]; then
    SA_ENABLE=False   # 互斥
fi

# --- Data Paths ---
# 2026-09-04 起默认 hide 形态教师数据 (region 版已删除; 单张黑底原位合成图, 保画布几何)
TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-${PROJECT_ROOT}/data/TreeVGR-RL-37K/train_sa4k_hide.parquet}"

MODEL_NAME=$(basename "$MODEL_PATH")
if [[ "$EVT_ENABLE" == "True" ]]; then
    if [[ "$EVT_NEGATIVE_ONLY" == "True" ]]; then
        EXPERIMENT_NAME="EVTneg-${MODEL_NAME}"
    else
        EXPERIMENT_NAME="EVT-${MODEL_NAME}"
    fi
elif [[ "$SA_ENABLE" == "True" ]]; then
    if [[ "$SA_ZERO_FLOOR" == "True" || "$SA_CENTER_U" == "True" ]]; then
        EXPERIMENT_NAME="SA-OPDv2-${MODEL_NAME}"   # v2 修复版, 与 v1 checkpoint 目录分开
    else
        EXPERIMENT_NAME="SA-OPD-${MODEL_NAME}"
    fi
else
    EXPERIMENT_NAME="StdOPD-region-${MODEL_NAME}"
fi

# EVT 不需要 top-k 分布机器 (只用实际 token logp), 关闭以省显存和时间
FULL_LOGIT=True; TOPK=100
if [[ "$EVT_ENABLE" == "True" ]]; then
    FULL_LOGIT=False; TOPK=null
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

echo "Running: $EXPERIMENT_NAME  (state_adaptive=$SA_ENABLE)"
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
    actor_rollout_ref.actor.self_distillation.evt_enable=$EVT_ENABLE \
    actor_rollout_ref.actor.self_distillation.evt_token_fraction=$EVT_FRACTION \
    actor_rollout_ref.actor.self_distillation.evt_clip_neg=$EVT_CLIP_NEG \
    actor_rollout_ref.actor.self_distillation.evt_clip_pos=$EVT_CLIP_POS \
    actor_rollout_ref.actor.self_distillation.evt_ema_eta=$EVT_EMA_ETA \
    actor_rollout_ref.actor.self_distillation.evt_neutral_base=$EVT_NEUTRAL_BASE \
    actor_rollout_ref.actor.self_distillation.evt_ubar_init=$EVT_UBAR_INIT \
    actor_rollout_ref.actor.self_distillation.evt_negative_only=$EVT_NEGATIVE_ONLY \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
    actor_rollout_ref.actor.self_distillation.teacher_model_source=$TEACHER_MODEL_SOURCE \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=$TEACHER_REGULARIZATION \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE \
    actor_rollout_ref.actor.self_distillation.teacher_image_key=bbox_images \
    actor_rollout_ref.actor.self_distillation.state_adaptive=$SA_ENABLE \
    actor_rollout_ref.actor.self_distillation.sa_lambda=$SA_LAMBDA \
    actor_rollout_ref.actor.self_distillation.sa_tau_e=$SA_TAU_E \
    actor_rollout_ref.actor.self_distillation.sa_temp_e=$SA_TEMP_E \
    actor_rollout_ref.actor.self_distillation.sa_rho=$SA_RHO \
    actor_rollout_ref.actor.self_distillation.sa_eps_max=$SA_EPS_MAX \
    actor_rollout_ref.actor.self_distillation.sa_beta_max=$SA_BETA_MAX \
    actor_rollout_ref.actor.self_distillation.sa_binary_iters=$SA_BINARY_ITERS \
    actor_rollout_ref.actor.self_distillation.sa_zero_floor=$SA_ZERO_FLOOR \
    actor_rollout_ref.actor.self_distillation.sa_center_u=$SA_CENTER_U \
    actor_rollout_ref.actor.self_distillation.sa_u_clip=$SA_U_CLIP \
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
