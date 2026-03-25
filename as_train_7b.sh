MODEL_PATH=${2:-Qwen/Qwen2.5-7B}
GAMMA=${4:-0.0} 
ALPHA=${6:-0.0} 
BETA=${8:-1.0} 
LAMBDA=${10:-0.0} 

RW=${12:-em} # f1, em
ADV=${14:-grpo_prm_step}

MAX_TURN=${16:-5}
TOPK=${18:-3}
SEARCH_MODE=${20:-local}

TBS=${22:-64}
MBS=${24:-32}
LR=${26:-1e-6}
KL=${28:-0}
RO=${30:-5}
NUM_GPUS_PER_NODE=${32:-8}

WAND_PROJECT='MR-Search'
ROOT_DIR=/path/to/project
MODEL_NAME="${MODEL_PATH##*/}"

cd $ROOT_DIR

SAVE_DIR=$ROOT_DIR/save
DATA_DIR=$ROOT_DIR/data/asearcher

EXPERIMENT_NAME=${MODEL_NAME}_${RW}-gamma${GAMMA}-lambda${LAMBDA}-ALPHA${ALPHA}-beta${BETA}_${SEARCH_MODE}-k${TOPK}-t${MAX_TURN}_tbs${TBS}-mbs${MBS}-kl${KL}-lr${LR}
mkdir -p $SAVE_DIR/$EXPERIMENT_NAME/

export SEARCH_IP="http://0.0.0.0:8000/retrieve"

export VLLM_ATTENTION_BACKEND=XFORMERS
python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$TBS \
    data.val_batch_size=1024 \
    data.max_prompt_length=16384 \
    data.max_response_length=1024 \
    data.max_obs_length=1024 \
    algorithm.adv_estimator=$ADV \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MBS \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=$RO \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=false \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    reward_model.reward_manager=naive \
    reward_model.reward_manager_val=search \
    reward_model.reward_kwargs.reward_to_use=$RW \
    reward_model.reward_kwargs.use_gae=false \
    reward_model.reward_kwargs.improve_gamma=$GAMMA \
    reward_model.discount=1 \
    reward_model.reward_kwargs.penalty_lambda=$LAMBDA \
    reward_model.reward_kwargs.prm_alpha=$ALPHA \
    reward_model.reward_kwargs.prm_beta=$BETA \
    trainer.logger=['console','wandb'] \
    trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir=$SAVE_DIR/$EXPERIMENT_NAME \
    trainer.max_turns=$MAX_TURN \
    trainer.max_retry=3 \
    retriever.url=$SEARCH_IP \
    retriever.topk=$TOPK \
    retriever.mode=$SEARCH_MODE \
    2>&1 | tee $SAVE_DIR/$EXPERIMENT_NAME/log.txt

    