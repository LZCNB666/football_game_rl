import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from football_env_ppo import FootballEnv

# 创建日志和保存目录
log_dir = "ppo_football_logs/"
os.makedirs(log_dir, exist_ok=True)

# 创建训练环境与评估环境
train_env = Monitor(FootballEnv())
eval_env = Monitor(FootballEnv())

# 创建评估回调：每1万步评估一次，保存最优模型
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=10000,
    deterministic=True,
    render=False
)

# 初始化 PPO 模型
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    verbose=1,
    tensorboard_log=os.path.join(log_dir, "tensorboard"),
    learning_rate=1e-4,
    n_steps=2048,  # 每个更新周期的步数，越大越稳定
    batch_size=128,
    n_epochs=10,  # 每轮训练周期的更新次数
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,  # PPO 特有的截断范围
    ent_coef=0.005  # 熵奖励，鼓励探索
)

# 开始训练
model.learn(total_timesteps=500000, callback=eval_callback)

# 保存最终模型（可选）
model.save(os.path.join(log_dir, "ppo_football_final"))

print("PPO训练完成，模型保存在：", os.path.join(log_dir, "ppo_football_final.zip"))

# tensorboard --logdir ppo_football_logs/tensorboard
