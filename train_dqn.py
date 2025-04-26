import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from football_env_ppo import FootballEnv

# 输出路径
log_dir = "dqn_football_logs/"
os.makedirs(log_dir, exist_ok=True)

# 创建向量化环境
env = make_vec_env(FootballEnv, n_envs=4)

# 创建评估环境
eval_env = Monitor(FootballEnv())

# 创建评估回调：每1万步评估一次，保留最优模型
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=10000,
    deterministic=True,
    render=False
)

# 创建并训练模型
model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log=os.path.join(log_dir, "tensorboard"),
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=64,
    tau=0.01,  # 目标网络软更新
    target_update_interval=500,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    gamma=0.99,
)

model.learn(total_timesteps=100000, callback=eval_callback)

# 保存最终模型（可选）
model.save(os.path.join(log_dir, "dqn_football_final"))

print("DQN训练完成，最优模型保存在：", os.path.join(log_dir, "best_model.zip"))

# tensorboard --logdir ./dqn_football_logs/tensorboard
