import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from football_env_ppo_8d import FootballEnv

# create log and save directory
log_dir = "ppo_football_logs2/"
os.makedirs(log_dir, exist_ok=True)

# creating training and evaluation environments
train_env = Monitor(FootballEnv())
eval_env = Monitor(FootballEnv())

# create evaluation callback: evaluate every 10000 steps and save the optimal model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=10000,
    deterministic=True,
    render=False
)

# initialize PPO model
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    verbose=1,
    tensorboard_log=os.path.join(log_dir, "tensorboard"),
    learning_rate=1e-4,
    n_steps=2048,  # steps per update cycle
    batch_size=128,
    n_epochs=10,  # number of updates per training cycle
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,  # PPO specific clip range
    ent_coef=0.005  # entropy reward, encouraging exploration
)

# start training
model.learn(total_timesteps=500000, callback=eval_callback)

# save best model
model.save(os.path.join(log_dir, "ppo_football_final"))

print("Training complete. Best model saved at: ", os.path.join(log_dir, "ppo_football_final.zip"))

# tensorboard --logdir ppo_football_logs/tensorboard
