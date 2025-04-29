### 基于强化学习的Python足球游戏
___
#### 项目结构
```text
football_game
|
├── football_env_ppo.py:  
|   用于训练PPO的足球游戏环境，使用gymnasium风格，12维观测空间
├── football_env_ppo_8d.py:  
|   用于训练PPO的足球游戏环境，使用gymnasium风格，8维观测空间
├── train_ppo.py: 
|   使用stable-baselines3训练12维观测空间的PPO
├── train_ppo_8d.py: 
|   使用stable-baselines3训练8维观测空间的PPO
|
├── game_rule_based.py: 
|   AI的策略只基于规则
├── game_ppo.py: 
|   AI通过PPO来进行决策
├── game_hybrid.py: 
|   AI使用混合策略(基于规则和PPO)
|
├── test_ruleBased_ppo.py:
|   基于规则的AI vs PPO AI
├── test_hybrid_ppo.py:
|   混合型AI vs PPO AI
```

#### 环境配置
1. Python版本需要 >= 3.9
```bash
conda create -n football python=3.9.13
```
```bash
conda activate football
```
2. 安装必要的依赖包
```bash
pip install numpy matplotlib pygame gymnasium
```
3. PyTorch版本需要 >= 2.3
```bash
pip install torch==2.5.0 torchvision==2.5.0
```
4. 安装stable-baselines3
```bash
pip install stable-baselines3[extra]
```

#### 训练

#### 运行游戏

#### 参考