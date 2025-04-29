### Python Football Game Based on Reinforcement Learning
___
#### Project Structure
```text
football_game
|
├── football_env_ppo.py:  
|   training environment for PPO with gymnasium style
├── train_ppo.py: 
|   train PPO using stable-baselines3 
|
├── game_rule_based.py: 
|   AI is rule-based
├── game_ppo.py: 
|   AI using PPO to make decisions
├── game_hybrid.py: 
|   AI using a hybrid strategy (rule-based and PPO)
|
├── test_ruleBased_ppo.py:
|   Rule-based AI vs PPO AI
├── test_hybrid_ppo.py:
|   Hybrid AI vs PPO AI
```

#### Environment Settings
1. Python >= 3.9 is required
```bash
conda create -n football python=3.9.13
```
```bash
conda activate football
```
2. Install required packages
```bash
pip install numpy matplotlib pygame gymnasium
```
3. PyTorch >= 2.3 is required
```bash
pip install torch==2.5.0 torchvision==2.5.0
```
4. Install stable-baselines3
```bash
pip install stable-baselines3[extra]
```

#### Training

#### Run the Game

#### References