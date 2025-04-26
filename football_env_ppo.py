import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import random
import pygame

# 全局常量
WIDTH, HEIGHT = 800, 600
PLAYER_WIDTH, PLAYER_HEIGHT = 30, 50
ENEMY_WIDTH, ENEMY_HEIGHT = 30, 50
BALL_RADIUS = 15
GOAL_WIDTH, GOAL_HEIGHT = 20, 150
MAX_BALL_SPEED = 15
MIN_BALL_SPEED = 2
FRICTION = 0.99
KICK_FORCE = 5
PLAYER_SPEED = 5
ENEMY_SPEED = 3


class Player:
    def __init__(self, x, y, is_enemy=False):
        self.rect = pygame.Rect(x, y, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.is_enemy = is_enemy

    def move(self, dx, dy):
        new_rect = self.rect.move(dx * (ENEMY_SPEED if self.is_enemy else PLAYER_SPEED),
                                  dy * (ENEMY_SPEED if self.is_enemy else PLAYER_SPEED))

        # 移动边界限制
        if self.is_enemy:
            new_rect.left = max(WIDTH//2, new_rect.left)
            new_rect.right = min(WIDTH, new_rect.right)
        else:
            new_rect.left = max(0, new_rect.left)
            new_rect.right = min(WIDTH//2, new_rect.right)

        new_rect.top = max(0, new_rect.top)
        new_rect.bottom = min(HEIGHT, new_rect.bottom)
        self.rect = new_rect

class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = random.randint(WIDTH//4, 3*WIDTH//4)
        self.y = random.randint(HEIGHT//4, 3*HEIGHT//4)

        angle = random.uniform(0, 2*math.pi)
        speed = random.uniform(MIN_BALL_SPEED, MIN_BALL_SPEED+2)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.x += self.vx
        self.y += self.vy

        # 边界碰撞
        if self.x - BALL_RADIUS < 0 or self.x + BALL_RADIUS > WIDTH:
            self.vx *= -0.8
            self.x = np.clip(self.x, BALL_RADIUS, WIDTH-BALL_RADIUS)
        if self.y - BALL_RADIUS < 0 or self.y + BALL_RADIUS > HEIGHT:
            self.vy *= -0.8
            self.y = np.clip(self.y, BALL_RADIUS, HEIGHT-BALL_RADIUS)

        # 摩擦处理
        self.vx *= FRICTION
        self.vy *= FRICTION

        # 最小速度维持
        speed = math.hypot(self.vx, self.vy)
        if 0 < speed < MIN_BALL_SPEED:
            self.vx = (self.vx/speed) * MIN_BALL_SPEED
            self.vy = (self.vy/speed) * MIN_BALL_SPEED

class FootballEnv(gym.Env):
    def __init__(self, max_steps=3000):
        super().__init__()

        self.MAX_STEP = max_steps
        self.current_step = 0

        # 离散动作空间（上、下、左、右、踢球）
        self.action_space = spaces.Discrete(5)

        # 观察空间：12维归一化向量
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # 初始化游戏对象
        self.player = Player(WIDTH//4, HEIGHT//2)
        self.enemy = Player(3 * WIDTH//4, HEIGHT//2, is_enemy=True)
        self.ball = Ball()

    def _get_obs(self):
        # 精确的状态观测表示
        return np.array([
            # 玩家位置（归一化）
            self.player.rect.centerx / WIDTH,
            self.player.rect.centery / HEIGHT,

            # 敌人位置
            self.enemy.rect.centerx / WIDTH,
            self.enemy.rect.centery / HEIGHT,

            # 球的状态
            self.ball.x / WIDTH,
            self.ball.y / HEIGHT,
            self.ball.vx / MAX_BALL_SPEED,
            self.ball.vy / MAX_BALL_SPEED,

            # 相对位置
            (self.ball.x - self.enemy.rect.centerx) / WIDTH,
            (self.ball.y - self.enemy.rect.centery) / HEIGHT,
            (self.ball.x - self.player.rect.centerx) / WIDTH,
            (self.ball.y - self.player.rect.centery) / HEIGHT,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # 重置对象位置
        self.player.rect.center = (WIDTH//4, HEIGHT//2)
        self.enemy.rect.center = (3*WIDTH//4, HEIGHT//2)
        self.ball.reset()

        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1

        # 处理敌人动作
        if action in [0, 1, 2, 3]:
            dx, dy = 0, 0
            if action == 0: dy = -1
            elif action == 1: dy = 1
            elif action == 2: dx = -1
            elif action == 3: dx = 1
            self.enemy.move(dx, dy)
        elif action == 4:
            self._enemy_kick()

        # 玩家自动移动
        self._update_player()

        # 更新球状态
        self.ball.update()

        # 检查进球
        goal_result = self._check_goal()

        # 计算奖励和终止条件
        reward = self._calculate_reward(goal_result)

        # 终止条件
        terminated = goal_result is not None
        truncated = self.current_step >= self.MAX_STEP

        return self._get_obs(), reward, terminated, truncated, {}

    def _enemy_kick(self):
        if self.enemy.rect.colliderect(pygame.Rect(
                self.ball.x - BALL_RADIUS, self.ball.y - BALL_RADIUS,
                2*BALL_RADIUS, 2*BALL_RADIUS)):

            dx = self.ball.x - self.enemy.rect.centerx
            dy = self.ball.y - self.enemy.rect.centery
            dist = max(1.0, math.hypot(dx, dy))

            self.ball.vx += (dx / dist) * KICK_FORCE
            self.ball.vy += (dy / dist) * KICK_FORCE

            # 速度限制
            speed = math.hypot(self.ball.vx, self.ball.vy)
            if speed > MAX_BALL_SPEED:
                self.ball.vx = (self.ball.vx / speed) * MAX_BALL_SPEED
                self.ball.vy = (self.ball.vy / speed) * MAX_BALL_SPEED

    def _update_player(self):
        # 玩家简易追球逻辑
        if random.random() < 0.1:
            dx, dy = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
        else:
            dx = np.sign(self.ball.x - self.player.rect.centerx)
            dy = np.sign(self.ball.y - self.player.rect.centery)
        self.player.move(dx, dy)

    def _calculate_reward(self, goal_result=None):
        reward = 0.0
        enemy_to_ball = math.hypot(
            self.ball.x - self.enemy.rect.centerx,
            self.ball.y - self.enemy.rect.centery
        )

        # 进攻奖励（球向左移动）
        if self.ball.vx < 0:
            reward += 0.2 * abs(self.ball.vx / MAX_BALL_SPEED)
        # 防守惩罚（球向右移动）
        else:
            reward -= 0.15 * (self.ball.vx / MAX_BALL_SPEED)

        # 与球的距离奖励（指数衰减）
        reward += 0.3 * math.exp(-enemy_to_ball / 200)

        # 当球在左半场（攻击有利区域）
        if self.ball.x < WIDTH/2:
            reward += 0.1 * (self.ball.x - WIDTH/2)/(WIDTH/2)
        else:
            reward -= 0.1 * (self.ball.x - WIDTH/2)/(WIDTH/2)

        # 进球奖励
        if goal_result:
            if goal_result == "player":  # 玩家得分
                reward -= 2.0
            else:  # 敌人得分
                reward += 3.0

        return reward

    def _check_goal(self):
        # 检查左球门（玩家球门）
        if self.ball.x - BALL_RADIUS < GOAL_WIDTH and HEIGHT // 2 - GOAL_HEIGHT // 2 < self.ball.y < HEIGHT // 2 + GOAL_HEIGHT // 2:
            return "enemy"
        # 检查右球门（敌人球门）
        if self.ball.x + BALL_RADIUS > WIDTH - GOAL_WIDTH and HEIGHT // 2 - GOAL_HEIGHT // 2 < self.ball.y < HEIGHT // 2 + GOAL_HEIGHT // 2:
            return "player"
        return None
