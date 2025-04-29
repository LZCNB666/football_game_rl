import pygame
import sys
import random
import math
import numpy as np
from stable_baselines3 import PPO

# 加载 PPO 模型
model = PPO.load("./ppo_football_logs/best_model.zip")

# 初始化pygame
pygame.init()

# 游戏常量
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
ENEMY_SPEED = 5
TARGET_MATCH_COUNT = 10

# 颜色
GREEN = (0, 128, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# 创建游戏窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Football Game Auto Play")
clock = pygame.time.Clock()

# 游戏对象
class Player:
    def __init__(self, x, y, color, speed):
        self.rect = pygame.Rect(x, y, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.color = color
        self.speed = speed

    def move(self, dx, dy):
        new_rect = self.rect.move(dx * self.speed, dy * self.speed)

        if self.color == BLUE:
            if new_rect.left < 0:
                new_rect.left = 0
            if new_rect.right > WIDTH // 2:
                new_rect.right = WIDTH // 2
        else:
            if new_rect.left < WIDTH // 2:
                new_rect.left = WIDTH // 2
            if new_rect.right > WIDTH:
                new_rect.right = WIDTH

        if new_rect.top < 0:
            new_rect.top = 0
        if new_rect.bottom > HEIGHT:
            new_rect.bottom = HEIGHT

        self.rect = new_rect

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)

    def kick(self, ball):
        if self.rect.colliderect(pygame.Rect(ball.x - BALL_RADIUS, ball.y - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)):
            dx = ball.x - self.rect.centerx
            dy = ball.y - self.rect.centery
            distance = max(1.0, math.sqrt(dx * dx + dy * dy))
            ball.vx += (dx / distance) * KICK_FORCE
            ball.vy += (dy / distance) * KICK_FORCE

            speed = math.sqrt(ball.vx ** 2 + ball.vy ** 2)
            if speed > MAX_BALL_SPEED:
                ball.vx = (ball.vx / speed) * MAX_BALL_SPEED
                ball.vy = (ball.vy / speed) * MAX_BALL_SPEED

            return True  # 踢到球了
        return False  # 没踢到球

class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = random.randint(WIDTH // 4, 3 * WIDTH // 4)
        self.y = random.randint(HEIGHT // 4, 3 * HEIGHT // 4)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(MIN_BALL_SPEED, MIN_BALL_SPEED + 2)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.x += self.vx
        self.y += self.vy

        if self.x - BALL_RADIUS < 0 or self.x + BALL_RADIUS > WIDTH:
            self.vx = -self.vx * 0.8
            self.x = max(BALL_RADIUS, min(self.x, WIDTH - BALL_RADIUS))

        if self.y - BALL_RADIUS < 0 or self.y + BALL_RADIUS > HEIGHT:
            self.vy = -self.vy * 0.8
            self.y = max(BALL_RADIUS, min(self.y, HEIGHT - BALL_RADIUS))

        self.vx *= FRICTION
        self.vy *= FRICTION

        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if 0 < speed < MIN_BALL_SPEED:
            self.vx = (self.vx / speed) * MIN_BALL_SPEED
            self.vy = (self.vy / speed) * MIN_BALL_SPEED

    def draw(self):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), BALL_RADIUS)

    def check_goals(self):
        if self.x - BALL_RADIUS < GOAL_WIDTH and HEIGHT // 2 - GOAL_HEIGHT // 2 < self.y < HEIGHT // 2 + GOAL_HEIGHT // 2:
            return "enemy"
        if self.x + BALL_RADIUS > WIDTH - GOAL_WIDTH and HEIGHT // 2 - GOAL_HEIGHT // 2 < self.y < HEIGHT // 2 + GOAL_HEIGHT // 2:
            return "player"
        return None

class PPOAgent:
    def __init__(self, player):
        self.player = player

    def get_state(self, ball, enemy):
        return np.array([
            self.player.rect.centerx / WIDTH,
            self.player.rect.centery / HEIGHT,
            enemy.rect.centerx / WIDTH,
            enemy.rect.centery / HEIGHT,
            ball.x / WIDTH,
            ball.y / HEIGHT,
            ball.vx / MAX_BALL_SPEED,
            ball.vy / MAX_BALL_SPEED,
            (ball.x - self.player.rect.centerx) / WIDTH,
            (ball.y - self.player.rect.centery) / HEIGHT,
            (ball.x - enemy.rect.centerx) / WIDTH,
            (ball.y - enemy.rect.centery) / HEIGHT,
            ], dtype=np.float32)

    def update(self, ball, enemy):
        state = self.get_state(ball, enemy)
        action, _ = model.predict(state, deterministic=True)
        dx, dy = 0, 0
        if action == 0:
            dy = -1
        elif action == 1:
            dy = 1
        elif action == 2:
            dx = -1
        elif action == 3:
            dx = 1
        elif action == 4:
            dx = (ball.x - self.player.rect.centerx)
            dy = (ball.y - self.player.rect.centery)
            dist = max(1.0, math.sqrt(dx * dx + dy * dy))
            dx /= dist
            dy /= dist

        self.player.move(dx, dy)

class RuleBasedEnemy:
    def __init__(self, player):
        self.player = player

    def update(self, ball):
        dx = ball.x - self.player.rect.centerx
        dy = ball.y - self.player.rect.centery
        dist = max(1.0, math.sqrt(dx * dx + dy * dy))
        dx /= dist
        dy /= dist
        self.player.move(dx, dy)


# 创建对象
player = Player(WIDTH // 4, HEIGHT // 2, BLUE, PLAYER_SPEED)
enemy = Player(3 * WIDTH // 4, HEIGHT // 2, RED, ENEMY_SPEED)
ball = Ball()

ppo_agent = PPOAgent(player)
rule_based_enemy = RuleBasedEnemy(enemy)

# 分数记录
player_score = 0
enemy_score = 0
matches_played = 0

# 统计数据
ppo_hold_time = 0
rule_based_hold_time = 0
ppo_attack_count = 0
rule_based_attack_count = 0
ppo_defense_count = 0
rule_based_defense_count = 0

# 持球归属
ball_holder = None  # "ppo" / "rule"

font = pygame.font.Font(None, 36)

# 自动对战循环
running = True
while running and matches_played < TARGET_MATCH_COUNT:
    clock.tick(60)

    # 更新
    ball.update()
    ppo_agent.update(ball, enemy)
    rule_based_enemy.update(ball)

    # 踢球检测
    player_kicked = player.kick(ball)
    enemy_kicked = enemy.kick(ball)

    # 判断当前持球方
    if player_kicked:
        ball_holder = "ppo"
        if abs(ball.x - WIDTH // 2) < 100:  # 距离中线 < 100px时击球，算作进攻行为
            ppo_attack_count += 1
        if ball.x < WIDTH - 100:  # 距离己方球门 < 100px时击球，算作防守行为
            ppo_defense_count += 1
    elif enemy_kicked:
        ball_holder = "rule"
        if abs(ball.x - WIDTH // 2) < 100:
            rule_based_attack_count += 1
        if ball.x > WIDTH - 100:
            rule_based_defense_count += 1

    # 持球时间累计
    if ball_holder == "ppo":
        ppo_hold_time += 1
    elif ball_holder == "rule":
        rule_based_hold_time += 1

    # 检查进球
    goal = ball.check_goals()
    if goal:
        if goal == "player":
            player_score += 1
        else:
            enemy_score += 1
        matches_played += 1
        ball.reset()
        player.rect.center = (WIDTH // 4, HEIGHT // 2)
        enemy.rect.center = (3 * WIDTH // 4, HEIGHT // 2)
        ball_holder = None

    # 绘制
    screen.fill(GREEN)
    pygame.draw.line(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)
    pygame.draw.rect(screen, WHITE, (0, HEIGHT // 2 - GOAL_HEIGHT // 2, GOAL_WIDTH, GOAL_HEIGHT))
    pygame.draw.rect(screen, WHITE, (WIDTH - GOAL_WIDTH, HEIGHT // 2 - GOAL_HEIGHT // 2, GOAL_WIDTH, GOAL_HEIGHT))

    player.draw()
    enemy.draw()
    ball.draw()

    player_text = font.render(f"PPO: {player_score}", True, WHITE)
    enemy_text = font.render(f"Rule-Based: {enemy_score}", True, WHITE)
    match_text = font.render(f"Match {matches_played}/{TARGET_MATCH_COUNT}", True, WHITE)

    screen.blit(player_text, (20, 20))
    screen.blit(enemy_text, (WIDTH - 200, 20))
    screen.blit(match_text, (WIDTH // 2 - 80, 20))

    pygame.display.flip()

# 打印最终结果
print("========== Battle Result ==========")
print(f"PPO Agent Wins: {player_score}")
print(f"Rule-Based Enemy Wins: {enemy_score}")
print("\n========== Statistics ==========")
print(f"PPO Agent Hold Time (frames): {ppo_hold_time}")
print(f"Rule-Based Enemy Hold Time (frames): {rule_based_hold_time}")
print(f"PPO attack: {ppo_attack_count}")
print(f"Rule-Based attack: {rule_based_attack_count}")
print(f"PPO defense: {ppo_defense_count}")
print(f"Rule-Based defense: {rule_based_defense_count}")

pygame.quit()
sys.exit()
