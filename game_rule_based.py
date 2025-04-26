import pygame
import sys
import random
import math

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
ENEMY_SPEED = 3

# 颜色
GREEN = (0, 128, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# 创建游戏窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Football Game")
clock = pygame.time.Clock()

# 游戏对象
class Player:
    def __init__(self, x, y, color):
        self.rect = pygame.Rect(x, y, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.color = color
        self.speed = PLAYER_SPEED

    def move(self, dx, dy):
        # 确保玩家不会移出边界
        new_rect = self.rect.move(dx * self.speed, dy * self.speed)

        # 限制在本方半场内
        if self.color == BLUE:  # 玩家在左半场
            if new_rect.left < 0:
                new_rect.left = 0
            if new_rect.right > WIDTH // 2:
                new_rect.right = WIDTH // 2
        else:  # 敌人在右半场
            if new_rect.left < WIDTH // 2:
                new_rect.left = WIDTH // 2
            if new_rect.right > WIDTH:
                new_rect.right = WIDTH

        # 限制在上下边界内
        if new_rect.top < 0:
            new_rect.top = 0
        if new_rect.bottom > HEIGHT:
            new_rect.bottom = HEIGHT

        self.rect = new_rect

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)

    def kick(self, ball):
        # 检查是否接触到球
        if self.rect.colliderect(pygame.Rect(ball.x - BALL_RADIUS, ball.y - BALL_RADIUS,
                                             BALL_RADIUS * 2, BALL_RADIUS * 2)):
            # 计算踢球方向
            dx = ball.x - self.rect.centerx
            dy = ball.y - self.rect.centery
            distance = max(1.0, math.sqrt(dx * dx + dy * dy))

            # 标准化方向向量并应用踢球力量
            ball.vx += (dx / distance) * KICK_FORCE
            ball.vy += (dy / distance) * KICK_FORCE

            # 限制球速
            speed = math.sqrt(ball.vx ** 2 + ball.vy ** 2)
            if speed > MAX_BALL_SPEED:
                ball.vx = (ball.vx / speed) * MAX_BALL_SPEED
                ball.vy = (ball.vy / speed) * MAX_BALL_SPEED

class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        # 球在中间区域随机位置出现
        self.x = random.randint(WIDTH // 4, 3 * WIDTH // 4)
        self.y = random.randint(HEIGHT // 4, 3 * HEIGHT // 4)

        # 随机初始速度
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(MIN_BALL_SPEED, MIN_BALL_SPEED + 2)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        # 更新位置
        self.x += self.vx
        self.y += self.vy

        # 边界碰撞检测和反弹
        if self.x - BALL_RADIUS < 0 or self.x + BALL_RADIUS > WIDTH:
            self.vx = -self.vx * 0.8  # 反弹并减速
            # 确保球不会卡在边界
            if self.x - BALL_RADIUS < 0:
                self.x = BALL_RADIUS
            else:
                self.x = WIDTH - BALL_RADIUS

        if self.y - BALL_RADIUS < 0 or self.y + BALL_RADIUS > HEIGHT:
            self.vy = -self.vy * 0.8  # 反弹并减速
            # 确保球不会卡在边界
            if self.y - BALL_RADIUS < 0:
                self.y = BALL_RADIUS
            else:
                self.y = HEIGHT - BALL_RADIUS

        # 应用摩擦力
        self.vx *= FRICTION
        self.vy *= FRICTION

        # 确保最小速度
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if 0 < speed < MIN_BALL_SPEED:
            self.vx = (self.vx / speed) * MIN_BALL_SPEED
            self.vy = (self.vy / speed) * MIN_BALL_SPEED

    def draw(self):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), BALL_RADIUS)

    def check_goals(self):
        # 检查左球门（玩家球门）
        if self.x - BALL_RADIUS < GOAL_WIDTH and HEIGHT // 2 - GOAL_HEIGHT // 2 < self.y < HEIGHT // 2 + GOAL_HEIGHT // 2:
            return "enemy"
        # 检查右球门（敌人球门）
        if self.x + BALL_RADIUS > WIDTH - GOAL_WIDTH and HEIGHT // 2 - GOAL_HEIGHT // 2 < self.y < HEIGHT // 2 + GOAL_HEIGHT // 2:
            return "player"
        return None

class EnemyAI:
    def __init__(self, enemy):
        self.enemy = enemy

    def update(self, ball):
        # 简单AI：向球移动
        dx = ball.x - self.enemy.rect.centerx
        dy = ball.y - self.enemy.rect.centery

        # 标准化方向向量
        distance = max(1.0, math.sqrt(dx * dx + dy * dy))
        dx = dx / distance
        dy = dy / distance

        # 移动敌人
        self.enemy.move(dx, dy)

        # 随机改变方向（模拟撞墙行为）
        if random.random() < 0.02:  # 2%的几率改变方向
            self.enemy.move(random.uniform(-1, 1), random.uniform(-1, 1))


# 创建游戏对象
player = Player(WIDTH // 4, HEIGHT // 2, BLUE)
enemy = Player(3 * WIDTH // 4, HEIGHT // 2, RED)
enemy_ai = EnemyAI(enemy)
ball = Ball()

# 计分
player_score = 0
enemy_score = 0
font = pygame.font.Font(None, 36)

# 游戏主循环
running = True
while running:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 玩家输入
    keys = pygame.key.get_pressed()
    dx, dy = 0, 0
    if keys[pygame.K_LEFT]:
        dx = -1
    if keys[pygame.K_RIGHT]:
        dx = 1
    if keys[pygame.K_UP]:
        dy = -1
    if keys[pygame.K_DOWN]:
        dy = 1
    player.move(dx, dy)

    # 更新游戏状态
    ball.update()
    enemy_ai.update(ball)

    # 踢球检测
    player.kick(ball)
    enemy.kick(ball)

    # 检查进球
    goal = ball.check_goals()
    if goal:
        if goal == "player":
            player_score += 1
        else:
            enemy_score += 1
        ball.reset()

    # 绘制游戏
    screen.fill(GREEN)

    # 绘制中线
    pygame.draw.line(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)

    # 绘制球门
    pygame.draw.rect(screen, WHITE, (0, HEIGHT // 2 - GOAL_HEIGHT // 2, GOAL_WIDTH, GOAL_HEIGHT))
    pygame.draw.rect(screen, WHITE, (WIDTH - GOAL_WIDTH, HEIGHT // 2 - GOAL_HEIGHT // 2, GOAL_WIDTH, GOAL_HEIGHT))

    # 绘制游戏对象
    player.draw()
    enemy.draw()
    ball.draw()

    # 绘制分数
    player_text = font.render(f"YOU: {player_score}", True, BLUE)
    enemy_text = font.render(f"AI: {enemy_score}", True, RED)
    screen.blit(player_text, (20, 20))
    screen.blit(enemy_text, (WIDTH - 150, 20))

    # 更新显示
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()