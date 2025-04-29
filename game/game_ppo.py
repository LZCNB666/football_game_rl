import pygame
import sys
import random
import math
import numpy as np
from stable_baselines3 import PPO

# load the model
model = PPO.load("../rf/ppo_football_logs/best_model.zip")

# initialize pygame
pygame.init()

# global constants
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

# color
GREEN = (0, 128, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Football Game")
clock = pygame.time.Clock()

# game objects
class Player:
    def __init__(self, x, y, color):
        self.rect = pygame.Rect(x, y, PLAYER_WIDTH, PLAYER_HEIGHT)
        self.color = color
        self.speed = PLAYER_SPEED

    def move(self, dx, dy):
        # make sure player doesn't move outside the screen
        new_rect = self.rect.move(dx * self.speed, dy * self.speed)

        # limit player in its own half
        if self.color == BLUE:  # player on left half
            if new_rect.left < 0:
                new_rect.left = 0
            if new_rect.right > WIDTH // 2:
                new_rect.right = WIDTH // 2
        else:  # enemy on right half
            if new_rect.left < WIDTH // 2:
                new_rect.left = WIDTH // 2
            if new_rect.right > WIDTH:
                new_rect.right = WIDTH

        # limit the upper and lower boundaries
        if new_rect.top < 0:
            new_rect.top = 0
        if new_rect.bottom > HEIGHT:
            new_rect.bottom = HEIGHT

        self.rect = new_rect

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)

    def kick(self, ball):
        # check if player touches ball
        if self.rect.colliderect(pygame.Rect(ball.x - BALL_RADIUS, ball.y - BALL_RADIUS,
                                             BALL_RADIUS * 2, BALL_RADIUS * 2)):
            # calculate kicking direction
            dx = ball.x - self.rect.centerx
            dy = ball.y - self.rect.centery
            distance = max(1.0, math.sqrt(dx * dx + dy * dy))

            # apply kick force
            ball.vx += (dx / distance) * KICK_FORCE
            ball.vy += (dy / distance) * KICK_FORCE

            # limit ball speed
            speed = math.sqrt(ball.vx ** 2 + ball.vy ** 2)
            if speed > MAX_BALL_SPEED:
                ball.vx = (ball.vx / speed) * MAX_BALL_SPEED
                ball.vy = (ball.vy / speed) * MAX_BALL_SPEED

class Ball:
    def __init__(self):
        self.reset()

    def reset(self):
        # the ball appears at random in the middle area
        self.x = random.randint(WIDTH // 4, 3 * WIDTH // 4)
        self.y = random.randint(HEIGHT // 4, 3 * HEIGHT // 4)

        # random initial velocity
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(MIN_BALL_SPEED, MIN_BALL_SPEED + 2)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        # update ball position
        self.x += self.vx
        self.y += self.vy

        # boundary collision detection and bounce
        if self.x - BALL_RADIUS < 0 or self.x + BALL_RADIUS > WIDTH:
            self.vx = -self.vx * 0.8  # bounce and slow down
            # make sure the ball doesn't get stuck on the boundary
            if self.x - BALL_RADIUS < 0:
                self.x = BALL_RADIUS
            else:
                self.x = WIDTH - BALL_RADIUS

        if self.y - BALL_RADIUS < 0 or self.y + BALL_RADIUS > HEIGHT:
            self.vy = -self.vy * 0.8  # bounce and slow down
            # make sure the ball doesn't get stuck on the boundary
            if self.y - BALL_RADIUS < 0:
                self.y = BALL_RADIUS
            else:
                self.y = HEIGHT - BALL_RADIUS

        # friction
        self.vx *= FRICTION
        self.vy *= FRICTION

        # make sure the minimum speed
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if 0 < speed < MIN_BALL_SPEED:
            self.vx = (self.vx / speed) * MIN_BALL_SPEED
            self.vy = (self.vy / speed) * MIN_BALL_SPEED

    def draw(self):
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), BALL_RADIUS)

    def check_goals(self):
        # check left goal (player's goal)
        if self.x - BALL_RADIUS < GOAL_WIDTH and HEIGHT // 2 - GOAL_HEIGHT // 2 < self.y < HEIGHT // 2 + GOAL_HEIGHT // 2:
            return "enemy"
        # check right goal (enemy's goal)
        if self.x + BALL_RADIUS > WIDTH - GOAL_WIDTH and HEIGHT // 2 - GOAL_HEIGHT // 2 < self.y < HEIGHT // 2 + GOAL_HEIGHT // 2:
            return "player"
        return None

class EnemyAI:
    def __init__(self, enemy, model=model):
        self.enemy = enemy
        self.model = model

    def get_state(self, ball, player):
        return np.array([
            # player position
            player.rect.centerx / WIDTH,
            player.rect.centery / HEIGHT,

            # enemy position
            enemy.rect.centerx / WIDTH,
            enemy.rect.centery / HEIGHT,

            # ball state
            ball.x / WIDTH,
            ball.y / HEIGHT,
            ball.vx / MAX_BALL_SPEED,
            ball.vy / MAX_BALL_SPEED,

            # relative position of ball to player and enemy
            (ball.x - enemy.rect.centerx) / WIDTH,
            (ball.y - enemy.rect.centery) / HEIGHT,
            (ball.x - player.rect.centerx) / WIDTH,
            (ball.y - player.rect.centery) / HEIGHT,
            ], dtype=np.float32)

    def move_to_ball(self, ball):
        # move to ball
        dx = ball.x - self.enemy.rect.centerx
        dy = ball.y - self.enemy.rect.centery
        # moving direction
        distance = max(1.0, math.sqrt(dx * dx + dy * dy))
        dx = dx / distance
        dy = dy / distance
        # move
        self.enemy.move(dx, dy)

    def update(self, ball, player):
        state = self.get_state(ball, player)
        action, _ = self.model.predict(state, deterministic=True)
        # enemy actions
        if action in [0, 1, 2, 3]:
            dx, dy = 0, 0
            if action == 0: dy = -1
            elif action == 1: dy = 1
            elif action == 2: dx = -1
            elif action == 3: dx = 1
            self.enemy.move(dx, dy)
        elif action == 4:
            self.move_to_ball(ball)

        # randomly change direction (simulate wall impact behavior)
        if random.random() < 0.02:  # 2% chance to change direction
            self.enemy.move(random.uniform(-1, 1), random.uniform(-1, 1))


# create game objects
player = Player(WIDTH // 4, HEIGHT // 2, BLUE)
enemy = Player(3 * WIDTH // 4, HEIGHT // 2, RED)
enemy_ai = EnemyAI(enemy)
ball = Ball()

# count scores
player_score = 0
enemy_score = 0
font = pygame.font.Font(None, 36)

# game main loop
running = True
while running:
    # get events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # player keyboard input
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

    # update game objects
    ball.update()
    enemy_ai.update(ball, player)

    # kicking detection
    player.kick(ball)
    enemy.kick(ball)

    # check goal
    goal = ball.check_goals()
    if goal:
        if goal == "player":
            player_score += 1
        else:
            enemy_score += 1
        ball.reset()

    # draw background
    screen.fill(GREEN)

    # draw center line
    pygame.draw.line(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)

    # draw goal
    pygame.draw.rect(screen, WHITE, (0, HEIGHT // 2 - GOAL_HEIGHT // 2, GOAL_WIDTH, GOAL_HEIGHT))
    pygame.draw.rect(screen, WHITE, (WIDTH - GOAL_WIDTH, HEIGHT // 2 - GOAL_HEIGHT // 2, GOAL_WIDTH, GOAL_HEIGHT))

    # draw game objects
    player.draw()
    enemy.draw()
    ball.draw()

    # draw scores
    player_text = font.render(f"YOU: {player_score}", True, BLUE)
    enemy_text = font.render(f"AI: {enemy_score}", True, RED)
    screen.blit(player_text, (20, 20))
    screen.blit(enemy_text, (WIDTH - 150, 20))

    # update game scene
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()