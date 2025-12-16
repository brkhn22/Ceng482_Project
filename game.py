import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Don't initialize pygame globally - let each process do it when needed
# pygame.init() will be called only when rendering

# Constants
BLOCK_SIZE = 40
SPEED = 40

# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')


class SnakeGameAI:
    
    def __init__(self, w=640, h=480, render_mode=True, verbose=False, display_speed_multiplier=1):
        self.w = w
        self.h = h
        self.render_mode = render_mode
        self.verbose = verbose  # Show death messages only when verbose is True
        self.display_speed_multiplier = display_speed_multiplier  # Multiplier for visual display speed only
        
        # Initialize display only if rendering is enabled
        if self.render_mode:
            pygame.init()  # Initialize pygame only when rendering
            self.font = pygame.font.Font(None, 25)
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI')
            self.clock = pygame.time.Clock()
        else:
            self.font = None
            self.display = None
            self.clock = None
            
        self.reset()
        
    def reset(self):
        # Initialize game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        # Prevent infinite recursion when board is almost full
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                return
        
        # If we can't find a spot after 100 tries, find any empty spot
        for x in range(0, self.w, BLOCK_SIZE):
            for y in range(0, self.h, BLOCK_SIZE):
                pt = Point(x, y)
                if pt not in self.snake:
                    self.food = pt
                    return
        
        # Board is completely full - game won (extremely rare)
        # Just place food on head, game will end naturally
        self.food = self.snake[0]
            
    def play_step(self, action):
        self.frame_iteration += 1
        
        # 1. Collect user input (only when rendering)
        if self.render_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # 2. Move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. Check if game over
        reward = 0
        game_over = False
        
        # Check collision
        collision_type = self.get_collision_type()
        if collision_type:
            game_over = True
            reward = -10
            if self.verbose:
                if collision_type == 'wall':
                    print("💀 Died: COLLISION - Hit Wall")
                elif collision_type == 'self':
                    print("💀 Died: COLLISION - Hit Self")
            return reward, game_over, self.score
        
        # Check starvation timeout - snake must eat within reasonable time
        starvation_limit = 100 * len(self.snake)
        if self.frame_iteration > starvation_limit:
            if self.verbose:
                print(f"⌛ Died: STARVATION - No food for {self.frame_iteration} frames (limit: {starvation_limit})")
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.frame_iteration = 0  # Reset counter when food is eaten
        else:
            self.snake.pop()
        
        # 5. Update ui and clock (only if rendering is enabled)
        if self.render_mode:
            self._update_ui()
            self.clock.tick(SPEED * self.display_speed_multiplier)
        
        # 6. Return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, point=None):
        if point is None:
            point = self.head
        # Hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # Hits itself
        if point in self.snake[1:]:
            return True
        
        return False
    
    def get_collision_type(self, point=None):
        """Return the type of collision: 'wall' or 'self' or None"""
        if point is None:
            point = self.head
        # Hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return 'wall'
        # Hits itself
        if point in self.snake[1:]:
            return 'self'
        
        return None
    
    def _update_ui(self):
        if not self.render_mode or self.display is None:
            return
            
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
    def _move(self, action):
        # [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d
        
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        
        self.head = Point(x, y)

