import random
import numpy as np
import gymnasium as gym

def frontier(grid, x, y):
    f = set()
    height, width = grid.shape
    if x >= 0 and x < width and y >= 0 and y < height:
        if x > 1 and not grid[y, x-2]:
            f.add((x-2, y))
        if x + 2 < width and not grid[y, x+2]:
            f.add((x+2, y))
        if y > 1 and not grid[y-2, x]:
            f.add((x, y-2))
        if y + 2 < height and not grid[y+2, x]:
            f.add((x, y+2))

    return f

def neighbors(grid, x, y):
    n = set()
    height, width = grid.shape
    if x >= 0 and x < width and y >= 0 and y < height:
        if x > 1 and grid[y, x-2]:
            n.add((x-2, y))
        if x + 2 < width and grid[y, x+2]:
            n.add((x+2, y))
        if y > 1 and grid[y-2, x]:
            n.add((x, y-2))
        if y + 2 < height and grid[y+2, x]:
            n.add((x, y+2))

    return n

def generate_maze(width, height):
    s = set()
    grid = np.zeros((height, width), dtype=bool)
    x, y = (random.randint(0, width - 1), random.randint(0, height - 1))
    grid[y, x] = True
    fs = frontier(grid, x, y)
    for f in fs:
        s.add(f)
    while s:
        x, y = random.choice(tuple(s))
        s.remove((x, y))
        ns = neighbors(grid, x, y)
        if ns:
            nx, ny = random.choice(tuple(ns))
            grid[y, x] = True
            grid[(y + ny) // 2, (x + nx) // 2] = True

        fs = frontier(grid, x, y)
        for f in fs:
            s.add(f)

    return grid


class MazeEnv(gym.Env):
    def __init__(self, width=99, height=99):
        self.width = width
        self.height = height
        self.reset()

    def is_valid_move(self, y, x):
        return 0 <= x < self.width and 0 <= y < self.height and self.maze[y, x]

    def get_random_positions(self):
        while True:
            yield random.randint(0, self.height - 1), random.randint(0, self.width - 1)

    def obs(self):
        obs = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        obs[:] = self.maze[:, :, None] * 255
        obs[self.current_position] = (255, 0, 0)
        obs[self.goal_position] = (0, 0, 255)
        return obs

    def info(self):
        return {'position': self.current_position, 'goal': self.goal_position}

    def step(self, action):
        print(self.current_position)
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # action: 0=up, 1=right, 2=down, 3=left
        move = moves[action]
        next_position = (self.current_position[0] + move[0], self.current_position[1] + move[1])
        if self.is_valid_move(*next_position):
            self.current_position = next_position
        if self.current_position == self.goal_position:
            return self.obs(), 1, True, False, self.info()
        else:
            return self.obs(), -1, False, False, self.info()

    def reset(self):
        self.maze = generate_maze(self.width, self.height)
        self.start_position = next(p for p in self.get_random_positions() if self.maze[p])
        self.goal_position = next(p for p in self.get_random_positions() if self.maze[p] and p != self.start_position)
        self.current_position = self.start_position
        return self.obs(), self.info()


if __name__ == "__main__":
    import sys
    import cv2
    import pygame
    assert len(sys.argv) == 3, "Usage: python environment.py <width> <height>"

    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    WIDTH, HEIGHT = int(sys.argv[1]), int(sys.argv[2])
    SCALE = 10

    pygame.init()
    screen = pygame.display.set_mode((WIDTH * SCALE, HEIGHT * SCALE))
    clock = pygame.time.Clock()
    env = MazeEnv(WIDTH, HEIGHT)

    def draw(obs, *_):
        screen.fill(WHITE)
        maze_img = cv2.resize(obs, (WIDTH * SCALE, HEIGHT * SCALE), interpolation=cv2.INTER_NEAREST)
        maze_img = pygame.surfarray.make_surface(maze_img.swapaxes(0, 1))
        screen.blit(maze_img, (0, 0))
        pygame.display.flip()

    draw(*env.reset())

    while True:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    action = 0
                elif event.key == pygame.K_d:
                    action = 1
                elif event.key == pygame.K_s:
                    action = 2
                elif event.key == pygame.K_a:
                    action = 3

        if action is None:
            continue

        obs, reward, done, _, _ = env.step(action)
        draw(obs)

        if done:
            draw(*env.reset())
        clock.tick(5)
