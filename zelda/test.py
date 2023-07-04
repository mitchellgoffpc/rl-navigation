import cv2
import time
import random
import pygame
import argparse
import numpy as np
from zelda.environment import ZeldaEnvironment

KEYMAP = [pygame.K_j, pygame.K_k, pygame.K_g, pygame.K_h, pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d]

def get_action(keys):
  return sum((1 << i) * k for i, k in enumerate(keys))

def step_from_keyboard(env):
  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      return None, False, False

  pressed = pygame.key.get_pressed()
  running = not pressed[pygame.K_ESCAPE] and not pressed[pygame.K_q]
  action = get_action([pressed[key] for key in KEYMAP])
  if pressed[pygame.K_r]:
    frame = env.reset()
  else:
    frame, _, _, _ = env.step(action)
  return frame, running

def step_from_model(model, frame, goal):
  torch_frame = torch.as_tensor(frame[None]).permute(0,3,1,2).float()
  torch_goal = torch.as_tensor(goal[None]).permute(0,3,1,2).float()
  action_probs = agent(torch_frame, torch_goal)[0]
  # action = torch.argmax(action_probs)
  action, = random.choices(range(len(action_probs)), weights=torch.softmax(action_probs, 0))
  return action, True, False


# ENTRY POINT

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default=False, action='store_true')
  args = parser.parse_args()

  env = ZeldaEnvironment()
  frame = env.reset()
  screen_h, screen_w = frame.shape[0] * 2, frame.shape[1] * 2

  pygame.init()
  screen = pygame.display.set_mode((screen_w, screen_h))

  def draw(image):
    pygame.surfarray.blit_array(screen, cv2.resize(image, (screen_w, screen_h)).swapaxes(0, 1))
    pygame.display.flip()

  agent = None
  if args.model:
    import torch
    from zelda.train import Agent
    agent = Agent(128).eval()
    agent.load_state_dict(torch.load('checkpoint.pt'))
    torch.set_grad_enabled(False)

  running = True
  while running:
    if args.model:
      frame, running = step_from_model(env, model, frame, goal)
    else:
      frame, running = step_from_keyboard(env)

    pos_x, pos_y = env.screen_pos
    map_x, map_y = env.map_pos
    footer = np.zeros((screen_h // 8, screen_w, 3), dtype=np.uint8)
    frame = cv2.resize(frame, (screen_w, screen_h))
    cv2.putText(footer, f'POS: X={pos_x}, Y={pos_y} | MAP: X={map_x}, Y={map_y}', (10,screen_h//8-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))
    frame = np.concatenate([frame, footer], axis=0)
    draw(frame)
    time.sleep(0.01)
