import cv2
import time
import random
import pygame
import argparse
import numpy as np
from pathlib import Path
from zelda.environment import ZeldaEnvironment

KEYMAP = {
  pygame.K_j: ZeldaEnvironment.A,
  pygame.K_k: ZeldaEnvironment.B,
  pygame.K_g: ZeldaEnvironment.SELECT,
  pygame.K_h: ZeldaEnvironment.START,
  pygame.K_w: ZeldaEnvironment.UP,
  pygame.K_s: ZeldaEnvironment.DOWN,
  pygame.K_a: ZeldaEnvironment.LEFT,
  pygame.K_d: ZeldaEnvironment.RIGHT}

ACTIONS = [ZeldaEnvironment.UP, ZeldaEnvironment.DOWN, ZeldaEnvironment.LEFT, ZeldaEnvironment.RIGHT]


# ENTRY POINT

if __name__ == '__main__':
  pygame.init()
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default=False, action='store_true')
  args = parser.parse_args()

  env = ZeldaEnvironment()
  screen_h, screen_w = env.observation_space.shape[0] * 2, env.observation_space.shape[1] * 2
  footer_h = screen_h // 12
  screen = pygame.display.set_mode((screen_w * 2, screen_h + footer_h))

  policy = None
  if args.model:
    import torch
    from zelda.models import ZeldaAgent
    policy = ZeldaAgent(4).eval()
    policy.load_state_dict(torch.load(Path(__file__).parent / 'checkpoints' / 'policy.ckpt'))
    torch.set_grad_enabled(False)

  frame, info = env.reset()
  goal, goal_pos = np.zeros((screen_h, screen_w, 3), dtype=np.float32), None
  while True:
    events = pygame.event.get()
    keydowns = {e.key for e in events if e.type == pygame.KEYDOWN}
    quit = any(e.type == pygame.QUIT for e in events)
    pressed = pygame.key.get_pressed()

    if pygame.K_q in keydowns or pygame.K_ESCAPE in keydowns or quit:
      break
    elif pygame.K_r in keydowns:
      goal, goal_pos = goal * 0, None
      frame, info = env.reset()
    elif pygame.K_TAB in keydowns:
      goal, goal_pos = frame.copy(), info['pos']
      frame, info = env.reset()
    elif args.model and goal_pos:
      action_preds = policy(frame[None], goal[None]).softmax(-1)
      action = torch.multinomial(action_preds, 1).numpy().item()
      frame, info = env.step(ACTIONS[action])
    else:
      action = sum(KEYMAP[k] for k in KEYMAP if pressed[k])
      frame, info = env.step(action)

    pos_x, pos_y, map_x, map_y, map_l = current_pos = info['pos']
    done = goal_pos and env.pos_matches(current_pos, goal_pos)
    current_frame = cv2.resize(frame, (screen_w, screen_h))
    goal_frame = cv2.resize(goal, (screen_w, screen_h))
    footer = np.zeros((footer_h, screen_w * 2, 3), dtype=np.uint8)
    footer_msg = f'POS: X={pos_x}, Y={pos_y} | MAP: X={map_x}, Y={map_y}, L={map_l} | DONE: {done}'
    cv2.putText(footer, footer_msg, (10, footer_h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255))

    full_frame = np.concatenate([current_frame, goal_frame], axis=1)
    full_frame = np.concatenate([full_frame, footer], axis=0)
    pygame.surfarray.blit_array(screen, full_frame.swapaxes(0, 1))
    pygame.display.flip()
    time.sleep(0.01)
