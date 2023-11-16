import cv2
import time
import random
import pygame
import argparse
import numpy as np
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


# ENTRY POINT

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default=False, action='store_true')
  args = parser.parse_args()

  env = ZeldaEnvironment()
  frame, _ = env.reset()
  goal_pos = (120, 120, 7, 7, 0)
  screen_h, screen_w = frame.shape[0] * 2, frame.shape[1] * 2
  footer_h = screen_h // 12

  pygame.init()
  screen = pygame.display.set_mode((screen_w, screen_h))

  def draw(image):
    pygame.surfarray.blit_array(screen, cv2.resize(image, (screen_w, screen_h)).swapaxes(0, 1))
    pygame.display.flip()

  policy = None
  if args.model:
    import torch
    from zelda.models import ZeldaAgent
    policy = ZeldaAgent(4).eval()
    policy.load_state_dict(torch.load(Path(__file__).parent / 'checkpoints' / 'policy.ckpt'))
    torch.set_grad_enabled(False)

  while True:
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_q] or pressed[pygame.K_ESCAPE] or any(e.type == pygame.QUIT for e in pygame.event.get()):
      break
    elif pressed[pygame.K_r]:
      frame, info = env.reset()
    elif args.model:
      action_preds = policy(state, goal).softmax(-1)
      action = torch.multinomial(action_preds, 1).numpy().item()
      frame, info = env.step(action)
    else:
      action = sum(KEYMAP[k] for k in KEYMAP if pressed[k])
      frame, info = env.step(action)

    pos_x, pos_y = info['screen_pos']
    map_x, map_y, map_l = info['map_pos']
    current_pos = (pos_x, pos_y, map_x, map_y, map_l)
    footer = np.zeros((footer_h, screen_w, 3), dtype=np.uint8)
    frame = cv2.resize(frame, (screen_w, screen_h))
    footer_msg = f'POS: X={pos_x}, Y={pos_y} | MAP: X={map_x}, Y={map_y}, L={map_l} | DONE: {env.pos_matches(current_pos, goal_pos)}'
    cv2.putText(footer, footer_msg, (10, footer_h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255))
    frame = np.concatenate([frame, footer], axis=0)
    draw(frame)
    time.sleep(0.01)
