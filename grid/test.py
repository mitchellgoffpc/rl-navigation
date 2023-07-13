#!/usr/bin/env python
import curses
from grid.environment import GridEnvironment

def main(stdscr):
  env = GridEnvironment(10, 10)
  grid, _ = env.reset()

  while True:
    stdscr.clear()

    for i in range(grid.shape[0]):
      for j in range(grid.shape[1]):
        if grid[i, j]:
          stdscr.addstr(i, j*2, 'X')
        else:
          stdscr.addstr(i, j*2, 'O')

    stdscr.refresh()

    c = stdscr.getch()
    if c == ord('w'):
      grid, _, _, _ = env.step(0)
    elif c == ord('s'):
      grid, _, _, _ = env.step(1)
    elif c == ord('a'):
      grid, _, _, _ = env.step(2)
    elif c == ord('d'):
      grid, _, _, _ = env.step(3)
    elif c == ord('q'):
      break


if __name__ == "__main__":
    curses.wrapper(main)
