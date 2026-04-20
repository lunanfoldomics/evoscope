import pygame
import numpy as np
import math
import glob
import time
import re

SIZE = 10

COLORS = {
    -1: (30,30,30),   # empty
    0: (200,200,200), # undetermined
    1: (255,0,0),
    2: (0,255,0),
    3: (255,255,0),
    4: (0,0,255),
    5: (255,0,255),
    6: (0,255,255),
    7: (255,255,255),
    8: (120,120,120),
}

'''
def hex_to_pixel(q, r, size):
    x = size * (3/2 * q)
    y = size * (math.sqrt(3) * (r + q/2))
    return int(x), int(y)
'''

def hex_to_pixel(q, r, size):
    import math
    x = size * (math.sqrt(3) * (q + r/2))
    y = size * (3/2 * r)
    return int(x), int(y)


def draw_hex(surface, x, y, size, color):
    points = []
    for i in range(6):
        angle = math.pi / 3 * i
        px = x + size * math.cos(angle)
        py = y + size * math.sin(angle)
        points.append((px, py))
    pygame.draw.polygon(surface, color, points)



def load_snapshots():
    files = glob.glob("snapshots/grid_*.npy")

    def extract_number(f):
        return int(re.search(r"grid_(\d+)\.npy", f).group(1))

    return sorted(files, key=extract_number)    

'''
def compute_center_offset(w, h, size):
    import math
    width_px = size * (3/2 * w)
    height_px = size * (math.sqrt(3) * (h + w/2))

    screen_h, screen_w = 1400, 950

    offset_x = (screen_w - width_px) // 2
    offset_y = (screen_h - height_px) // 2

    return offset_x, offset_y
'''


def compute_size(w, h, screen_w, screen_h):
    import math

    size_w = screen_w / (3/2 * w + 1)
    size_h = screen_h / (math.sqrt(3) * (h + w/2))

    return int(min(size_w, size_h))


def compute_center_offset(w, h, size, screen_w, screen_h):
    import math

    width_px = size * (3/2 * (w - 1)) + size * 2
    height_px = size * (math.sqrt(3) * (h + 0.5))

    offset_x = (screen_w - width_px) / 2
    offset_y = (screen_h - height_px) / 2

    return int(offset_x), int(offset_y)


def main():
    pygame.init()

    files = load_snapshots()
    if not files:
        print("No snapshots found")
        return

    first = np.load(files[0])
    h, w = first.shape

    screen_w, screen_h = 1400, 800
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Torex Viewer")

    hex_size = compute_size(w, h, screen_w, screen_h)
    offset_x, offset_y = compute_center_offset(w, h, hex_size, screen_w, screen_h)

    clock = pygame.time.Clock()
    running = True
    idx = 0
    paused = False
    font = pygame.font.SysFont("Arial", 20)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused:
                    idx = (idx + 1) % len(files)
                elif event.key == pygame.K_LEFT and paused:
                    idx = (idx - 1) % len(files)

        grid = np.load(files[idx])

        #screen.fill((0, 0, 0))
        screen.fill((240,240,240))

        for r in range(grid.shape[0]):
            for q in range(grid.shape[1]):
                val = int(grid[r, q])
                color = COLORS.get(val, (255, 255, 255))
                x, y = hex_to_pixel(q, r, hex_size)
                draw_hex(screen, x + offset_x, y + offset_y, hex_size, color)

        status = "PAUSED" if paused else "PLAYING"
        text = font.render(f"{status}  frame={idx+1}/{len(files)}", True, (0, 0, 0))
        screen.blit(text, (20, 20))

        pygame.display.flip()

        if not paused:
            idx = (idx + 1) % len(files)

        clock.tick(20)

    pygame.quit()

if __name__ == "__main__":
    main()