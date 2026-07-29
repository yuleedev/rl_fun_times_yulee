"""runnable file! Replays a recorded game in the original 2048 GUI.

    python play_ntuple.py --games 30 --budget 2000 --seed 1 --record replays/best.json
    python replay_pygame.py --file replays/best.json

Reads the json that play_ntuple.py --record writes and steps utils/renderer.py
through it, so you can watch a game you already played instead of hoping the
next one goes well.

There is a control bar under the board: restart, play/pause, and a speed slider.
Restart drops you back to the opening board and stays paused, so you can start
watching when you are ready rather than the moment you click.

Speed is in moves per second, not frames per second. Past 60 the viewer stops
drawing every move and advances several per frame instead, so 4000 moves/sec
costs no more to draw than 60 does. A 5,000 move game replays in about a second
at the top of the slider.

keyboard, if you prefer it
    space / enter   play or pause
    1 2 3 4 5       speed presets: 8, 60, 240, 1000, 4000 moves/sec
    up / down       double or halve the speed
    right / left    one move forward or back
    . / ,           jump 100 moves
    ] / [           jump 1000 moves
    r               restart
    e               skip to the end
    q / esc         quit
"""

import argparse
import json
import math
import os

import pygame

from utils.renderer import BOARD_SIZE, HEADER_HEIGHT, Board2048Renderer

DEFAULT_FILE = os.path.join("replays", "best.json")
ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}

MIN_SPEED = 1
MAX_SPEED = 8192
RENDER_FPS = 60  # no point drawing faster than a screen refreshes
SPEED_PRESETS = {
    pygame.K_1: 8,
    pygame.K_2: 60,
    pygame.K_3: 240,
    pygame.K_4: 1000,
    pygame.K_5: 4000,
}

CONTROLS_HEIGHT = 62
CONTROLS_TOP = HEADER_HEIGHT + BOARD_SIZE
WINDOW_SIZE = (BOARD_SIZE, CONTROLS_TOP + CONTROLS_HEIGHT)

BUTTON_COLOR = (143, 122, 102)
BUTTON_HOVER = (158, 137, 117)
BUTTON_TEXT = (249, 246, 242)
TRACK_COLOR = (205, 193, 180)
TRACK_FILL = (143, 122, 102)
MUTED_TEXT = (160, 150, 140)

RESTART_RECT = pygame.Rect(12, CONTROLS_TOP + 16, 78, 30)
PLAY_RECT = pygame.Rect(98, CONTROLS_TOP + 16, 78, 30)
SLIDER_X = 196
SLIDER_WIDTH = BOARD_SIZE - SLIDER_X - 16
SLIDER_Y = CONTROLS_TOP + 31
SLIDER_GRAB = pygame.Rect(SLIDER_X - 8, CONTROLS_TOP + 14, SLIDER_WIDTH + 16, 34)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument(
        "--speed", type=int, default=8, help="moves per second, 1 to 8192"
    )
    parser.add_argument("--start", type=int, default=0, help="move to open on")
    parser.add_argument("--paused", action="store_true")
    return parser.parse_args()


def speed_to_fraction(speed):
    """where a speed sits on the slider, 0 to 1, on a log scale"""
    return math.log2(speed) / math.log2(MAX_SPEED)


def fraction_to_speed(fraction):
    fraction = max(0.0, min(1.0, fraction))
    return int(round(2 ** (fraction * math.log2(MAX_SPEED))))


def board_from_hex(packed_hex):
    packed = int(packed_hex, 16)
    return [
        [
            (1 << exponent) if exponent else 0
            for exponent in (
                (packed >> (4 * (4 * row + col))) & 0xF for col in range(4)
            )
        ]
        for row in range(4)
    ]


def build_frames(game):
    """board, score, caption and best tile for every point in the game"""
    frames = [(board_from_hex(game["start"]), 0, "start", 0)]
    score = 0
    best_tile = 0
    for move_number, step in enumerate(game["steps"], start=1):
        score += step["r"]
        board = board_from_hex(step["b"])
        best_tile = max(best_tile, max(max(row) for row in board))
        caption = f"move {move_number:,}   {ACTION_NAMES[step['a']]}   {step['r']:+}"
        frames.append((board, score, caption, best_tile))
    return frames


def draw_controls(window, font, small_font, playing, speed, mouse_position):
    """the bar under the board: restart, play/pause, speed slider"""
    for rect, label in ((RESTART_RECT, "Restart"), (PLAY_RECT, "Pause" if playing else "Play")):
        hovered = rect.collidepoint(mouse_position)
        pygame.draw.rect(
            window, BUTTON_HOVER if hovered else BUTTON_COLOR, rect, border_radius=4
        )
        text = font.render(label, True, BUTTON_TEXT)
        window.blit(text, text.get_rect(center=rect.center))

    pygame.draw.rect(
        window, TRACK_COLOR, pygame.Rect(SLIDER_X, SLIDER_Y - 3, SLIDER_WIDTH, 6),
        border_radius=3,
    )
    fraction = speed_to_fraction(speed)
    filled = int(SLIDER_WIDTH * fraction)
    if filled > 0:
        pygame.draw.rect(
            window, TRACK_FILL, pygame.Rect(SLIDER_X, SLIDER_Y - 3, filled, 6),
            border_radius=3,
        )
    knob_x = SLIDER_X + filled
    pygame.draw.circle(window, TRACK_FILL, (knob_x, SLIDER_Y), 9)
    pygame.draw.circle(window, BUTTON_TEXT, (knob_x, SLIDER_Y), 4)

    label = small_font.render(f"{speed:,} moves/sec", True, MUTED_TEXT)
    window.blit(label, (SLIDER_X, SLIDER_Y + 12))


def main():
    args = parse_args()
    with open(args.file, encoding="utf-8") as handle:
        game = json.load(handle)

    frames = build_frames(game)
    print(
        f"{args.file}: {game['score']:,} points, tile {game['max_tile']}, "
        f"{game['moves']:,} moves"
    )

    pygame.init()
    pygame.display.init()
    window = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption(f"2048 Avalanche  -  {game['score']:,}")
    renderer = Board2048Renderer(window=window)
    button_font = pygame.font.SysFont("helveticaneue,helvetica,arial", 14, bold=True)
    small_font = pygame.font.SysFont("helveticaneue,helvetica,arial", 12)

    index = max(0, min(args.start, len(frames) - 1))
    speed = max(MIN_SPEED, min(args.speed, MAX_SPEED))
    playing = not args.paused
    dragging = False
    clock = pygame.time.Clock()
    running = True

    while running:
        mouse_position = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if RESTART_RECT.collidepoint(event.pos):
                    # back to the opening board, and wait there
                    index = 0
                    playing = False
                elif PLAY_RECT.collidepoint(event.pos):
                    if not playing and index >= len(frames) - 1:
                        index = 0  # finished, so play means play it again
                    playing = not playing
                elif SLIDER_GRAB.collidepoint(event.pos):
                    dragging = True
                    speed = fraction_to_speed((event.pos[0] - SLIDER_X) / SLIDER_WIDTH)

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False

            elif event.type == pygame.MOUSEMOTION and dragging:
                speed = fraction_to_speed((event.pos[0] - SLIDER_X) / SLIDER_WIDTH)

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE, pygame.K_BACKSPACE):
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                    if not playing and index >= len(frames) - 1:
                        index = 0
                    playing = not playing
                elif event.key == pygame.K_RIGHT:
                    playing = False
                    index = min(index + 1, len(frames) - 1)
                elif event.key == pygame.K_LEFT:
                    playing = False
                    index = max(index - 1, 0)
                elif event.key == pygame.K_PERIOD:
                    playing = False
                    index = min(index + 100, len(frames) - 1)
                elif event.key == pygame.K_COMMA:
                    playing = False
                    index = max(index - 100, 0)
                elif event.key == pygame.K_RIGHTBRACKET:
                    playing = False
                    index = min(index + 1000, len(frames) - 1)
                elif event.key == pygame.K_LEFTBRACKET:
                    playing = False
                    index = max(index - 1000, 0)
                elif event.key in SPEED_PRESETS:
                    speed = SPEED_PRESETS[event.key]
                elif event.key == pygame.K_UP:
                    speed = min(speed * 2, MAX_SPEED)
                elif event.key == pygame.K_DOWN:
                    speed = max(speed // 2, MIN_SPEED)
                elif event.key == pygame.K_r:
                    index = 0
                    playing = False
                elif event.key == pygame.K_e:
                    index = len(frames) - 1
                    playing = False

        # Above 60 moves/sec the limit is the screen, not the game, so advance
        # several moves per drawn frame instead of drawing faster.
        if speed <= RENDER_FPS:
            moves_per_frame = 1
            tick_rate = max(speed, 1)
        else:
            moves_per_frame = max(1, round(speed / RENDER_FPS))
            tick_rate = RENDER_FPS

        board, score, caption, best_tile = frames[index]
        renderer.draw(
            board,
            score,
            status=f"{caption}      {index:,}/{len(frames) - 1:,}",
            highest_tile=best_tile,
            flip=False,
        )
        draw_controls(window, button_font, small_font, playing, speed, mouse_position)
        pygame.display.update()

        if playing:
            if index >= len(frames) - 1:
                playing = False
            else:
                index = min(index + moves_per_frame, len(frames) - 1)
        clock.tick(tick_rate if playing else 30)

    renderer.close()
    print(f"stopped at move {index:,} of {len(frames) - 1:,}")


if __name__ == "__main__":
    main()
