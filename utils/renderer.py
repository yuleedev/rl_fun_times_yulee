"""The look of the original 2048, drawn with pygame.

The env's own renderer maps tile values through matplotlib's YlOrBr colormap,
which makes every tile a shade of the same brown and is hard to read at a
glance. The real game uses a fixed palette where each tile value has its own
colour, so you can tell a 512 from a 1024 without reading the number.

This is a drop-in drawing surface, not an environment. Hand it a board of tile
values and it draws a frame:

    renderer = Board2048Renderer()
    renderer.draw(env.board, env.total_score, status="up +48")

Colours and proportions are taken from the original game: a #bbada0 board with
#cdc1b4 empty cells, warm tiles up to 2048, then near-black for anything past
that. Font size steps down as the numbers get longer so 16384 still fits.
"""

from typing import Optional, Sequence

import pygame

BOARD_BACKGROUND = (187, 173, 160)
EMPTY_CELL = (205, 193, 180)
PAGE_BACKGROUND = (250, 248, 239)
DARK_TEXT = (119, 110, 101)
LIGHT_TEXT = (249, 246, 242)
SCORE_BOX = (187, 173, 160)
SCORE_LABEL = (238, 228, 218)

TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
BEYOND_2048 = (60, 58, 50)

CELL_SIZE = 85
CELL_GAP = 12
BOARD_SIZE = 4 * CELL_SIZE + 5 * CELL_GAP
HEADER_HEIGHT = 96
WINDOW_SIZE = (BOARD_SIZE, BOARD_SIZE + HEADER_HEIGHT)

# Original 2048 uses Clear Sans; these are the usual stand-ins per platform
FONT_CANDIDATES = "clearsans,helveticaneue,helvetica,arialroundedmtbold,arial"


def tile_color(value: int):
    if value in TILE_COLORS:
        return TILE_COLORS[value]
    return BEYOND_2048


def tile_text_color(value: int):
    # the two palest tiles keep the dark text, everything else goes light
    return DARK_TEXT if value in (2, 4) else LIGHT_TEXT


def _font_size(value: int) -> int:
    digits = len(str(value))
    if digits <= 2:
        return 44
    if digits == 3:
        return 38
    if digits == 4:
        return 30
    return 24


class Board2048Renderer:
    """Owns a pygame window and draws boards into it."""

    def __init__(self, caption: str = "2048 Avalanche", window=None):
        """
        Args:
            caption: window title
            window: an existing pygame surface to draw on. Leave it None to open
                a window of the right size.
        """
        if not pygame.get_init():
            pygame.init()
        if window is None:
            pygame.display.init()
            window = pygame.display.set_mode(WINDOW_SIZE)
            pygame.display.set_caption(caption)
        self.window = window
        pygame.font.init()
        self._fonts = {}
        self.label_font = pygame.font.SysFont(FONT_CANDIDATES, 15, bold=True)
        self.score_font = pygame.font.SysFont(FONT_CANDIDATES, 30, bold=True)
        self.title_font = pygame.font.SysFont(FONT_CANDIDATES, 34, bold=True)
        self.status_font = pygame.font.SysFont(FONT_CANDIDATES, 14)

    def _tile_font(self, value: int):
        size = _font_size(value)
        if size not in self._fonts:
            self._fonts[size] = pygame.font.SysFont(FONT_CANDIDATES, size, bold=True)
        return self._fonts[size]

    def _fit(self, text: str, max_width: int, max_size: int):
        """Largest font that keeps `text` inside `max_width`.

        Scores run past six figures late in a game and would otherwise spill out
        of the score box.
        """
        size = max_size
        while size > 10:
            font = self._tile_font_at(size)
            if font.size(text)[0] <= max_width:
                return font
            size -= 2
        return self._tile_font_at(10)

    def _tile_font_at(self, size: int):
        if size not in self._fonts:
            self._fonts[size] = pygame.font.SysFont(FONT_CANDIDATES, size, bold=True)
        return self._fonts[size]

    def draw(
        self,
        board: Sequence[Sequence[int]],
        score: int = 0,
        status: Optional[str] = None,
        highest_tile: Optional[int] = None,
        flip: bool = True,
    ) -> None:
        """Draw one frame. `board` is 4x4 of tile values, 0 meaning empty."""
        self.window.fill(PAGE_BACKGROUND)
        self._draw_header(score, status, highest_tile)
        self._draw_board(board)
        if flip:
            pygame.display.update()

    def _draw_header(self, score: int, status: Optional[str], highest_tile) -> None:
        title = self.title_font.render("2048", True, DARK_TEXT)
        self.window.blit(title, (CELL_GAP, 14))
        subtitle = self.status_font.render("avalanche", True, (160, 150, 140))
        self.window.blit(subtitle, (CELL_GAP + 2, 52))

        boxes = [("SCORE", f"{score:,}")]
        if highest_tile is not None:
            boxes.append(("BEST TILE", str(highest_tile)))

        box_width = 104
        right = BOARD_SIZE - CELL_GAP
        for label, value in reversed(boxes):
            rect = pygame.Rect(right - box_width, 12, box_width, 52)
            pygame.draw.rect(self.window, SCORE_BOX, rect, border_radius=4)
            label_surface = self.label_font.render(label, True, SCORE_LABEL)
            self.window.blit(
                label_surface, label_surface.get_rect(center=(rect.centerx, rect.y + 15))
            )
            value_surface = self._fit(value, box_width - 12, 30).render(
                value, True, (255, 255, 255)
            )
            self.window.blit(
                value_surface,
                value_surface.get_rect(center=(rect.centerx, rect.y + 36)),
            )
            right -= box_width + 8

        if status:
            surface = self.status_font.render(status, True, (160, 150, 140))
            self.window.blit(surface, (CELL_GAP + 2, HEADER_HEIGHT - 24))

    def _draw_board(self, board: Sequence[Sequence[int]]) -> None:
        pygame.draw.rect(
            self.window,
            BOARD_BACKGROUND,
            pygame.Rect(0, HEADER_HEIGHT, BOARD_SIZE, BOARD_SIZE),
            border_radius=6,
        )
        for row_index in range(4):
            for col_index in range(4):
                value = int(board[row_index][col_index])
                x = CELL_GAP + col_index * (CELL_SIZE + CELL_GAP)
                y = HEADER_HEIGHT + CELL_GAP + row_index * (CELL_SIZE + CELL_GAP)
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(
                    self.window,
                    tile_color(value) if value else EMPTY_CELL,
                    rect,
                    border_radius=4,
                )
                if value:
                    text = self._tile_font(value).render(
                        str(value), True, tile_text_color(value)
                    )
                    self.window.blit(text, text.get_rect(center=rect.center))

    def draw_game_over(self, score: int) -> None:
        """Wash out the board and print the final score, as the original does."""
        overlay = pygame.Surface((BOARD_SIZE, BOARD_SIZE), pygame.SRCALPHA)
        overlay.fill((238, 228, 218, 180))
        self.window.blit(overlay, (0, HEADER_HEIGHT))
        message = self.title_font.render("Game over", True, DARK_TEXT)
        self.window.blit(
            message,
            message.get_rect(center=(BOARD_SIZE // 2, HEADER_HEIGHT + BOARD_SIZE // 2 - 18)),
        )
        final = self.score_font.render(f"{score:,}", True, DARK_TEXT)
        self.window.blit(
            final,
            final.get_rect(center=(BOARD_SIZE // 2, HEADER_HEIGHT + BOARD_SIZE // 2 + 20)),
        )
        pygame.display.update()

    def close(self) -> None:
        pygame.display.quit()
        pygame.quit()
