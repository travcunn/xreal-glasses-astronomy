"""Rasterize short strings to GL textures for billboard labels."""

import pygame
import moderngl


def make_label_texture(ctx: moderngl.Context, text: str, size: int = 22):
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.SysFont("Helvetica", size)
    # Antialiased render already carries per-pixel alpha; no convert (which would
    # require a video mode and break headless use).
    surf = font.render(text, True, (220, 220, 255))
    w, h = surf.get_size()
    data = pygame.image.tobytes(surf, "RGBA", True)
    tex = ctx.texture((w, h), 4, data)
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return tex, (w, h)
