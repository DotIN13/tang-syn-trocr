import os
import yaml
import random
import warnings

from tqdm import tqdm
import numpy as np
import pygame
from fontTools.ttLib import TTFont

warnings.filterwarnings('ignore', category=UserWarning,
                        module='fontTools.ttLib.tables')

FALLBACK_FONT_NAMES = [
    "NotoSansCJKsc-VF.ttf",
    "NotoSansCJKhk-VF.ttf",
    "NotoSansCJKjp-VF.ttf",
    "NotoSansCJKkr-VF.ttf",
    "NotoSansCJKtc-VF.ttf",
    "Arial-Unicode-MS.ttf",
    "segoeui.ttf",
    "segoeuib.ttf",
    "segoeuii.ttf",
    "segoeuil.ttf",
    "segoeuisl.ttf",
    "segoeuiz.ttf",
    "seguibl.ttf",
    "seguibli.ttf",
    "seguihis.ttf",
    "seguili.ttf",
    "seguisb.ttf",
    "seguisbi.ttf",
    "seguisli.ttf",
    "seguisym.ttf",
    "seguiemj.ttf",
]


def can_render(cmaps, character):
    for cmap in cmaps:
        if ord(character) in cmap:
            return True

    return False


def apply_color_jitter(color, std=15):
    jittered_color = [max(
        0, min(255, c + int(np.random.normal(c, std)))) for c in color]
    return tuple(jittered_color)


def jitter_color_triplets(triplet):
    return tuple(apply_color_jitter(color) for color in triplet)


def calculate_luminance(color):
    """Calculate the luminance of a color."""
    # Convert the RGB values to the sRGB space
    sRGB = [c / 255.0 for c in color]

    # Calculate the linear RGB values
    linearRGB = [c / 12.92 if c <=
                 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in sRGB]

    # Calculate the luminance
    return 0.2126 * linearRGB[0] + 0.7152 * linearRGB[1] + 0.0722 * linearRGB[2]


def generate_complementary_color(luminance):
    """Generate a color with good contrast to the input color."""
    # If luminance is high, generate a brighter color
    # Otherwise, generate a darker color

    if luminance > 0.5:
        luminance += -0.15
        return [random.randint(0, int(255 * luminance)) for _ in range(3)]
    else:
        luminance += 0.15
        return [random.randint(int(255 * luminance), 255) for _ in range(3)]


def generate_color_triplet():
    color1 = [random.randint(0, 255) for _ in range(3)]
    text_luminance = calculate_luminance(color1)

    color2 = generate_complementary_color(text_luminance)
    color3 = generate_complementary_color(text_luminance)

    # text, bg, box
    return color1, color2, color3


def pick_predefined_color_triplet():
    predefined_color_triplets = [
        # Black text, white background, black box
        ((0, 0, 0), (255, 255, 255), (0, 0, 0)),
        # Blue text, white background, black box
        ((0, 0, 255), (255, 255, 255), (0, 0, 0)),
        # Black text, yellow background, black box
        ((0, 0, 0), (255, 255, 0), (0, 0, 0)),
        # Black text, white background, light gray box
        ((0, 0, 0), (255, 255, 255), (50, 50, 50)),
        # Black text, white background, light blue box
        ((0, 0, 0), (255, 255, 255), (173, 216, 230)),
    ]

    colors = random.choice(predefined_color_triplets)
    return jitter_color_triplets(colors)


def pygame_font(pth, size):
    """
    Creates a Pygame font object from a given font file path and size.

    Args:
        pth (str): The file path of the font file.
        size (int): The size of the font.

    Returns:
        A Pygame font object.
    """
    return pygame.freetype.Font(pth, size)


def ttfont_cmaps(pth):
    """
    Given a font file path, returns a list of its character maps.

    Args:
        pth (str): The file path of the font file.

    Returns:
        A list of character maps for the font.
    """
    ttfont = TTFont(pth)
    return [table.cmap for table in ttfont["cmap"].tables]


class TextlineSynthesisConfig:
    with open('tang_syn_config.yaml', 'r') as f:
        DEFAULT_CONFIG = yaml.safe_load(f)

    pygame.freetype.init()

    if os.environ.get("DEBUG"):
        FALLBACK_FONT_NAMES = ["AaYueRanTi-2.ttf"]

    FALLBACK_FONTS = []
    for fontname in tqdm(FALLBACK_FONT_NAMES):
        pth = os.path.join("fonts", fontname)
        FALLBACK_FONTS.append(
            (pygame_font(pth, DEFAULT_CONFIG["font_size"]), ttfont_cmaps(pth)))

    FONTS = []
    for file in tqdm(os.listdir("fonts")):
        if file.lower().endswith((".ttf", ".otf")):
            if DEFAULT_CONFIG["simplified_fonts_only"] and "FW" in file:
                continue

            pth = os.path.join("fonts", file)
            FONTS.append((pth, ttfont_cmaps(pth)))

            if os.environ.get("DEBUG"):
                break

    print(f"Usable font: {len(FONTS)}")

    def __init__(self, config={}):
        self.config = self.DEFAULT_CONFIG.copy()

        self.config.update(config)

        self.config["canvas_height"] = self.config["height"] - \
            self.config["margin_top"] - self.config["margin_bottom"]

        if self.config["limit_font_size"]:
            self.config["font_size"] = min(
                self.config["canvas_height"], self.config["font_size"])

        font_pth, cmaps = random.choice(self.FONTS)
        self.config["font"] = (pygame_font(
            font_pth, self.config["font_size"]), cmaps)

    def __getattr__(self, item):
        if item in self.config:
            return self.config[item]

        raise AttributeError(f"Object has no attribute '{item}'")

    @classmethod
    def random_config(cls):
        random_config = cls.DEFAULT_CONFIG.copy()

        random_font_size = random_config.get("random_font_size", False)
        if random_font_size:
            random_config["font_size"] = random.randint(
                random_config["min_font_size"], random_config["max_font_size"])

        random_config["font_size_jittor"] = (
            random.random() < random_config["font_size_jittor_prob"])

        random_config["font_weight_jittor"] = (
            random.random() < random_config["font_weight_jittor_prob"])

        random_margin = random_config.get("random_margin", False)
        if random_margin is not None:
            random_config['margin_left'] *= random.random()
            random_config['margin_right'] *= random.random()
            random_config['margin_top'] *= random.random()
            random_config['margin_bottom'] *= random.random()

        # 30% of the time, apply random colors
        color_prob = random.random()
        if color_prob < random_config.get("random_color_prob", 0.0):
            colors = generate_color_triplet()
        else:
            colors = pick_predefined_color_triplet()

        random_config['text_color'] = colors[0]
        random_config['bg_color'] = colors[1]
        random_config['box_color'] = colors[2]

        # 20% of the time, apply graph grids
        # 30% of the time, apply chinese grids
        random_config['graph_grid'] = False
        random_config['chinese_grid'] = False
        grid_prob = random.random()
        if grid_prob < 0.1:  # 10% chance of graph grid
            random_config['graph_grid'] = True
        elif grid_prob < 0.3:  # 20% chance of Chinese grid
            random_config['chinese_grid'] = True

        random_config['graph_grid_size'] = np.random.randint(
            random_config["graph_grid_min_size"], random_config["graph_grid_max_size"])

        random_config['chinese_grid_padding'] = np.random.randint(
            random_config['chinese_grid_min_padding'], random_config['chinese_grid_max_padding'])

        # 40% of the time, apply elastic transform
        elastic_prob = random.random()
        random_config['elastic_transform'] = elastic_prob < random_config.get(
            'elastic_transform_prob', 0.0)

        random_config["random_crossout"] = (
            random.random() < random_config["random_crossout_prob"])
        if random_config["random_crossout"]:
            random_config["random_crossout_rotation"] = np.random.uniform(
                -60, 60)
            random_config["random_crossout_thickness"] = random.randint(2, 3)

        return cls(random_config)

    # Optional: add a method to modify configuration values
    def set_value(self, key, value):
        self.config[key] = value


if __name__ == "__main__":
    TextlineSynthesisConfig.random_config()
