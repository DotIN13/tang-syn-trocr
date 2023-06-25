import os
import yaml
import random
import warnings

from tqdm import tqdm
import numpy as np
import pygame
from fontTools.ttLib import TTFont

warnings.filterwarnings('ignore', category=UserWarning,
                        module='fontTools.ttLib.tables._p_o_s_t')

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


def can_render(font, character):
    for table in font['cmap'].tables:
        if ord(character) in table.cmap:
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


def generate_complementary_color(is_light):
    """Generate a color with good contrast to the input color."""
    # If is_light is True, generate a brighter color
    # Otherwise, generate a darker color
    if is_light:
        return [random.randint(128, 255) for _ in range(3)]

    return [random.randint(0, 127) for _ in range(3)]


def generate_color_triplet():
    color1 = [random.randint(0, 255) for _ in range(3)]
    luminance1 = calculate_luminance(color1)

    is_light2 = luminance1 <= 0.5
    color2 = generate_complementary_color(is_light2)

    is_light3 = not is_light2 if random.random() > 0.5 else is_light2
    color3 = generate_complementary_color(is_light3)

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
        ((0, 0, 0), (255, 255, 255), (211, 211, 211)),
        # Black text, white background, light blue box
        ((0, 0, 0), (255, 255, 255), (173, 216, 230)),
    ]

    colors = random.choice(predefined_color_triplets)
    return jitter_color_triplets(colors)


class TextlineSynthesisConfig:
    with open('tang_syn_config.yaml', 'r') as f:
        DEFAULT_CONFIG = yaml.safe_load(f)

    pygame.freetype.init()

    FONTS = []
    TTFONTS = []
    FALLBACK_FONT_IDS = []
    for file in tqdm(os.listdir("fonts")):
        if file.lower().endswith((".ttf", ".otf")):
            if DEFAULT_CONFIG["simplified_fonts_only"] and "FW" in file:
                continue

            pth = os.path.join("fonts", file)
            FONTS.append(pygame.freetype.Font(
                pth, DEFAULT_CONFIG["font_size"]))
            TTFONTS.append(TTFont(pth))

            if file in FALLBACK_FONT_NAMES:
                FALLBACK_FONT_IDS.append(len(FONTS) - 1)

    print(f"Usable font: {len(FONTS)}")

    def __init__(self, config={}):
        self.config = self.DEFAULT_CONFIG.copy()
        self.config["font_id"] = random.randint(0, len(self.FONTS) - 1)
        self.config["fonts"] = self.FONTS
        self.config["ttfonts"] = self.TTFONTS
        self.config["fallback_font_ids"] = self.FALLBACK_FONT_IDS

        self.config.update(config)

        self.config["canvas_height"] = self.config["height"] - \
            self.config["margin_top"] - self.config["margin_bottom"]

        if self.config["limit_font_size"]:
            self.config["font_size"] = min(
                self.config["canvas_height"], self.config["font_size"])

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

        font_size_jittor = random.random()
        if font_size_jittor < random_config["font_size_jittor_prob"]:
            random_config["font_size_jittor"] = True
        else:
            random_config["font_size_jittor"] = False

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

        random_config['graph_grid_size'] = np.random.randint(5, 15)
        random_config['chinese_grid_padding'] = np.random.randint(8, 12)

        # 40% of the time, apply elastic transform
        elastic_prob = random.random()
        if elastic_prob < random_config.get('elastic_transform_prob', 0.0):
            random_config['elastic_transform'] = True
        else:
            random_config['elastic_transform'] = False

        return cls(random_config)

    # Optional: add a method to modify configuration values
    def set_value(self, key, value):
        self.config[key] = value


if __name__ == "__main__":
    TextlineSynthesisConfig.random_config()
