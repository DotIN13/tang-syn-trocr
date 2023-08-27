import os
import random
import warnings
from multiprocessing import get_context, cpu_count

import yaml
import numpy as np
import pygame
import pygame.freetype
from fontTools.ttLib import TTFont

warnings.filterwarnings('ignore', category=UserWarning,
                        module='fontTools.ttLib.tables')

pygame.freetype.init()

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
    """Check if a character can be rendered by a font."""
    for cmap in cmaps:
        if ord(character) in cmap:
            return True

    return False


def apply_color_jitter(color, std=15):
    """Apply color jitter to a RGB color."""
    jittered_color = [max(
        0, min(255, c + int(np.random.normal(c, std)))) for c in color]
    return tuple(jittered_color)


def jitter_color_triplets(triplet):
    """Apply color jitter to a color triplet consisting of three RGB colors."""
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
    """Generate a color triplet consisting of three RGB colors."""
    color1 = [random.randint(0, 255) for _ in range(3)]
    text_luminance = calculate_luminance(color1)

    color2 = generate_complementary_color(text_luminance)
    color3 = generate_complementary_color(text_luminance)

    # text, bg, box
    return color1, color2, color3


def pick_predefined_color_triplet():
    """Randomly pick a predefined color triplet consisting of three RGB colors."""
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


def load_pygame_font(pth, size):
    """Creates a Pygame font object from a given font file path and size.

    Args:
        pth (str): The file path of the font file.
        size (int): The size of the font.

    Returns:
        A Pygame font object.
    """
    return pygame.freetype.Font(pth, size)


def ttfont_cmaps(pth):
    """Given a font file path, returns a list of its character maps.

    Args:
        pth (str): The file path of the font file.

    Returns:
        A list of character maps for the font.
    """
    ttfont = TTFont(pth)
    return [set(table.cmap.keys()) for table in ttfont["cmap"].tables]


def valid_font(filename, config):
    """Check if a font file is valid.

    Args:
        filename (str): The file name of the font file.
        config (dict): The configuration dictionary.

    Returns:
        True if the font file is valid, False otherwise.
    """

    if not filename.lower().endswith((".ttf", ".otf")):
        return False

    if config["simplified_fonts_only"] and "FW" in filename:
        return False

    return True


def process_font(file):
    """Loads a font from a font file path.

    Args:
        file (str): The file path of the font file.

    Returns:
        A tuple containing a Pygame font object and a list of character maps.
    """

    pth = os.path.join("fonts", file)
    return [pth, ttfont_cmaps(pth)]


def complete_font(font, font_size):
    """Given a font tuple, returns a font tuple with a Pygame font object and a list of character maps.

    Args:
        font (tuple): A tuple containing a font file path and a list of character maps.
        config (dict): The configuration dictionary.

    Returns:
        A tuple containing a Pygame font object and a list of character maps.
    """

    pth = font[0]

    if not isinstance(pth, str):
        return font

    font[0] = load_pygame_font(pth, font_size)
    return font


def load_fonts(font_paths, config):
    """Loads fonts from a list of font file paths.

    Args:
        font_paths (list): A list of font file paths.
        config (dict): The configuration dictionary.

    Returns:
        A list of tuples, each containing a Pygame font object and a list of character maps.
    """

    font_paths = list(filter(lambda pth: valid_font(pth, config), font_paths))

    # fonts = [process_font(f, config, is_fallback) for f in tqdm(font_paths)]
    # return list(filter(lambda x: x is not None, fonts))

    processes = min(cpu_count(), 16)
    processes = min(len(font_paths) // 4, processes)

    with get_context("spawn").Pool(processes=processes) as pool:
        results = pool.map(process_font, font_paths)

    if config["preload_all_fonts"]:
        results = [complete_font(font, config["font_size"])
                   for font in results]

    return results


def preload_fonts(config):
    debug_fonts = os.environ.get("DEBUG") in ["1", "all", "fonts"]

    def load_fallback_fonts():
        return load_fonts(FALLBACK_FONT_NAMES, config)

    fallback_fonts = load_fallback_fonts()

    print("Fallback font cmaps loaded.")

    if debug_fonts:
        fonts = load_fallback_fonts()
    else:
        fonts = load_fonts(os.listdir("fonts"), config)

    print(f"All font cmaps loaded: {len(fonts)}")

    return {"fonts": fonts, "fallback_fonts": fallback_fonts}


def load_default_config(name=None):

    name = name or "tang_syn_config-64"

    with open(f'{name}.yml', 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


class TextlineSynthesisConfig:

    def __init__(self, config=None, default_config=None, fonts=None, fallback_fonts=None):

        if default_config is not None:
            self.config = default_config.copy()
        else:
            self.config = load_default_config()

        if config is not None:
            self.config.update(config)

        self.config["canvas_height"] = self.config["height"] - \
            self.config["margin_top"] - self.config["margin_bottom"]

        if self.config["limit_font_size"]:
            self.config["font_size"] = min(
                self.config["canvas_height"], self.config["font_size"])

        self.config["fallback_fonts"] = fallback_fonts

        # Choose the main font from the list
        font = random.choice(fonts)
        complete_font(font, self.config["font_size"])

        self.config["font"] = font

    def __getattr__(self, item):
        if item in self.config:
            return self.config[item]

        raise AttributeError(f"Object has no attribute '{item}'")

    @classmethod
    def random_config(cls, default_config=None, **kwargs):

        if default_config is not None:
            config = default_config.copy()
        else:
            config = load_default_config()

        random_font_size = config.get("random_font_size", False)
        if random_font_size:
            config["font_size"] = random.randint(
                config["min_font_size"], config["max_font_size"])

        config["font_size_jittor"] = (
            random.random() < config["font_size_jittor_prob"])

        config["font_weight_jittor"] = (
            random.random() < config["font_weight_jittor_prob"])

        random_margin = config.get("random_margin", False)
        if random_margin is not None:
            config['margin_left'] *= random.random()
            config['margin_right'] *= random.random()
            config['margin_top'] *= random.random()
            config['margin_bottom'] *= random.random()

        # 30% of the time, apply random colors
        color_prob = random.random()
        if color_prob < config.get("random_color_prob", 0.0):
            colors = generate_color_triplet()
        else:
            colors = pick_predefined_color_triplet()

        config['text_color'] = colors[0]
        config['bg_color'] = colors[1]
        config['box_color'] = colors[2]

        # 20% of the time, apply graph grids
        # 30% of the time, apply chinese grids
        config['graph_grid'] = False
        config['chinese_grid'] = False
        grid_prob = random.random()
        if grid_prob < 0.1:  # 10% chance of graph grid
            config['graph_grid'] = True
        elif grid_prob < 0.3:  # 20% chance of Chinese grid
            config['chinese_grid'] = True

        config['graph_grid_size'] = np.random.randint(
            config["graph_grid_min_size"], config["graph_grid_max_size"])

        config['chinese_grid_padding'] = np.random.randint(
            config['chinese_grid_min_padding'], config['chinese_grid_max_padding'])

        # 40% of the time, apply elastic transform
        elastic_prob = random.random()
        config['elastic_transform'] = elastic_prob < config.get(
            'elastic_transform_prob', 0.0)

        config["random_crossout"] = (
            random.random() < config["random_crossout_prob"])
        if config["random_crossout"]:
            config["random_crossout_rotation"] = np.random.uniform(
                -60, 60)
            config["random_crossout_thickness"] = random.randint(2, 3)

        config["mask_with_ellipses"] = (
            random.random() < config["mask_with_ellipses_prob"])

        return cls(config, default_config, **kwargs)

    def set_value(self, key, value):
        """Set a value in the configuration dictionary."""
        self.config[key] = value


if __name__ == "__main__":
    default_config = load_default_config()
    TextlineSynthesisConfig.random_config(
        default_config=default_config, **preload_fonts(default_config))
