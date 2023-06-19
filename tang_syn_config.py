import os
import yaml
import random
import numpy as np

from tqdm import tqdm
import numpy as np
import pygame


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


def generate_random_config():
    config_dict = {}

    # 70% of the time, apply random colors
    color_prob = random.random()
    if color_prob < 0.7:  # 70% chance of random colors
        config_dict['random_colors'] = False
    else:  # 30% chance of predefined colors
        config_dict['random_colors'] = True

    # 20% of the time, apply graph grids
    # 30% of the time, apply chinese grids
    config_dict['graph_grid'] = False
    config_dict['chinese_grid'] = False
    grid_prob = random.random()
    if grid_prob < 0.2:  # 20% chance of graph grid
        config_dict['graph_grid'] = True
    elif grid_prob < 0.5:  # 30% chance of Chinese grid
        config_dict['chinese_grid'] = True

    elastic_prob = random.random()
    if elastic_prob > 0.5:
        config_dict['elastic_transform'] = True

    return config_dict


class TextlineSynthesisConfig:
    with open('tang_syn_config.yaml', 'r') as f:
        DEFAULT_CONFIG = yaml.safe_load(f)

    pygame.freetype.init()

    FONTS = []
    for file in tqdm(os.listdir("fonts")):
        if file.endswith((".ttf", ".otf")):
            if DEFAULT_CONFIG["simplified_fonts_only"] and "FW" in file:
                continue

            FONTS.append(pygame.freetype.Font(
                os.path.join("fonts", file), DEFAULT_CONFIG["font_size"]))

    def __init__(self, config={}):
        self.config = self.DEFAULT_CONFIG.copy()
        self.font = random.choice(self.FONTS)
        self.config.update(config)

    def __getattr__(self, item):
        if item in self.config:
            return self.config[item]

        raise AttributeError(f"'MyClass' object has no attribute '{item}'")

    @classmethod
    def random_config(cls, **kwargs):
        random_config = cls.DEFAULT_CONFIG.copy()

        colors = (generate_color_triplet()
                  if kwargs.get("random_colors", False) else pick_predefined_color_triplet())
        random_config['text_color'] = colors[0]
        random_config['bg_color'] = colors[1]
        random_config['box_color'] = colors[2]

        random_config['margin_left'] = np.random.randint(5, 15)
        random_config['margin_right'] = np.random.randint(5, 15)
        random_config['graph_grid'] = kwargs.get("graph_grid", False)
        random_config['graph_grid_size'] = np.random.randint(5, 15)
        random_config['chinese_grid'] = kwargs.get("chinese_grid", False)
        random_config['chinese_grid_padding'] = np.random.randint(8, 12)
        random_config['elastic_transform'] = kwargs.get("elastic_transform", False)
        return cls(random_config)

    # Optional: add a method to modify configuration values
    def set_value(self, key, value):
        if key == "font_size":
            raise KeyError(f"Modifying {key} is not allowed")

        self.config[key] = value


if __name__ == "__main__":
    config = generate_random_config()
    TextlineSynthesisConfig.random_config(**config)
