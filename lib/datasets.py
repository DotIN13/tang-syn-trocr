import os
from os import path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from PIL import Image

from .data_aug_v2 import build_data_aug
from .tang_syn import synthesize
from .tang_syn_config import TextlineSynthesisConfig, load_pygame_font

MAX_LENGTH = 64

def random_slice(s, min_length=1, max_length=MAX_LENGTH, tokenizer=None):
    assert (min_length <= max_length)

    res = tokenizer(s, return_offsets_mapping=True)
    tokens = res["input_ids"][1:-1]
    offsets = res["offset_mapping"][1:-1]

    length = random.randint(min_length, max_length)

    if len(tokens) <= length:
        return s

    start = random.randint(0, len(tokens) - length)
    end = start + length - 1

    start_index = offsets[start][0]
    end_index = offsets[end][-1]

    return s[start_index: end_index]


def candidate_slice(s, length=512):
    if len(s) <= length:
        return s

    start = random.randint(0, len(s) - length)
    return s[start: start + length - 1]


def list_text_files():
    """Load text indexes, return list of file paths along with their line counts"""

    index_dir = path.join("dataset-syn", "indexes")
    labels_dir = path.join("dataset-syn", "labels")

    file_paths = []

    for file in tqdm(os.listdir(index_dir)):
        if not file.endswith(".tsv"):
            continue

        # print(f"Reading {file}")

        with open(path.join(index_dir, file), "r", encoding="utf-8") as index_file:
            for line in index_file:
                line = line.strip()
                if line:
                    file, length = line.split("\t")
                    file_path = path.join(labels_dir, file)
                    file_paths.append((file_path, int(length)))

    return file_paths


def load_texts(text_files):
    """
    Iterate through file_handles, then for each text file,
    construct a list of textlines to sample from
    """

    print("Loading text files...")

    # is_distributed = torch.distributed.is_available() and \
    #     torch.distributed.is_initialized()

    # worker_info = torch.utils.data.get_worker_info()
    # if worker_info:
    #     num_workers = worker_info.num_workers

    #     world_size = 1
    #     if is_distributed:
    #         world_size = torch.distributed.get_world_size()

    #     choices_k = len(text_files) // (num_workers * world_size)
    #     choices_k = max(choices_k, 1)
    #     text_files = random.choices(text_files, k=choices_k)

    files = []

    for file_path, _handle_len in tqdm(text_files):
        with open(file_path, "r", encoding="utf-8") as f:
            df = pd.DataFrame(f.readlines())
            files.append(df)

    print(f"{len(files)} text files loaded.")

    return files


class OCRDataset(Dataset):

    def __init__(self,
                 dataset_dir,
                 labels_dir,
                 transform,
                 processor,
                 tokenizer,
                 mode="train",
                 max_target_length=MAX_LENGTH,
                 device=None,
                 text_files=None,
                 default_config=None,
                 fonts=None,
                 debug=False):

        self.dataset_dir = dataset_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.device = device
        self.processor = processor
        self.mode = mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.debug = debug

        if mode == "online":
            self.df = None
            self.df_len = 0
        elif mode in ("train", "eval"):
            self.df = self.build_df()
            self.df_len = len(self.df)

        if mode in ("train", "online"):
            self.default_config = default_config
            self.fonts = fonts
            self.texts = text_files
            # self.load_fonts()
            # self.load_texts()
            self.arbitrary_len = 36000000

    def __len__(self):
        return self.arbitrary_len

    def __getitem__(self, idx):

        image, text = self.get_image_and_text(idx)

        if self.debug:
            return image, text

        pixel_values = self.processor(image, return_tensors="pt")

        labels = self.tokenizer(text,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_target_length,
                                return_tensors="pt")

        encoding = {
            "pixel_values": pixel_values.pixel_values.squeeze(),
            "labels": labels.input_ids.squeeze(),
            "decoder_attention_mask": labels.attention_mask.squeeze(),
        }
        return encoding

    def get_image_and_text(self, idx):

        if idx < self.df_len:
            text = self.df['text'][idx]
            # get file name + text
            file_name = self.df["file_name"][idx]
            # prepare image (i.e. resize + normalize)
            image = Image.open(path.join(self.dataset_dir,
                                         file_name)).convert("RGB")

        else:
            text, bgr_image = self.sample_and_synthesize()
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_image)

        if self.transform:
            image = self.transform(image)

        # Remove spaces from text, as only data from tang-syn 1.0 had spaces before text
        text = text.strip()
        # print(text) # Debug

        return image, text

    def build_df(self):
        li = []
        for file in tqdm(os.listdir(self.labels_dir)):
            if not file.endswith(".tsv"):
                continue

            # print(f"Reading {file}")

            li.append(
                pd.read_table(path.join(self.labels_dir, file),
                              names=["file_name", "text"]))

        return pd.concat(li, axis=0, ignore_index=True)

    def load_fonts(self):
        print("Loading pygame fonts...")

        for font in self.fonts["fonts"]:
            font[0] = load_pygame_font(
                font[0], self.default_config["font_size"])

        for font in self.fonts["fallback_fonts"]:
            font[0] = load_pygame_font(
                font[0], self.default_config["font_size"])

        print("Pygame fonts loaded.")

    def sample_line_from_texts(self):
        """Sample line from file handles"""
        file_df = random.choice(self.texts)
        line = file_df.sample(n=1).iloc[0, 0]
        line = line.strip()

        if not (isinstance(line, str) and len(line) > 0):
            raise ValueError("Empty line.")

        return line

    def sample_and_synthesize(self):
        try:
            if os.environ.get("DEBUG") in ["1", "all", "text"]:
                text = "test"
            else:
                text = self.sample_line_from_texts()

            text = candidate_slice(text, length=512)
            text = random_slice(
                text, max_length=self.max_target_length, tokenizer=self.tokenizer)

            syn_conf = TextlineSynthesisConfig.random_config(
                default_config=self.default_config, **self.fonts)

            if os.environ.get("DEBUG") in ["1", "all", "syn"]:
                bgr_image = np.random.randint(
                    0, 256, (64, 1024, 3), dtype=np.uint8)
            else:
                bgr_image = synthesize(text, syn_conf=syn_conf)

            return text, bgr_image

        except ValueError as sample_err:
            print(sample_err)
            return self.sample_and_synthesize()


class EvalDataset(OCRDataset):

    def __len__(self):
        return len(self.df)


def load_datasets(processor, tokenizer, fonts=None, texts=None, default_config=None, debug=False):
    """Load train, eval datasets."""

    dataset_dir = 'dataset/data'

    print("Initializing training dataset.")

    train_dataset = OCRDataset(dataset_dir=dataset_dir,
                               labels_dir="dataset/labels/train",
                               tokenizer=tokenizer,
                               processor=processor,
                               mode="online",
                               transform=build_data_aug(
                                   height=64, mode="train", resizepad=False),
                               max_target_length=MAX_LENGTH,
                               text_files=texts,
                               default_config=default_config,
                               fonts=fonts,
                               debug=debug)

    print("Initializing eval datasets.")

    eval_dataset = EvalDataset(dataset_dir=dataset_dir,
                               labels_dir="dataset/labels/test-ic13",
                               tokenizer=tokenizer,
                               processor=processor,
                               mode="eval",
                               transform=build_data_aug(
                                   height=64, mode="eval", resizepad=False),
                               max_target_length=MAX_LENGTH)

    niandai_dataset = EvalDataset(dataset_dir=dataset_dir,
                                  labels_dir="dataset/labels/test-niandai",
                                  tokenizer=tokenizer,
                                  processor=processor,
                                  mode="eval",
                                  transform=build_data_aug(
                                      height=64, mode="eval", resizepad=False),
                                  max_target_length=MAX_LENGTH)

    # Define the number of samples to keep in eval dataset
    # Create a random subset of the dataset
    num_samples = 500
    subset_indices = torch.randperm(len(eval_dataset))[:num_samples]
    eval_dataset = Subset(eval_dataset, subset_indices.tolist())

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(
        eval_dataset), len(niandai_dataset))

    return train_dataset, {"hwdb": eval_dataset, "niandai": niandai_dataset}


if __name__ == "__main__":
    from transformers import AutoTokenizer, TrOCRProcessor

    trocr_model = 'models/tang-syn-3.0-epoch-1'
    tokenizer = AutoTokenizer.from_pretrained(trocr_model)

    processor = TrOCRProcessor.from_pretrained(trocr_model)

    train_dataset, eval_dataset = load_datasets(processor, tokenizer)
