from datetime import datetime
import os
from os import path
from glob import iglob
import random

import pytz
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import evaluate

from tqdm import tqdm

from PIL import Image

from transformers import VisionEncoderDecoderModel, AutoTokenizer, TrOCRProcessor, AutoImageProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from transformers import get_scheduler, get_polynomial_decay_schedule_with_warmup

from data_aug_v2 import build_data_aug
from torch.utils.data import Subset
from tang_syn import synthesize

FULL_TRAINING = False
RESUME = False
MAX_LENGTH = 64

SHT = pytz.timezone("Asia/Shanghai")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")


def load_model():
    model = None
    tokenizer = None

    processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten", size=384, resample=2,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

    if FULL_TRAINING:
        vision_hf_model = 'facebook/deit-base-distilled-patch16-384'
        nlp_hf_model = "hfl/chinese-macbert-base"
        # nlp_hf_model = "Langboat/mengzi-bert-L6-H768"

        # Reference: https://github.com/huggingface/transformers/issues/15823
        # initialize the encoder from a pretrained ViT and the decoder from a pretrained BERT model.
        # Note that the cross-attention layers will be randomly initialized, and need to be fine-tuned on a downstream dataset
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            vision_hf_model, nlp_hf_model)

        new_words = ["“", "”", "‘", "’"]  # Add new words
        tokenizer = AutoTokenizer.from_pretrained(nlp_hf_model)
        tokenizer.add_tokens(new_tokens=new_words)
    else:
        trocr_model = 'models/tang-syn-online-epoch-1'
        model = VisionEncoderDecoderModel.from_pretrained(trocr_model)
        tokenizer = AutoTokenizer.from_pretrained(trocr_model)

    if model:
        # set special tokens used for creating the decoder_input_ids from the labels
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        model.decoder.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = model.config.decoder.vocab_size

        # set beam search parameters
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.max_length = MAX_LENGTH
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 4

    return model, processor, tokenizer


def random_slice(s, min_length=1, max_length=MAX_LENGTH):
    length = random.randint(min_length, max_length)

    if len(s) <= length:
        return s

    start = random.randint(0, len(s) - length)
    return s[start: start + length]


class OCRDataset(Dataset):

    def __init__(self,
                 dataset_dir,
                 labels_dir,
                 transform,
                 processor,
                 tokenizer,
                 mode="train",
                 max_target_length=MAX_LENGTH,
                 device=None):
        self.dataset_dir = dataset_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.device = device
        self.processor = processor
        self.mode = mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer

        if mode == "online":
            self.df = None
            self.df_len = 0
        else:
            self.df = self.build_df()
            self.df_len = len(self.df)

        if mode == "train" or mode == "online":
            self.file_handles = self.load_texts()
            self.arbitrary_len = 1920000

    def __len__(self):
        return self.arbitrary_len

    def __getitem__(self, idx):

        # Thirty percent of the time, use existing dataset
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

        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Remove spaces from text, as only data from tang-syn 1.0 had spaces before text
        text = text.strip()

        labels = self.tokenizer(text,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_target_length).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [
            label if label != self.tokenizer.pad_token_id else -100
            for label in labels
        ]

        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }
        return encoding

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

    def load_texts(self):
        """Load text files"""
        index_dir = path.join("dataset_syn", "indexes")
        labels_dir = path.join("dataset_syn", "labels")

        file_handles = []

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
                        file_handles.append((file_path, int(length)))

        return file_handles

    def sample_line_from_texts(self):
        """Sample line from file handles"""
        file_path, handle_len = random.choice(self.file_handles)
        line_index = random.randint(0, handle_len - 1)

        # print(file_path, line_index)
        line = None

        with open(file_path, "r", encoding="utf-8") as handle:
            for i, lne in enumerate(handle):
                if i != line_index:
                    continue

                line = lne.strip()

        if not (isinstance(line, str) and len(line) > 0):
            raise ValueError(f"Empty line at line {line_index} of {file_path}")

        # print(
        #     f"Empty line at line {line_index} of {file_path}, retrying.")
        return line

    def sample_and_synthesize(self):
        try:
            text = self.sample_line_from_texts()
            text = random_slice(text, max_length=self.max_target_length)
            bgr_image = synthesize(text)
            return text, bgr_image
        except ValueError as sample_err:
            print(sample_err)
            return self.sample_and_synthesize()


class EvalDataset(OCRDataset):

    def __len__(self):
        return len(self.df)


def load_datasets(processor, tokenizer):
    dataset_dir = 'dataset/data'

    train_dataset = OCRDataset(dataset_dir=dataset_dir,
                               labels_dir="dataset/labels/train",
                               tokenizer=tokenizer,
                               processor=processor,
                               mode="train",
                               transform=build_data_aug(64, "train"),
                               max_target_length=MAX_LENGTH)

    # Define the number of samples to keep in eval dataset

    eval_dataset = EvalDataset(dataset_dir=dataset_dir,
                               labels_dir="dataset/labels/test-ic13",
                               tokenizer=tokenizer,
                               processor=processor,
                               mode="eval",
                               transform=None,
                               max_target_length=MAX_LENGTH)

    niandai_dataset = EvalDataset(dataset_dir=dataset_dir,
                                  labels_dir="dataset/labels/test-niandai",
                                  tokenizer=tokenizer,
                                  processor=processor,
                                  mode="eval",
                                  transform=None,
                                  max_target_length=MAX_LENGTH)

    # Create a random subset of the dataset
    num_samples = 500
    subset_indices = torch.randperm(len(eval_dataset))[:num_samples]
    eval_dataset = Subset(eval_dataset, subset_indices.tolist())

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(
        eval_dataset), len(niandai_dataset))

    return train_dataset, {"hwdb": eval_dataset, "niandai": niandai_dataset}


def build_metrics(tokenizer):
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        labels_str = tokenizer.batch_decode(labels_ids,
                                            skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=labels_str)
        wer = wer_metric.compute(predictions=pred_str, references=labels_str)

        return {"cer": cer, "wer": wer}

    return compute_metrics


def init_trainer(model, tokenizer, compute_metrics, train_dataset,
                 eval_dataset):

    class TangSynTrainer(Seq2SeqTrainer):
        def create_scheduler(self, num_training_steps, optimizer=None):
            """
            Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
            passed as an argument.

            Args:
                num_training_steps (int): The number of training steps to do.
            """
            if self.lr_scheduler is None:

                if self.args.lr_scheduler_type == "polynomial":
                    print("Using custom polynomial scheduler.")
                    self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                        optimizer=self.optimizer if optimizer is None else optimizer,
                        num_warmup_steps=self.args.get_warmup_steps(
                            num_training_steps),
                        num_training_steps=num_training_steps,
                        power=1.0,
                        lr_end=5e-6
                    )
                else:
                    self.lr_scheduler = get_scheduler(
                        self.args.lr_scheduler_type,
                        optimizer=self.optimizer if optimizer is None else optimizer,
                        num_warmup_steps=self.args.get_warmup_steps(
                            num_training_steps),
                        num_training_steps=num_training_steps,
                    )
            return self.lr_scheduler

    logging_dir = None

    if RESUME:
        logging_dir = max(iglob("./logs/*/"), key=os.path.getctime)
    else:
        logging_dir = f"./logs/{datetime.now().astimezone(SHT).strftime('%Y_%m_%d-%p%I_%M_%S')}"

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=48,
        gradient_accumulation_steps=8,
        # gradient_checkpointing=True,
        num_train_epochs=1,
        fp16=True,
        learning_rate=3e-5,
        output_dir="./checkpoints",
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=1,
        log_level="info",
        save_strategy="steps",
        save_total_limit=8,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        resume_from_checkpoint="./checkpoints/",
        dataloader_num_workers=4,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=768,
        weight_decay=1e-4,
        load_best_model_at_end=True,
        metric_for_best_model="hwdb_cer",
        greater_is_better=False,
        dataloader_pin_memory=True,
    )

    # instantiate trainer
    return TangSynTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )


def save_checkpoint(trainer):
    SCALER_NAME = "scaler.pt"
    SCHEDULER_NAME = "scheduler.pt"
    OPTIMIZER_NAME = "optimizer.pt"

    output_dir = "./models/temp"

    trainer.save_model(output_dir=output_dir)
    trainer.save_state()

    torch.save(trainer.scaler.state_dict(),
               os.path.join(output_dir, SCALER_NAME))
    torch.save(trainer.optimizer.state_dict(),
               os.path.join(output_dir, OPTIMIZER_NAME))
    torch.save(trainer.lr_scheduler.state_dict(),
               os.path.join(output_dir, SCHEDULER_NAME))
    # Single gpu
    rng_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.cuda.random.get_rng_state_all()
    }
    torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))


if __name__ == "__main__":
    model, processor, tokenizer = load_model()
    compute_metrics = build_metrics(tokenizer)
    train_dataset, eval_dataset = load_datasets(processor, tokenizer)
    trainer = init_trainer(model, tokenizer, compute_metrics, train_dataset,
                           eval_dataset)
    try:
        result = trainer.train(resume_from_checkpoint=RESUME)
        print_summary(result)
    except Exception as err:
        save_checkpoint(trainer)
        raise err from err
