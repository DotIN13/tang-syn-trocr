import os
from os import path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import evaluate

from PIL import Image

from transformers import VisionEncoderDecoderModel, AutoTokenizer, TrOCRProcessor, BasicTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator

from data_aug_v2 import build_data_aug
from torch.utils.data import Subset
from tang_syn import synthesize

FULL_TRAINING = True
MAX_LENGTH = 64


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def load_model():
    model = None
    tokenizer = None

    if FULL_TRAINING:
        vision_hf_model = 'microsoft/beit-base-patch16-384'
        nlp_hf_model = "uer/roberta-base-word-chinese-cluecorpussmall"

        # Reference: https://github.com/huggingface/transformers/issues/15823
        # initialize the encoder from a pretrained ViT and the decoder from a pretrained BERT model.
        # Note that the cross-attention layers will be randomly initialized, and need to be fine-tuned on a downstream dataset
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            vision_hf_model, nlp_hf_model)
        tokenizer = AutoTokenizer.from_pretrained(nlp_hf_model)
    else:
        trocr_model = 'models/epoch-1/'
        model = VisionEncoderDecoderModel.from_pretrained(trocr_model)
        tokenizer = AutoTokenizer.from_pretrained(trocr_model)

    if model:
        # set special tokens used for creating the decoder_input_ids from the labels
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size

        # set beam search parameters
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.max_length = MAX_LENGTH
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 6

    return model, tokenizer


class OCRDataset(Dataset):

    def __init__(self,
                 dataset_dir,
                 labels_dir,
                 transform,
                 processor,
                 tokenizer,
                 mode="train",
                 max_target_length=32,
                 device=None):
        self.dataset_dir = dataset_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.device = device
        self.processor = processor
        self.mode = mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.df = self.build_df()
        self.df_len = len(self.df)
        self.len = 12000000

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        # Thirty percent of the time, use existing dataset
        if idx < self.df_len:
            text = self.df['text'][idx]
            # get file name + text
            file_name = self.df["file_name"][idx]
            # prepare image (i.e. resize + normalize)
            image = Image.open(path.join(self.dataset_dir,
                                         file_name)).convert("RGB")
        # 70% percent of the time, use online generated data
        else:
            text_idx = int(idx / self.len * (self.df_len - 1))
            text = self.df['text'][text_idx]
            text, bgr_image = synthesize(text)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_image)

        if self.transform:
            image = self.transform(image)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.tokenizer(text,
                                padding="max_length",
                                stride=32,
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
        for root, _dirs, files in os.walk(self.labels_dir):
            for file in files:  # Loop through the dataset tsvfiles
                if not file.endswith(".tsv"):
                    continue

                print(f"Processing {file}")
                li.append(
                    pd.read_table(path.join(root, file),
                                  names=["file_name", "text"]))

        return pd.concat(li, axis=0, ignore_index=True)


def load_datasets(processor, tokenizer):
    dataset_dir = 'dataset/data'

    train_dataset = OCRDataset(dataset_dir=dataset_dir,
                               labels_dir="dataset/labels/train",
                               tokenizer=tokenizer,
                               processor=processor,
                               mode="train",
                               transform=build_data_aug(32, "train"),
                               max_target_length=MAX_LENGTH)

    # Define the number of samples to keep in eval dataset
    num_samples = 100

    eval_dataset = OCRDataset(dataset_dir=dataset_dir,
                              labels_dir="dataset/labels/test",
                              tokenizer=tokenizer,
                              processor=processor,
                              mode="eval",
                              transform=None,
                              max_target_length=MAX_LENGTH)

    # Create a random subset of the dataset
    subset_indices = torch.randperm(len(eval_dataset))[:num_samples]
    eval_dataset = Subset(eval_dataset, subset_indices.tolist())

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))

    return train_dataset, eval_dataset


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
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=28,
        per_device_eval_batch_size=28,
        num_train_epochs=3,
        fp16=True,
        learning_rate=4e-5,
        output_dir="./checkpoints",
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_total_limit=5,
        save_steps=1000,
        eval_steps=1000,
        resume_from_checkpoint="./checkpoints/",
        dataloader_num_workers=6,
        optim="adamw_torch")

    # instantiate trainer
    return Seq2SeqTrainer(
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
    torch.save(trainer.scaler.state_dict(),
               os.path.join(output_dir, SCALER_NAME))
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
    # processor = TrOCRProcessor.from_pretrained(
    #     "microsoft/trocr-base-handwritten")
    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten")
    model, tokenizer = load_model()
    compute_metrics = build_metrics(tokenizer)
    train_dataset, eval_dataset = load_datasets(processor, tokenizer)
    trainer = init_trainer(model, tokenizer, compute_metrics, train_dataset,
                           eval_dataset)
    try:
        result = trainer.train()
        print_summary(result)
    except Exception as err:
        save_checkpoint(trainer)
        raise err from err
