import os
import random
from datetime import datetime
from glob import iglob

import yaml
import cv2
import pytz
import numpy as np
import torch
import evaluate

# Override default transformers trainer_utils
import lib.trainer_utils

from transformers import VisionEncoderDecoderModel, AutoTokenizer, TrOCRProcessor
from transformers import Seq2SeqTrainingArguments
from transformers import default_data_collator
from transformers.utils.import_utils import is_torch_bf16_gpu_available

from lib.datasets import load_datasets
from lib.tang_syn_config import preload_fonts, load_default_config
from lib.datasets import list_text_files, load_texts
from lib.tang_syn_trainer import TangSynTrainer

FULL_TRAINING = True
RESUME = True
MAX_LENGTH = 64

torch.backends.cuda.matmul.allow_tf32 = True


def load_training_config(name):
    """Load training config from yaml file."""
    config = None
    with open(os.path.join("configs", f"{name}.yml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    training_args = config.get("training_args", {})

    checkpoints_dir = os.path.join(
        training_args.get("output_dir", "checkpoints"), name)
    logging_dir = os.path.join(training_args.get("logging_dir", "logs"), name)

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    existing_logs = list(iglob(f"{logging_dir}/*/"))

    if RESUME and len(existing_logs) > 0:
        logging_dir = max(existing_logs, key=os.path.getctime)
    else:
        SHT = pytz.timezone("Asia/Shanghai")
        logging_dir = f"{logging_dir}/{datetime.now().astimezone(SHT).strftime('%Y_%m_%d-%p%I_%M_%S')}"

    training_args["output_dir"] = checkpoints_dir
    training_args["resume_from_checkpoint"] = checkpoints_dir
    training_args["logging_dir"] = logging_dir

    return config


def load_model(training_config=None):
    model = None
    tokenizer = None

    processor_args = training_config.get("processor_args", {})

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten", **processor_args)

    if FULL_TRAINING:
        vision_hf_model = training_config.get(
            "encoder", 'facebook/deit-small-distilled-patch16-224')

        nlp_hf_model = training_config.get(
            "decoder", "Langboat/mengzi-bert-L6-H768")

        # Reference: https://github.com/huggingface/transformers/issues/15823
        # initialize the encoder from a pretrained ViT and the decoder from a pretrained BERT model.
        # Note that the cross-attention layers will be randomly initialized, and need to be fine-tuned on a downstream dataset
        model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            vision_hf_model, nlp_hf_model)

        new_words = ["“", "”", "‘", "’"]  # Add new words
        tokenizer = AutoTokenizer.from_pretrained(nlp_hf_model)
        tokenizer.add_tokens(new_tokens=new_words)
    else:
        trocr_model = 'models/tang-syn-5.0-online-epoch-1'
        model = VisionEncoderDecoderModel.from_pretrained(trocr_model)
        tokenizer = AutoTokenizer.from_pretrained(trocr_model)

    if model:
        # set special tokens used for creating the decoder_input_ids from the labels
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        new_embeddings = model.decoder.resize_token_embeddings(
            len(tokenizer), pad_to_multiple_of=8)

        new_vocab_size = new_embeddings.weight.shape[0]

        print(f"New vocab size: {new_vocab_size}")

        model.config.decoder.vocab_size = new_vocab_size
        model.decoder.config.vocab_size = new_vocab_size

        # set beam search parameters
        model.config.eos_token_id = tokenizer.sep_token_id
        model.config.max_length = MAX_LENGTH
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 4

    return model, processor, tokenizer


def build_metrics(tokenizer):
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_str = tokenizer.batch_decode(labels_ids,
                                            skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=labels_str)
        wer = wer_metric.compute(predictions=pred_str, references=labels_str)

        return {"cer": cer, "wer": wer}

    return compute_metrics


def init_trainer(model, tokenizer, compute_metrics, train_dataset,
                 eval_dataset, training_config=None, syn_config=None):

    bf16 = is_torch_bf16_gpu_available()
    fp16 = not bf16

    training_args = training_config.get("training_args", {})

    args = Seq2SeqTrainingArguments(
        bf16=bf16,
        fp16=fp16,
        **training_args
    )

    # instantiate trainer
    return TangSynTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        training_config=training_config,
        syn_config=syn_config
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


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")


if __name__ == "__main__":

    # torch.set_num_threads(1)

    # torch.multiprocessing.set_start_method('forkserver')
    # torch.multiprocessing.set_forkserver_preload(
    #     ["os", "random", "torch", "torchvision", "pandas", "numpy", "cv2",
    #      "pygame", "pygame.freetype", "PIL", "scipy.ndimage", "kornia"])
    # torch.multiprocessing.set_forkserver_preload(["train", "lib.datasets", "lib.tang_syn_config", "lib.data_aug_v2", "lib.tang_syn"])

    cv2.setNumThreads(4)

    training_config = load_training_config("deit-small-mengzi-l6-no-elastic")

    # Load model
    model, processor, tokenizer = load_model(training_config)
    compute_metrics = build_metrics(tokenizer)

    # Load fonts
    default_config = load_default_config("tang_syn_config-64-no-elastic")
    fonts = preload_fonts(default_config)

    # Load texts
    text_files = list_text_files()
    texts = load_texts(text_files)

    # Load datasets
    train_dataset, eval_dataset = load_datasets(
        processor, tokenizer, fonts=fonts, texts=texts, default_config=default_config)

    trainer = init_trainer(
        model, tokenizer, compute_metrics, train_dataset,
        eval_dataset, training_config=training_config, syn_config=default_config)

    try:
        result = trainer.train(resume_from_checkpoint=RESUME)
        print_summary(result)
    except Exception as err:
        save_checkpoint(trainer)
        raise err from err
