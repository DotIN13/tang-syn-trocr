import os
import random
from datetime import datetime
from glob import iglob

import pytz
import numpy as np
import torch
import evaluate

# Override default transformers trainer_utils
import lib.trainer_utils

from transformers import VisionEncoderDecoderModel, AutoTokenizer, TrOCRProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
from transformers import get_scheduler, get_polynomial_decay_schedule_with_warmup

from lib.datasets import load_datasets
from lib.tang_syn_config import preload_fonts, load_default_config
from lib.datasets import list_text_files, load_texts

FULL_TRAINING = True
RESUME = False
MAX_LENGTH = 64

SHT = pytz.timezone("Asia/Shanghai")

torch.backends.cuda.matmul.allow_tf32 = True


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")


def load_model():
    model = None
    tokenizer = None

    processor = TrOCRProcessor.from_pretrained(
        "microsoft/trocr-base-handwritten", size=224, resample=3,
        image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

    if FULL_TRAINING:
        vision_hf_model = 'facebook/deit-small-distilled-patch16-224'
        # nlp_hf_model = "hfl/chinese-macbert-base"
        nlp_hf_model = "Langboat/mengzi-bert-L6-H768"

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
                 eval_dataset):

    class TangSynTrainer(Seq2SeqTrainer):
        def create_scheduler(self, num_training_steps, optimizer=None):
            """
            Setup the scheduler. The optimizer of the trainer must have been
            set up either before this method is called or passed as an argument.

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
        per_device_train_batch_size=180,
        per_device_eval_batch_size=100,
        # gradient_accumulation_steps=4,
        # gradient_checkpointing=True,
        num_train_epochs=1,
        bf16=True,
        learning_rate=5e-5,
        output_dir="./checkpoints",
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_steps=100,
        log_level="info",
        save_strategy="steps",
        save_total_limit=8,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=1000,
        resume_from_checkpoint="./checkpoints/",
        dataloader_num_workers=24,
        optim="adamw_torch",
        lr_scheduler_type="polynomial",
        # warmup_steps=1024,
        # weight_decay=1e-2,
        load_best_model_at_end=True,
        metric_for_best_model="hwdb_loss",
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

    # torch.set_num_threads(1)

    # torch.multiprocessing.set_start_method('forkserver')
    # torch.multiprocessing.set_forkserver_preload(
    #     ["os", "random", "torch", "torchvision", "pandas", "numpy", "cv2",
    #      "pygame", "pygame.freetype", "PIL", "scipy.ndimage", "kornia"])
    # torch.multiprocessing.set_forkserver_preload(["train", "lib.datasets", "lib.tang_syn_config", "lib.data_aug_v2", "lib.tang_syn"])

    # Load model
    model, processor, tokenizer = load_model()
    compute_metrics = build_metrics(tokenizer)

    # Load fonts
    default_config = load_default_config()
    fonts = preload_fonts(default_config)

    # Load texts
    text_files = list_text_files()
    texts = load_texts(text_files)

    # Load datasets
    train_dataset, eval_dataset = load_datasets(
        processor, tokenizer, fonts=fonts, texts=texts, default_config=default_config)

    trainer = init_trainer(
        model, tokenizer, compute_metrics, train_dataset, eval_dataset)

    try:
        result = trainer.train(resume_from_checkpoint=RESUME)
        print_summary(result)
    except Exception as err:
        save_checkpoint(trainer)
        raise err from err
