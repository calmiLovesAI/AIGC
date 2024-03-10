import torch
import numpy as np
import evaluate

from functools import partial
from PIL import Image
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from src_old.train.metrics import HUGGINGFACE_METRICS
from src_old.utils.file_ops import get_absolute_path
from typing import List
from datasets import load_dataset
from transformers import AutoImageProcessor, DefaultDataCollator, AutoModelForImageClassification, TrainingArguments, \
    Trainer
from src_old.data import HUGGINGFACE_DATASETS, CUSTOM_DATASETS
from src_old.models import HUGGINGFACE_MODELS


def compute_metrics_for_image_classification(eval_pred, metric_name):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    if metric_name in HUGGINGFACE_METRICS:
        accuracy = evaluate.load(metric_name)
        return accuracy.compute(predictions=predictions, references=labels)
    else:
        raise ValueError


def data_transform_for_image_classification(examples, _transforms):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples


def train_image_classification_model(model_id, dataset: str, output_dir, epochs, train_batch_size, eval_batch_size, learning_rate,
                                     first_n_data: int = 5000, test_size: float = 0.2,
                                     metric_name='accuracy', data_collator_type: str = 'default',
                                     device='cuda'):
    # process dataset
    label2id = dict()
    id2label = dict()
    dataset = dataset.lower()
    if dataset in HUGGINGFACE_DATASETS:
        train_set = load_dataset(dataset, split="train[:5000]")
        train_set = train_set.train_test_split(test_size=test_size)
        labels = train_set["train"].features["label"].names
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label
    elif dataset in CUSTOM_DATASETS:
        raise NotImplementedError(f"dataset: {dataset} not implemented.")
    else:
        raise ValueError(f"dataset: {dataset} not found.")
    if model_id in HUGGINGFACE_MODELS:
        image_processor = AutoImageProcessor.from_pretrained(model_id)

        normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        size = (
            image_processor.size["shortest_edge"]
            if "shortest_edge" in image_processor.size
            else (image_processor.size["height"], image_processor.size["width"])
        )
        _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

        train_set = train_set.with_transform(
            partial(data_transform_for_image_classification, _transforms=_transforms))

    if data_collator_type == 'default':
        data_collator = DefaultDataCollator()

    # build model
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # train
    output_dir = get_absolute_path(relative_path=output_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        metric_for_best_model=metric_name,
        push_to_hub=False,
    )
    train = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_set["train"],
        eval_dataset=train_set["test"],
        tokenizer=image_processor,
        compute_metrics=partial(compute_metrics_for_image_classification, metric_name=metric_name),
    )
    train.train()


def do_image_classification(checkpoint_path, image_path, device='cuda'):
    checkpoint_path = get_absolute_path(checkpoint_path)
    image = Image.open(image_path).convert('RGB')

    image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)
    inputs = image_processor(image, return_tensors="pt")

    model = AutoModelForImageClassification.from_pretrained(checkpoint_path)
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_label]
    print(f"The label of {image_path} is {predicted_label}.")
