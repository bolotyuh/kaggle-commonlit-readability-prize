from typing import Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, Value, load_dataset
from transformers import AutoTokenizer, TrainingArguments

from src.configs import DataTrainingArguments
from src.utils import logger


def setup_dataset(data_args: DataTrainingArguments, cache_dir: str):
    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
    raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=cache_dir)

    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    # Regression problem
    label_list = None
    num_labels = 1

    # regression requires float as label type, let's cast it if needed
    for split in raw_datasets.keys():
        if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
            logger.warning(f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}")
            features = raw_datasets[split].features
            features.update({"label": Value("float32")})
            try:
                raw_datasets[split] = raw_datasets[split].cast(features)
            except TypeError as error:
                logger.error(
                    f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                )
                raise error

    return raw_datasets, num_labels, label_list


def process_dataset(
    data_args: DataTrainingArguments,
    tokenizer: AutoTokenizer,
    training_args: TrainingArguments,
    raw_datasets: Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset],
):
    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")
            # join together text columns into "sentence" column
            examples["sentence"] = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                for i in range(len(examples[column])):
                    examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]

        # Tokenize the texts
        result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)

        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = raw_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    return train_dataset, eval_dataset
