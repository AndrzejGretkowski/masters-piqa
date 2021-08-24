import argparse
from typing import Optional
from warnings import warn

import torch
from torch import nn as nn
from torch import optim as optim
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from piqa.data import PiqaDataset
from piqa.model.roberta_for_classification import RobertaPIQA
from piqa.model.roberta_tokenizer import RobertaPIQATokenizer

import pytorch_lightning as pl


def main(
    model_type: str,
    model_path: Optional[str],
    save_path: Optional[str],
    gpus: int,
    batch_size: int,
    learning_rate: float,
    fix_valid_set: bool,
    evaluation_only: bool,
    notebook: bool,
):
    # Device
    if gpus > 0 and not torch.cuda.is_available():
        gpus = 0
        warn('GPU is not available on this machine. Using CPU instead.')
    device = torch.device('cuda') if gpus > 0 else torch.device('cpu')

    # TQDM
    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    # Training data
    train_set = PiqaDataset("train", fix=fix_valid_set)
    valid_set = PiqaDataset("valid", fix=fix_valid_set)
    test_set = PiqaDataset("test", fix=fix_valid_set)

    # Model & Tokenizer
    if model_type == "roberta":
        model = RobertaPIQA()
        tokenizer = RobertaPIQATokenizer.from_pretrained("roberta-large")
    elif model_type == "alberta":
        raise NotImplementedError("Alberta model has not been implemented yet.")
    else:
        raise RuntimeError(f"{model_type} is not supported.")

    # Load finetuned weights
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    # Just evaluation
    if evaluation_only:
        model.eval()
    else:
        model.train()
    model.to(device)

    # Pre-tokenize data sets
    collate_fn = lambda x: tokenizer.collate_fn(x, pad_token=tokenizer.pad_token_id)
    train_set = tokenizer.tokenize_data_set(train_set)
    trainloader = DataLoader(train_set, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    test_set = tokenizer.tokenize_data_set(test_set)
    testloader = DataLoader(test_set, shuffle=False, collate_fn=collate_fn)
    valid_set = tokenizer.tokenize_data_set(valid_set)
    validloader = DataLoader(valid_set, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

    # Training
    trainer = pl.Trainer(gpus=gpus)
    trainer.fit(model, trainloader, validloader)

    print("Finished Training")


def parse_args():
    parser = argparse.ArgumentParser(description="PIQA training commandline")

    parser.add_argument(
        "-fix",
        "--fix-valid-set",
        action="store_true",
        help="if valid test should be patched",
    )
    parser.add_argument(
        "-m",
        "--model-type",
        type=str,
        choices=["roberta", "alberta"],
        default="roberta",
        help="type of model to use",
    )
    parser.add_argument(
        "-p", "--model-path", type=str, default=None, help="Path to the saved model"
    )
    parser.add_argument(
        "-s", "--save-path", type=str, default=None, help="Where to save the model"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Just evaluate the test set"
    )
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size.")
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of gpus used.",
    )
    parser.add_argument(
        "--notebook", action="store_true", help="Use notebook progress bar"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help='Learning rate.'
    )

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        model_type=args.model_type,
        model_path=args.model_path,
        save_path=args.save_path,
        fix_valid_set=args.fix_valid_set,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_only=args.eval_only,
        gpus=args.gpus,
        notebook=args.notebook,
    )
