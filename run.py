import argparse
import pickle
from pathlib import Path
from typing import Optional
from warnings import warn

import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader

from piqa.data import PiqaDataset
from piqa.model.models import PIQAModel
from piqa.model.tokenizers import PIQATokenizer
from piqa.model.conceptnet_base import AffordanceType

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, GPUStatsMonitor, ModelCheckpoint


def main(
    experiment_type: str,
    affordance_type: AffordanceType,
    ngrams: int,
    return_words: int,
    definition_length: int,
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

    # Training data
    train_set = PiqaDataset("train", fix=fix_valid_set)
    valid_set = PiqaDataset("valid", fix=fix_valid_set)
    test_set = PiqaDataset("test", fix=fix_valid_set)

    # Model & Tokenizer
    try:
        model = PIQAModel.get(model_type)(learning_rate=learning_rate, model_type=model_type)
        tokenizer = PIQATokenizer.get(model_type)(
            experiment_type, ngrams, return_words, definition_length, affordance_type, model_type, tqdm_arg=notebook)
    except TypeError:
        raise RuntimeError(f'{model_type} has not been implemented.')

    # Load finetuned weights
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    # Just evaluation
    if evaluation_only:
        model.eval()
    else:
        model.train()

    # Pre-tokenize data sets
    collate_fn = lambda x: tokenizer.collate_fn(x, pad_token=tokenizer.pad_token_id)
    if tokenizer._type == 'affordance':
        all_sets_path = Path(f'./data/{tokenizer._type}_{ngrams}_{return_words}_{definition_length}_{affordance_type}.pkl')
    if tokenizer._type == 'definition':
        all_sets_path = Path(f'./data/{tokenizer._type}_{ngrams}_{return_words}_{definition_length}.pkl')
    else:
        all_sets_path = Path(f'./data/{tokenizer._type}.pkl')

    if all_sets_path.exists():
        with open(all_sets_path, 'rb') as f:
            all_sets = pickle.load(f)
        train_set = all_sets['train']
        test_set = all_sets['test']
        valid_set = all_sets['valid']
    else:
        train_set = tokenizer.pretokenize_data_set(train_set)
        valid_set = tokenizer.pretokenize_data_set(valid_set)
        test_set = tokenizer.pretokenize_data_set(test_set)
        with open(all_sets_path, 'wb') as f:
            pickle.dump({'train': train_set, 'test': test_set, 'valid': valid_set}, f)

    valid_set = tokenizer.tokenize_data_set(valid_set)
    test_set = tokenizer.tokenize_data_set(test_set)
    train_set = tokenizer.tokenize_data_set(train_set)
    trainloader = DataLoader(train_set, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    validloader = DataLoader(valid_set, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)
    testloader = DataLoader(test_set, shuffle=False, collate_fn=collate_fn)

    # Load callbacks
    callbacks = []
    callbacks.append(EarlyStopping('val_accuracy', min_delta=0.001, patience=5, mode='max', verbose=True))
    if save_path is not None:
        callbacks.append(ModelCheckpoint(save_path, filename='{epoch}-{val_loss:.2f}'))
    if gpus > 0:
        callbacks.append(GPUStatsMonitor(True, False, False, False, False, False))

    # Training
    trainer = pl.Trainer(gpus=gpus, auto_scale_batch_size=False, callbacks=callbacks)
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
        "-t",
        "--experiment-type",
        type=str,
        choices=["baseline", "definition", "affordance"],
        default="baseline",
        help="What type of experiment to run."
    )
    parser.add_argument(
        "-ngram",
        "--ngram-amount",
        type=int,
        default=1,
        help="Amount of ngrams in extraction."
    )
    parser.add_argument(
        "-words",
        "--return-words",
        type=int,
        default=7,
        help="Amount of words returned in extraction."
    )
    parser.add_argument(
        "-deflength",
        "--definition-length",
        type=int,
        default=25,
        help="Amount of words returned in definitions."
    )
    parser.add_argument(
        "-afftype",
        "--affordance-type",
        type=lambda type: AffordanceType[type],
        choices=list(AffordanceType),
        default=AffordanceType.STANDALONE.name,
        help="Type of affordance to use."
    )
    parser.add_argument(
        "-m",
        "--model-type",
        type=str,
        choices=list(PIQAModel.model_mapping.keys()),
        default="roberta-base",
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
        experiment_type=args.experiment_type,
        affordance_type=args.affordance_type,
        ngrams=args.ngram_amount,
        return_words=args.return_words,
        definition_length=args.definition_length,
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
