from typing import Iterable, List

import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from nlp_pipeline.utils.utils import get_progress


# These functions are taken from transformers repo
def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def freeze_params(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model: nn.Module):
    """
    Freeze token embeddings and positional embeddings for bart, just token embeddings for T5.
    """
    try:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    except AttributeError:
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"None of {npars} weights require grad"


def extract_features(
    dataloader: DataLoader,
    model: nn.Module,
    verbose=True,
    device="cuda",
    use_aggre: bool = True,
) -> np.array:
    temp_lst = []
    model = model.encoder
    model = model.to(device)
    for _, batch in enumerate(get_progress(iterable=dataloader, disable=verbose)):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        features = model(input_ids, attention_mask)
        features = features.last_hidden_state
        # features = features.mean(axis=1) if use_aggre else features
        temp_lst.append(features.detach().cpu().numpy())

    return np.concatenate(temp_lst)
