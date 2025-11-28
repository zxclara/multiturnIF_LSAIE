### Modified based on torchtune/datasets/_alpaca.py, and
###   torchtune/datasets/_sft.py
###   commit id 0a08c2f5f25de42404948a66ae7806f0f0a2c485

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional, Union

from torchtune.data._messages import IFToMessages

from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer

def if_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "allenai/tulu-3-sft-personas-instruction-following",
    train_on_input: bool = True,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Support for family of Tulu3 sft-personas-if-style datasets from Hugging Face Datasets using the `data input 
    format <https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following#dataset-structure>`, 
    where ``messages`` and ``constraints`` are fields from the dataset.

    Masking of the prompt during training is controlled by the ``train_on_input`` flag, which is
    set to ``True`` by `default`_
    - If ``train_on_input`` is True, the prompt is used during training and
    contributes to the loss.
    - If ``train_on_input`` is False, the prompt is masked out (tokens replaced with -100)

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See `Hugging Face's
            <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path>`_
            ``load_dataset`` for more details. Default is ``allenai/tulu-3-sft-personas-instruction-following``.
        train_on_input (bool): Whether the model is trained on the prompt or not. Default is True.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        **load_dataset_kwargs (dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.

    Returns:
        Union[SFTDataset, PackedDataset]: dataset configured with source data and transform

    Raises:
        ValueError: If ``packed`` is True and ``max_seq_len`` is not set on the tokenizer.

    Example:
        >>> if_ds = if_dataset(tokenizer=tokenizer)
        >>> for batch in Dataloader(if_ds, batch_size=8):
        >>>     print(f"Batch size: {len(batch)}")
        >>> Batch size: 8
    """

    message_transform = IFToMessages(
        train_on_input=train_on_input
    )
    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds

