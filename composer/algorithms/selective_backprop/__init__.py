# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""`Selective Backprop <https://arxiv.org/abs/1910.00762>`_ prunes minibatches according to the difficulty of the
individual training examples, and only computes weight gradients over the pruned subset, reducing iteration time and
speeding up training.

The algorithm runs on :attr:`~composer.core.event.Event.INIT` and :attr:`~composer.core.event.Event.AFTER_DATLOADER`.
On Event.INIT, it gets the loss function before the model is wrapped. On Event.AFTER_DATALOADER, it applies selective
backprop if the time is between ``self.start`` and ``self.end``.

See the :doc:`Method Card </method_cards/selective_backprop>` for more details.
"""

import importlib
import os
import sys
from pathlib import Path
from typing import Callable

from composer.algorithms.selective_backprop.selective_backprop import SelectiveBackprop as SelectiveBackprop
from composer.algorithms.selective_backprop.selective_backprop import select_using_loss as select_using_loss
from composer.algorithms.selective_backprop.selective_backprop import \
    should_selective_backprop as should_selective_backprop

__all__ = ['SelectiveBackprop', 'select_using_loss', 'should_selective_backprop']

SCORING_FXN_REGISTRY = {}


def register_scoring_fxn(name: str):
    """Registers scoring functions.
    This decorator allows composer to add custom scoring heuristics, even if the
    scoring heuristic is not part of composer. To use it, apply this decorator
    to a scoring function like this:
    .. code-block:: python
        @register_scoring_fxn('scoring_fxn_name')
        def scoring_fxn():
            ...
    and place the file in composer/algorithms/selective_backprop/scoring_functions"""

    def register_scoring_fxn_internal(fxn: Callable[..., Callable]):
        if name in SCORING_FXN_REGISTRY:
            raise ValueError("Cannot register duplicate scoring function ({})".format(name))

        SCORING_FXN_REGISTRY[name] = fxn
        return fxn

    return register_scoring_fxn_internal


def import_scoring_functions() -> None:
    scoring_fxns_path = os.path.join(Path(__file__).parent, "scoring_functions")
    base_module = "composer.algorithms.selective_backprop.scoring_functions"
    for file in os.listdir(scoring_fxns_path):
        if file.endswith((".py", ".pyc")) and not file.startswith("_"):
            module = file[:file.find(".py")]
            if module not in sys.modules:
                module_name = ".".join([base_module, module])
                importlib.import_module(module_name)


import_scoring_functions()
