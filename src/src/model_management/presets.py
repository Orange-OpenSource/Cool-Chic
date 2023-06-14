# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


"""Gather the different encoding preset here."""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Literal


@dataclass
class WarmupPhase():
    candidates: int     # Keep the first <candidates> best systems at the beginning of this warmup phase
    iterations: int     # Number of iterations for each candidates
    lr: float = 1e-2    # Learning rate used during the whole phase

    _col_width_print: ClassVar[int] = 20      # Used only for the to_string() function

    def to_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""
        s = f'{self.candidates:^{self._col_width_print}}|'
        s += f'{self.iterations:^{self._col_width_print}}|'
        s += f'{self.lr:^{self._col_width_print}}|'
        return s

    @classmethod
    def to_string_column_name(cls) -> str:
        """Return the name of the column aligned with the to_string function"""
        col_width = cls._col_width_print
        s = f'{"Starting candidates":^{col_width}}|'
        s += f'{"Itr / candidate":^{col_width}}|'
        s += f'{"Learning rate":^{col_width}}|'
        return s


MODULE_TO_OPTIMIZE = Literal['all', 'arm', 'upsampling', 'synthesis', 'latent']

@dataclass
class TrainerPhase:
    """Represent a particular phase inside a encoding preset"""
    lr: float = 1e-2                # Learning rate used during the whole phase
    max_itr: int = 5000             # Maximum number of iterations
    freq_valid: int = 100           # Check the actual performance each <freq_valid> iterations
    patience: int = 10000           # Stop the phase if no progress after <patience> iterations
    dist: str = 'mse'               # Distortion used during the optimization
    ste: bool = False               # Use the actual quantization during training if True
    quantize_model: bool = False    # Quantize the model at the end of the training phase

    # Python quirk: if we were to do something like optimized_module = []
    # Then the default empty list is stored as a class attribute i.e. shared
    # By all instance of the class see
    # https://stackoverflow.com/questions/53632152/why-cant-dataclasses-have-mutable-defaults-in-their-class-attributes-declaratio
    # List of module to optimized: default is all
    optimized_module: List[MODULE_TO_OPTIMIZE] = field(default_factory=lambda: ['all'])

    _col_width_print: ClassVar[int] = 20      # Used only for the to_string() function

    def __post_init__(self):
        # If all is present in the list of modules to be optimized, alongside something else,
        # it overrides everything, leaving the list of modules to be optimized to just ['all'].
        if 'all' in self.optimized_module:
            self.optimized_module == ['all']

    def to_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""
        col_width = TrainerPhase._col_width_print

        s = f'{self.lr:^{col_width}}|'
        s += f'{self.max_itr:^{col_width}}|'
        s += f'{self.patience:^{col_width}}|'
        s += f'{self.ste:^{col_width}}|'
        s += f'{self.quantize_model:^{col_width}}|'
        return s

    @classmethod
    def to_string_column_name(cls) -> str:
        """Return the name of the column aligned with the to_string function"""
        col_width = cls._col_width_print
        s = f'{"Learning rate":^{col_width}}|'
        s += f'{"Max iterations":^{col_width}}|'
        s += f'{"Patience [itr]":^{col_width}}|'
        s += f'{"STE":^{col_width}}|'
        s += f'{"Quantize model":^{col_width}}|'
        return s



@dataclass
class Preset:
    """Dummy parent class of all encoder presets."""
    preset_name: str = ''                                           # Name of this preset

    # See in TrainingPhase the explanation of the field(default_factory) syntax
    all_phases: List[TrainerPhase] = field(default_factory=list)    # All the training phases
    all_warmups: List[WarmupPhase] = field(default_factory=list)    # All the warm-up phases

    def to_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""
        s = f'Preset: {self.preset_name}\n'
        s += '\nWarm-up\n'
        s += WarmupPhase.to_string_column_name() + '\n'
        for warmup_phase in self.all_warmups:
            s += warmup_phase.to_string() + '\n'

        s += '\nMain training\n'
        s += f'{"index":^10}|{TrainerPhase.to_string_column_name()}\n'
        for idx, training_phase in enumerate(self.all_phases):
            s += f'{idx:^10}|{training_phase.to_string()}\n'
        return s

    def get_total_warmup_iterations(self) -> int:
        """Return the total number of iterations for the whole warm-up."""
        return sum([phase.candidates * phase.iterations for phase in self.all_warmups])


class PresetInstant(Preset):
    """Dummy preset to check that all the possible phases work fine"""
    def __init__(self, start_lr: float = 1e-2, dist: str = 'mse'):
        super().__init__(preset_name='instant')
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(lr=start_lr     , max_itr=100, dist=dist, optimized_module=['all']),

            TrainerPhase(lr=start_lr / 2 , max_itr=10 , dist=dist, optimized_module=['arm']),
            TrainerPhase(lr=start_lr / 2 , max_itr=10 , dist=dist, optimized_module=['synthesis']),

            # --------------------------------- Activate the STE -------------------------------- #
            TrainerPhase(lr=start_lr / 10, max_itr=100 , dist=dist, optimized_module=['all'], ste=True),
            TrainerPhase(lr=start_lr / 10, max_itr=10 , dist=dist, optimized_module=['arm'], ste=True),
            # Quantize the model at the end of this stage
            TrainerPhase(lr=start_lr / 10, max_itr=10 , dist=dist, optimized_module=['synthesis'], ste=True, quantize_model=True),
            # --------------------------------- Activate the STE -------------------------------- #

            # Retrain the latent after quantization
            TrainerPhase(lr=start_lr / 5 , max_itr=10 , dist=dist, optimized_module=['latent'], ste=True),
        ]

        self.all_warmups: List[WarmupPhase] = [
            WarmupPhase(candidates=i, iterations=10, lr=start_lr) for i in range(2, 1, -1)
        ]



class PresetFaster(Preset):
    def __init__(self, start_lr: float = 1e-2, dist: str = 'mse'):
        super().__init__(preset_name='faster')
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(lr=start_lr     , max_itr=4000, dist=dist, optimized_module=['all']),

            TrainerPhase(lr=start_lr / 2 , max_itr=500 , dist=dist, optimized_module=['all']),
            TrainerPhase(lr=start_lr / 2 , max_itr=100 , dist=dist, optimized_module=['latent']),
            TrainerPhase(lr=start_lr / 2 , max_itr=100 , dist=dist, optimized_module=['arm']),
            TrainerPhase(lr=start_lr / 2 , max_itr=100 , dist=dist, optimized_module=['synthesis']),

            TrainerPhase(lr=start_lr / 10, max_itr=500 , dist=dist, optimized_module=['all']),

            TrainerPhase(lr=start_lr / 20, max_itr=500 , dist=dist, optimized_module=['all']),
            # --------------------------------- Activate the STE -------------------------------- #
            TrainerPhase(lr=start_lr / 20, max_itr=200 , dist=dist, optimized_module=['all'], ste=True),
            TrainerPhase(lr=start_lr / 20, max_itr=100 , dist=dist, optimized_module=['arm'], ste=True),
            # Quantize the model at the end of this stage
            TrainerPhase(lr=start_lr / 20, max_itr=100 , dist=dist, optimized_module=['synthesis'], ste=True, quantize_model=True),
            # --------------------------------- Activate the STE -------------------------------- #

            # Retrain the latent after quantization
            TrainerPhase(lr=start_lr / 5 , max_itr=100 , dist=dist, optimized_module=['latent'], ste=True),
        ]

        self.all_warmups: List[WarmupPhase] = [
            WarmupPhase(candidates=i, iterations=200, lr=start_lr) for i in range(5, 1, -1)
        ]



class PresetFast(Preset):
    def __init__(self, start_lr: float = 1e-2, dist: str = 'mse'):
        super().__init__(preset_name='fast')
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(lr=start_lr     , max_itr=4000, dist=dist, optimized_module=['all']),

            TrainerPhase(lr=start_lr / 2 , max_itr=500 , dist=dist, optimized_module=['all']),

            TrainerPhase(lr=start_lr / 5 , max_itr=500 , dist=dist, optimized_module=['all']),
            TrainerPhase(lr=start_lr / 5 , max_itr=100 , dist=dist, optimized_module=['latent']),
            TrainerPhase(lr=start_lr / 5 , max_itr=100 , dist=dist, optimized_module=['arm']),
            TrainerPhase(lr=start_lr / 5 , max_itr=100 , dist=dist, optimized_module=['synthesis']),

            TrainerPhase(lr=start_lr / 10, max_itr=500 , dist=dist, optimized_module=['all']),

            TrainerPhase(lr=start_lr / 20, max_itr=500 , dist=dist, optimized_module=['all']),
            # --------------------------------- Activate the STE -------------------------------- #
            TrainerPhase(lr=start_lr / 20, max_itr=200 , dist=dist, optimized_module=['all'], ste=True),
            TrainerPhase(lr=start_lr / 20, max_itr=100 , dist=dist, optimized_module=['arm'], ste=True),
            # Quantize the model at the end of this stage
            TrainerPhase(lr=start_lr / 20, max_itr=100 , dist=dist, optimized_module=['synthesis'], ste=True, quantize_model=True),
            # --------------------------------- Activate the STE -------------------------------- #

            # Retrain the latent after quantization
            TrainerPhase(lr=start_lr / 5 , max_itr=100 , dist=dist, optimized_module=['latent'], ste=True),
        ]

        self.all_warmups: List[WarmupPhase] = [
            WarmupPhase(candidates=i, iterations=200, lr=start_lr) for i in [16, 8, 4, 2]
        ]


class PresetMedium(Preset):
    def __init__(self, start_lr: float = 1e-2, dist: str = 'mse'):
        super().__init__(preset_name='medium')
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(lr=start_lr     , max_itr=20000, patience=1500, dist=dist, optimized_module=['all']),

            TrainerPhase(lr=start_lr / 2 , max_itr=500 , dist=dist, optimized_module=['all']),

            TrainerPhase(lr=start_lr / 5 , max_itr=500 , dist=dist, optimized_module=['all']),
            TrainerPhase(lr=start_lr / 5 , max_itr=100 , dist=dist, optimized_module=['latent']),
            TrainerPhase(lr=start_lr / 5 , max_itr=100 , dist=dist, optimized_module=['arm']),
            TrainerPhase(lr=start_lr / 5 , max_itr=100 , dist=dist, optimized_module=['synthesis']),

            TrainerPhase(lr=start_lr / 10, max_itr=500 , dist=dist, optimized_module=['all']),

            TrainerPhase(lr=start_lr / 20, max_itr=500 , dist=dist, optimized_module=['all']),
            # --------------------------------- Activate the STE -------------------------------- #
            TrainerPhase(lr=start_lr / 20, max_itr=200 , dist=dist, optimized_module=['all'], ste=True),
            TrainerPhase(lr=start_lr / 20, max_itr=100 , dist=dist, optimized_module=['arm'], ste=True),
            # Quantize the model at the end of this stage
            TrainerPhase(lr=start_lr / 20, max_itr=100 , dist=dist, optimized_module=['synthesis'], ste=True, quantize_model=True),
            # --------------------------------- Activate the STE -------------------------------- #

            # Retrain the latent after quantization
            TrainerPhase(lr=start_lr / 5 , max_itr=500 , dist=dist, optimized_module=['latent'], ste=True),
        ]

        self.all_warmups: List[WarmupPhase] = [
            WarmupPhase(candidates=16, iterations=200, lr=start_lr),
            WarmupPhase(candidates=8 , iterations=400, lr=start_lr),
            WarmupPhase(candidates=2 , iterations=200, lr=start_lr),
        ]

class PresetSlow(Preset):
    """This preset operates based on patience, so the number of iteration per phase should be
    set to an arbitrary big value."""
    def __init__(self, start_lr: float = 1e-2, dist: str = 'mse', n_itr_per_phase: int = 100000):
        super().__init__(preset_name='slow')
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(lr=start_lr      , max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['all']),
            TrainerPhase(lr=start_lr / 2  , max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['all']),
            TrainerPhase(lr=start_lr / 4  , max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['all']),
            TrainerPhase(lr=start_lr / 8  , max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['all']),
            TrainerPhase(lr=start_lr / 16 , max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['all']),

            # --------------------------------- Activate the STE -------------------------------- #
            TrainerPhase(lr=start_lr / 32 , max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['all'], ste=True),
            TrainerPhase(lr=start_lr / 64 , max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['all'], ste=True),
            TrainerPhase(lr=start_lr / 128, max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['all'], ste=True),
            # Quantize the model at the end of this stage
            TrainerPhase(lr=start_lr / 256, max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['all'], ste=True, quantize_model=True),
            # --------------------------------- Activate the STE -------------------------------- #

            # Retrain the latent **without STE** after quantization, then quantize the model once more
            TrainerPhase(lr=start_lr / 256, max_itr=n_itr_per_phase, patience=500, dist=dist, optimized_module=['latent'], quantize_model=True),
            # Final re-tuning of the latent with the final quantized model
            TrainerPhase(lr=start_lr / 256, max_itr=n_itr_per_phase, patience=500, dist=dist, optimized_module=['latent'], ste=True),
        ]

        # From 5 to 2 starting candidates
        self.all_warmups: List[WarmupPhase] = [
            WarmupPhase(candidates=i, iterations=200, lr=start_lr) for i in range(5, 1, -1)
        ]

class PresetPlacebo(Preset):
    """This preset operates based on patience, so the number of iteration per phase should be
    set to an arbitrary big value."""
    def __init__(self, start_lr: float = 1e-2, dist: str = 'mse', n_itr_per_phase: int = 100000):
        super().__init__(preset_name='placebo')
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(lr=start_lr      , max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['all']),
            TrainerPhase(lr=start_lr      , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['latent']),
            TrainerPhase(lr=start_lr      , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['synthesis']),

            TrainerPhase(lr=start_lr / 2  , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['all']),
            TrainerPhase(lr=start_lr / 2  , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['latent']),
            TrainerPhase(lr=start_lr / 2  , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['synthesis']),

            TrainerPhase(lr=start_lr / 4  , max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['all']),
            TrainerPhase(lr=start_lr / 4  , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['latent']),
            TrainerPhase(lr=start_lr / 4  , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['synthesis']),

            TrainerPhase(lr=start_lr / 8  , max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['all']),
            TrainerPhase(lr=start_lr / 8  , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['latent']),
            TrainerPhase(lr=start_lr / 8  , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['synthesis']),

            TrainerPhase(lr=start_lr / 16 , max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['all']),
            TrainerPhase(lr=start_lr / 16 , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['latent']),
            TrainerPhase(lr=start_lr / 16 , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['synthesis']),

            # --------------------------------- Activate the STE -------------------------------- #
            TrainerPhase(lr=start_lr / 32 , max_itr=n_itr_per_phase, patience=1500, dist=dist, optimized_module=['latent'], ste=True),
            TrainerPhase(lr=start_lr / 32 , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['synthesis'], ste=True),
            TrainerPhase(lr=start_lr / 32 , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['upsampling'], ste=True),
            # Quantize the model at the end of this stage
            TrainerPhase(lr=start_lr / 32 , max_itr=n_itr_per_phase, patience= 500, dist=dist, optimized_module=['arm'], ste=True, quantize_model=True),
            # --------------------------------- Activate the STE -------------------------------- #

            # Retrain the latent **with STE** after quantization, then quantize the model once more
            TrainerPhase(lr=start_lr / 256, max_itr=n_itr_per_phase, patience=500, dist=dist, optimized_module=['latent'], ste=True, quantize_model=True),
            TrainerPhase(lr=start_lr / 256, max_itr=n_itr_per_phase, patience=500, dist=dist, optimized_module=['latent'], ste=True),
        ]

        # From 15 to 2 starting candidates
        self.all_warmups: List[WarmupPhase] = [
            WarmupPhase(candidates=i, iterations=200, lr=start_lr) for i in range(15, 1, -1)
        ]


AVAILABLE_PRESET: Dict[str, Preset] = {
    'instant': PresetInstant,
    'faster': PresetFaster,
    'fast': PresetFast,
    'medium': PresetMedium,
    'slow': PresetSlow,
    'placebo': PresetPlacebo,
}

