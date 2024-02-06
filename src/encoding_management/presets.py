# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
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
class IntraWarmupPhase():
    """Describe one phase of a FrameEncoder intra_warmup()
    """
    candidates: int                         # Keep the first <candidates> best systems at the beginning of this warmup phase
    iterations: int                         # Number of iterations for each candidates
    lr: float = 1e-2                        # Learning rate used during the whole phase
    freq_valid:float = 100                  # period for the validation

    _col_width_print: ClassVar[int] = 20    # Used only for the pretty_string() function

    def pretty_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""
        s = f'{self.candidates:^{self._col_width_print}}|'
        s += f'{self.iterations:^{self._col_width_print}}|'
        s += f'{f"{self.lr:1.2e}":^{self._col_width_print}}|'
        return s

    @classmethod
    def pretty_string_column_name(cls) -> str:
        """Return the name of the column aligned with the pretty_string function"""
        col_width = cls._col_width_print
        s = f'{"Starting candidates":^{col_width}}|'
        s += f'{"Itr / candidate":^{col_width}}|'
        s += f'{"Learning rate":^{col_width}}|'
        return s

@dataclass
class InterWarmupPhase():
    """Describe one phase of a FrameEncoder inter_warmup()
    """
    iterations: int                         # Number of iterations for each phase
    lr: float = 1e-2                        # Learning rate used during the whole phase

    _col_width_print: ClassVar[int] = 20    # Used only for the pretty_string() function

    def pretty_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""
        s = f'{self.iterations:^{self._col_width_print}}|'
        s += f'{f"{self.lr:1.2e}":^{self._col_width_print}}|'
        return s

    @classmethod
    def pretty_string_column_name(cls) -> str:
        """Return the name of the column aligned with the pretty_string function"""
        col_width = cls._col_width_print
        s = f'{"Itr / candidate":^{col_width}}|'
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
    ste: bool = False               # Use the actual quantization during training if True
    quantize_model: bool = False    # Quantize the model at the end of the training phase
    scheduling_period: bool = False # whether a scheduling is applied on the lr
    start_temperature_softround: float = 0.3    # Initial temperature of the softround
    end_temperature_softround: float = 0.3      # Final temperature of the softround

    start_kumaraswamy: float = 1.0              # Initial parameter for the kumaraswamy noise
    end_kumaraswamy: float = 1.0                # Final parameter for the kumaraswamy noise

    # Python quirk: if we were to do something like optimized_module = []
    # Then the default empty list is stored as a class attribute i.e. shared
    # By all instance of the class see
    # https://stackoverflow.com/questions/53632152/why-cant-dataclasses-have-mutable-defaults-in-their-class-attributes-declaratio
    # List of module to optimized: default is all
    optimized_module: List[MODULE_TO_OPTIMIZE] = field(default_factory=lambda: ['all'])

    _col_width_print: ClassVar[int] = 14      # Used only for the pretty_string() function

    def __post_init__(self):
        # If all is present in the list of modules to be optimized, alongside something else,
        # it overrides everything, leaving the list of modules to be optimized to just ['all'].
        if 'all' in self.optimized_module:
            self.optimized_module == ['all']

    def pretty_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""
        col_width = TrainerPhase._col_width_print

        s = f'{f"{self.lr:1.2e}":^{self._col_width_print}}|'
        s += f'{self.max_itr:^{8}}|'
        s += f'{self.patience:^{16}}|'
        s += f'{self.freq_valid:^{col_width}}|'
        s += f'{self.ste:^{5}}|'
        s += f'{self.quantize_model:^{col_width}}|'
        s += f'{self.scheduling_period:^{col_width}}|'
        s += f'{f"{self.start_temperature_softround:1.2e}":^{self._col_width_print}}|'
        s += f'{f"{self.end_temperature_softround:1.2e}":^{self._col_width_print}}|'
        s += f'{f"{self.start_kumaraswamy:.2f}":^{self._col_width_print}}|'
        s += f'{f"{self.end_kumaraswamy:.2f}":^{self._col_width_print}}|'
        return s

    @classmethod
    def pretty_string_column_name(cls) -> str:
        """Return the name of the column aligned with the pretty_string function"""
        col_width = cls._col_width_print
        s = f'{"Learn rate":^{col_width}}|'
        s += f'{"Max itr":^{8}}|'
        s += f'{"Patience [itr]":^{16}}|'
        s += f'{"Valid [itr]":^{col_width}}|'
        s += f'{"STE":^{5}}|'
        s += f'{"Quantize NN":^{col_width}}|'
        s += f'{"Schedule lr":^{col_width}}|'
        s += f'{"Start T sr":^{col_width}}|'
        s += f'{"End T sr":^{col_width}}|'
        s += f'{"Start kumara":^{col_width}}|'
        s += f'{"End kumara":^{col_width}}|'
        return s


@dataclass
class Preset:
    """Dummy parent class of all encoder presets."""
    preset_name: str = ''

    # See in TrainingPhase the explanation of the field(default_factory) syntax
    # All the training phases
    all_phases: List[TrainerPhase] = field(default_factory=lambda : [])
    # All the warm-up phases
    all_intra_warmups: List[IntraWarmupPhase] = field(default_factory=lambda : [])

    # Describe one stage (i.e. one reference distance) of the inter warmup. All stages
    # are identical
    inter_warmup: InterWarmupPhase = field(init=False)


    def pretty_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""
        s = f'Preset: {self.preset_name:>10}\n'
        s += '------------------\n'
        s += '\nIntra Warm-up\n'
        s += '-------------\n'
        s += IntraWarmupPhase.pretty_string_column_name() + '\n'
        for warmup_phase in self.all_intra_warmups:
            s += warmup_phase.pretty_string() + '\n'

        s += '\nInter Warm-up\n'
        s += '-------------\n'
        s += InterWarmupPhase.pretty_string_column_name() + '\n'
        s += self.inter_warmup.pretty_string() + '\n'

        s += '\nMain training\n'
        s += '-------------\n'
        s += f'{"Idx":^6}|{TrainerPhase.pretty_string_column_name()}\n'
        for idx, training_phase in enumerate(self.all_phases):
            s += f'{idx:^6}|{training_phase.pretty_string()}\n'
        return s

    def get_total_warmup_iterations(self) -> int:
        """Return the total number of iterations for the whole warm-up."""
        return sum([phase.candidates * phase.iterations for phase in self.all_intra_warmups])


class PresetC3(Preset):
    def __init__(self, start_lr: float = 1e-2, n_itr_per_phase: int = 100000):
        super().__init__(preset_name='c3')
        # 1st stage: with soft round and kumaraswamy noise
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=start_lr,
                max_itr=n_itr_per_phase,
                patience=100000,
                optimized_module=['all'],
                scheduling_period=True,
                start_temperature_softround=0.3,
                end_temperature_softround=0.1,
                start_kumaraswamy=2.0,
                end_kumaraswamy=1.0
            )
        ]

        # Second stage with STE
        lr = 0.0001
        while lr >  10.e-8:
            # max itr = 100
            self.all_phases.append(
                TrainerPhase(
                    lr=lr,
                    max_itr=100,
                    patience=50,
                    optimized_module=['all'],
                    freq_valid=10,
                    ste=True,
                    # This is only used to parameterize the backward of the quantization
                    start_temperature_softround=1e-4,
                    end_temperature_softround=1e-4,
                    # Kumaraswamy noise with parameter = 1 --> Uniform noise
                    start_kumaraswamy=1.0,
                    end_kumaraswamy=1.0
                )
            )
            lr *= 0.8

        # Final stage: quantize the networks and then re-tuned the latent
        self.all_phases.append(
            TrainerPhase(
                lr=lr,
                max_itr=100,
                patience=100,
                optimized_module=['all'],
                ste=True,
                quantize_model=True,                # ! This is the important parameter
                start_temperature_softround=1e-4,
                end_temperature_softround=1e-4,
                start_kumaraswamy=1.0,
                end_kumaraswamy=1.0
            )
        )
        self.all_phases.append(
            TrainerPhase(
                lr=1e-4,
                max_itr= 1000,
                patience= 50,
                optimized_module=['latent'],        # ! Only fine tune the latent
                freq_valid=10,
                ste=True,
                start_temperature_softround=1e-4,
                end_temperature_softround=1e-4,
                start_kumaraswamy=1.0,
                end_kumaraswamy=1.0
            )
        )

        # 5 candidates, then 2 then 1
        self.all_intra_warmups: List[IntraWarmupPhase] = [
            IntraWarmupPhase(candidates=5, iterations=400, lr=start_lr),
            IntraWarmupPhase(candidates=2, iterations=400, lr=start_lr)
        ]

        self.inter_warmup = InterWarmupPhase(iterations=500, lr=start_lr)

class PresetC3x(Preset):
    def __init__(self, start_lr: float = 1e-2, n_itr_per_phase: int = 100000):
        super().__init__(preset_name='c3x')
        # 1st stage: with soft round and kumaraswamy noise
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=start_lr,
                max_itr=n_itr_per_phase,
                patience=100000,
                optimized_module=['all'],
                scheduling_period=True,
                start_temperature_softround=0.3,
                end_temperature_softround=0.1,
                start_kumaraswamy=2.0,
                end_kumaraswamy=1.0
            )
        ]

        # Second stage with STE
        self.all_phases.append(
            TrainerPhase(
                lr=1.e-4,
                max_itr=2000,
                patience=2000,
                optimized_module=['all'],
                scheduling_period=True,
                ste=True,
                # This is only used to parameterize the backward of the quantization
                start_temperature_softround=1e-4,
                end_temperature_softround=1e-4,
                # Kumaraswamy noise with parameter = 1 --> Uniform noise
                start_kumaraswamy=1.0,
                end_kumaraswamy=1.0
            )
        )

        # Final stage: quantize the networks and then re-tuned the latent
        self.all_phases.append(
            TrainerPhase(
                lr=1.e-4,
                max_itr=100,
                patience=100,
                optimized_module=['all'],
                ste=True,
                quantize_model=True,                # ! This is the important parameter
                start_temperature_softround=1e-4,
                end_temperature_softround=1e-4,
                start_kumaraswamy=1.0,
                end_kumaraswamy=1.0
            )
        )
        self.all_phases.append(
            TrainerPhase(
                lr=1.e-4,
                max_itr= 1000,
                patience= 50,
                optimized_module=['latent'],        # ! Only fine tune the latent
                freq_valid=10,
                ste=True,
                start_temperature_softround=1e-4,
                end_temperature_softround=1e-4,
                start_kumaraswamy=1.0,
                end_kumaraswamy=1.0
            )
        )

        # 25 candidates, then 5 then 2
        self.all_intra_warmups: List[IntraWarmupPhase] = [
            # IntraWarmupPhase(candidates=20, iterations= 10, lr=start_lr, freq_valid=  10), # eliminates degenerations... but uses to much memory
            IntraWarmupPhase(candidates= 5, iterations=400, lr=start_lr, freq_valid= 400),
            IntraWarmupPhase(candidates= 2, iterations=400, lr=start_lr, freq_valid= 400)
        ]

        self.inter_warmup = InterWarmupPhase(iterations=500, lr=start_lr)


class PresetDebug(Preset):
    """Very fast training schedule, should only be used to ensure that the code works properly!"""
    def __init__(self, start_lr: float = 1e-2, n_itr_per_phase: int = 100000):
        super().__init__(preset_name='debug')
        self.all_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=start_lr,
                max_itr=100,
                patience=100000,
                optimized_module=['all'],
                scheduling_period=True,
                start_temperature_softround=0.3,
                end_temperature_softround=0.1,
                start_kumaraswamy=2.0,
                end_kumaraswamy=1.0
            )
        ]

        self.all_phases.append(
            TrainerPhase(
                lr=1e-4,
                max_itr=10,
                patience=100,
                optimized_module=['all'],
                ste=True,
                quantize_model=True,
                start_temperature_softround=1e-4,
                end_temperature_softround=1e-4,
                start_kumaraswamy=1.0,
                end_kumaraswamy=1.0
            )
        )

        self.all_phases.append(
            TrainerPhase(
                lr=1e-4,
                max_itr=10,
                patience=50,
                optimized_module=['latent'],
                freq_valid=10,
                ste=True,
                start_temperature_softround=1e-4,
                end_temperature_softround=1e-4,
                start_kumaraswamy=1.0,
                end_kumaraswamy=1.0
            )
        )

        self.all_intra_warmups: List[IntraWarmupPhase] = [
            IntraWarmupPhase(candidates=2, iterations=20, lr=start_lr)
        ]

        self.inter_warmup = InterWarmupPhase(iterations=30, lr=start_lr)


AVAILABLE_PRESETS: Dict[str, Preset] = {
    'c3': PresetC3,
    'c3x': PresetC3x,
    'debug': PresetDebug,
}

