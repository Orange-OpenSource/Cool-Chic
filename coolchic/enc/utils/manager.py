# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from dataclasses import dataclass, field, fields
from enc.training.presets import AVAILABLE_PRESETS, Preset


@dataclass
class FrameEncoderManager():
    """
    All the encoding option for a frame (loss, lambda, learning rate) as well as some
    counters monitoring the training time, the number of training iterations or the number
    of loops already done.
    """
    # ----- Encoding (i.e. training) options
    preset_name: str                                            # Preset name, should be a key in AVAILABLE_PRESETS utils/encoding_management/presets.py
    start_lr: float = 1e-2                                      # Initial learning rate
    lmbda: float = 1e-3                                         # Rate constraint. Loss = D + lmbda R
    n_itr: int = int(1e5)                                       # Maximum number of training iterations for a **single** phase
    n_loops: int = 1                                            # Number of training loop

    # ==================== Not set by the init function ===================== #
    # ----- Actual preset, instantiated from its name
    preset: Preset = field(init=False)                          # It contains the learning rate in the different phase

    # ----- Monitoring
    idx_best_loop: int = field(default=0, init=False)           # Index of the loop which gives the best results (i.e. the best_loss)
    best_loss: float = field(default=1e6, init=False)           # Overall best loss (for all loops)
    loop_counter: int = field(default=0, init=False)            # Number of loops already done

    # Total training time and iterations counter including warm-up and multi loops
    total_training_time_sec: float = field(default=0., init=False)
    iterations_counter: int = field(default=0, init=False)
    # ==================== Not set by the init function ===================== #

    def __post_init__(self):
        assert self.preset_name in AVAILABLE_PRESETS, f'Preset named {self.preset_name} does not exist.' \
            f' List of available preset:\n{list(AVAILABLE_PRESETS.keys())}.'

        self.preset = AVAILABLE_PRESETS.get(self.preset_name)(start_lr= self.start_lr, n_itr_per_phase=self.n_itr)

        flag_quantize_model = False
        for training_phase in self.preset.all_phases:
            if training_phase.quantize_model:
                flag_quantize_model = True
        assert flag_quantize_model, f'The selected preset ({self.preset_name}) does not include ' \
            f' a training phase with neural network quantization.\n{self.preset.pretty_string()}'

    def record_beaten(self, candidate_loss: float) -> bool:
        """Return True if the candidate loss is better (i.e. lower) than the best loss.

        Args:
            candidate_loss (float): Current candidate loss.

        Returns:
            bool: True if the candidate loss is better than the best loss
                (i.e. candidate < best).
        """
        return candidate_loss < self.best_loss

    def set_best_loss(self, new_best_loss: float):
        """Set the new best loss attribute. It automatically looks at the current loop_counter
        to fill the idx_best_loop attribute.

        Args:
            new_best_loss (float): The new best loss obtained at the current loop
        """
        self.best_loss = new_best_loss
        self.idx_best_loop = self.loop_counter


    def pretty_string(self) -> str:
        """Return a pretty string formatting the data within the class"""
        ATTRIBUTE_WIDTH = 25
        VALUE_WIDTH = 80

        s = 'FrameEncoderManager value:\n'
        s += '--------------------------\n'
        for k in fields(self):
            # Do not print preset, it's quite ugly
            if k.name == 'preset':
                continue
            s += f'{k.name:<{ATTRIBUTE_WIDTH}}: {str(getattr(self, k.name)):<{VALUE_WIDTH}}\n'
        s += '\n'
        return s


