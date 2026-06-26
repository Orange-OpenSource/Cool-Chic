# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

"""Gather the different encoding presets here."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

from coolchic.component.core.quantizer import (
    POSSIBLE_QUANTIZATION_NOISE_TYPE,
    POSSIBLE_QUANTIZER_TYPE,
)
from coolchic.training.loss import DISTORTION_METRIC

MODULE_TO_OPTIMIZE = Literal["all"]


@dataclass
class TrainerPhase:
    """Dataclass representing one phase of an encoding preset.

    Args:
        lr (float): Initial learning rate of the phase. Can vary if
            ``schedule_lr`` is True. Defaults to 0.01.
        max_itr (int): Maximum number of iterations for the phase. The actual
            number of iterations can be made smaller through the patience
            mechanism. Defaults to 10000.
        freq_valid: Check (and print) the performance
            each ``frequency_validation`` iterations. This drives the patience
            mechanism. Defaults to 100.
        patience: After ``patience`` iterations without any
            improvement to the results, exit the training. Patience is disabled
            by setting ``patience = max_iterations``. If patience is used alongside
            cosine_scheduling_lr, then it does not end the training. Instead,
            we simply reload the best model so far once we reach the patience,
            and the training continue. Defaults to 1000.
        quantize_model (bool): If ``True``, quantize the neural networks
            parameters at the end of the training phase. Defaults to ``False``.
        schedule_lr (bool): If ``True``, the learning rate is no longer
            constant. instead, it varies with a cosine scheduling, as suggested
            in  `C3: High-performance and low-complexity neural compression from
            a single image or video, Kim et al.
            <https://arxiv.org/abs/2312.02753>`_. Defaults to False.
        softround_temperature (Tuple[float, float]). Start, end temperature of
            the :doc:`softround function <../component/core/quantizer>`. It is
            used in the forward / backward if ``quantizer_type`` is set to
            ``"softround"`` or ``"softround_alone"``. It is also used in the
            backward pass if ``quantizer_type`` is set to ``"ste"``.
            The softround temperature is linearly scheduled
            during the training. At iteration n° 0 it is equal to
            ``softround_temperature[0]`` while at iteration n° ``max_itr`` it is
            equal to ``softround_temperature[1]``. Note that the patience might
            interrupt the training before it reaches this last value.
            Defaults to (0.3, 0.3).
        noise_parameter (Tuple[float, float]): The random noise temperature is
            linearly scheduled during the training. At iteration n° 0 it is equal
            to ``noise_parameter[0]`` while at iteration n° ``max_itr`` it is equal
            to ``noise_parameter[1]``. Note that the patience might interrupt
            the training before it reaches this last value. Defaults to (2.0,
            1.0).
        quantizer_noise_type (POSSIBLE_QUANTIZATION_NOISE_TYPE): The random noise
            used by the quantizer. More information available in
            :doc:`encoder/component/core/quantizer.py <../component/core/quantizer>`.
            Defaults to ``"kumaraswamy"``.
        quantizer_type (POSSIBLE_QUANTIZER_TYPE): What quantizer to
            use during training. See
            :doc:`encoder/component/core/quantizer.py <../component/core/quantizer>`
            for more information. Defaults to ``"softround"``.
    """

    lmbda: float
    lr: float = 1e-2
    betas_model: Tuple[float, float] = (0.95, 0.95)
    betas_latent: Tuple[float, float] = (0.9, 0.999)
    precondition_frequency_model: int = 10
    max_itr: int = 5000
    freq_valid: int = 100
    patience: int = 10000
    quantize_model: bool = False
    schedule_lr: bool = False
    softround_temperature: Tuple[float, float] = (0.3, 0.3)
    noise_parameter: Tuple[float, float] = (1.0, 1.0)
    quantizer_noise_type: POSSIBLE_QUANTIZATION_NOISE_TYPE = "kumaraswamy"
    quantizer_type: POSSIBLE_QUANTIZER_TYPE = "softround"
    dist_weight: Dict[DISTORTION_METRIC, float] = field(default_factory=lambda: {"mse": 1.0})
    optimized_module: List[MODULE_TO_OPTIMIZE] = field(init=False, default_factory=lambda: ["all"])

    def pretty_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""

        s = f"{f'{self.lr:1.1e}':^{14}}|"
        s += f"{self.max_itr:^{9}}|"
        s += f"{self.patience:^{14}}|"

        quantize_model_str = "Yes" if self.quantize_model else "No"
        s += f"{quantize_model_str:^{13}}|"
        schedule_lr_str = "Yes" if self.schedule_lr else "No"
        s += f"{schedule_lr_str:^{13}}|"

        softround_str = self.quantizer_type[:4]
        if self.quantizer_type not in ["hardround", "none"]:
            softround_str += " / " + " -> ".join([f"{x:1.2f}" for x in self.softround_temperature])
        s += f"{f'{softround_str}':^{22}}|"

        noise_str = self.quantizer_noise_type[:4]
        if self.quantizer_noise_type != "none":
            noise_str += " " + " -> ".join([f"{x:1.2f}" for x in self.noise_parameter])

        s += f"{f'{noise_str}':^{22}}|"

        distortion_str = " ".join([f"{k[:4]}: {v:1.4f}" for k, v in self.dist_weight.items()])
        s += f"{f'{distortion_str}':^{27}}|"
        return s

    @classmethod
    def _pretty_string_column_name(cls) -> str:
        """Return the name of the column aligned with the pretty_string function"""
        s = f"{'Learn rate':^{14}}|"
        s += f"{'Max itr':^{9}}|"
        s += f"{'Patience itr':^{14}}|"
        s += f"{'Quantize NN':^{13}}|"
        s += f"{'Schedule lr':^{13}}|"
        s += f"{'Quant / Temp':^{22}}|"
        s += f"{'Quant / Noise':^{22}}|"
        s += f"{'Distortion':^{27}}|"
        return s

    @classmethod
    def _vertical_line_array(cls) -> str:
        """Return a string made of "-" and "+" matching the columns
        of the print detailed above"""
        s = "-" * 14 + "+"
        s += "-" * 9 + "+"
        s += "-" * 14 + "+"
        s += "-" * 13 + "+"
        s += "-" * 13 + "+"
        s += "-" * 22 + "+"
        s += "-" * 22 + "+"
        s += "-" * 27 + "+"
        return s


@dataclass
class WarmupPhase:
    """Describe one phase of the :doc:`warm-up <../training/warmup>`. At the
    beginning of each warm-up phase, we start by keeping the best ``candidates``
    systems. We then perform a short training, and we go to the next phase.

    Args:
        candidates (int): How many candidates are kept at the beginning of the phase.
        training_phase (TrainerPhase): Describe how the candidates are trained.
    """

    candidates: (
        int  # Keep the first <candidates> best systems at the beginning of this warmup phase
    )
    training_phase: TrainerPhase

    def pretty_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""
        s = f"|{self.candidates:^{14}}|"
        s += f"{self.training_phase.pretty_string()}"
        return s

    @classmethod
    def _pretty_string_column_name(cls) -> str:
        """Return the name of the column aligned with the pretty_string function"""
        s = f"|{'Candidates':^{14}}|"
        s += f"{TrainerPhase._pretty_string_column_name()}"
        return s


@dataclass
class Warmup:
    """A :doc:`warm-up <../training/warmup>` is composed of different phases
    where the worse candidates are successively eliminated.

    Args:
        phase (List[WarmupPhase]): The successive phases of the Warmup.
            Defaults to ``[]``.
    """

    phases: List[WarmupPhase] = field(default_factory=lambda: [])

    def _get_total_iterations(self) -> int:
        """Return the total number of iterations for the whole warm-up."""
        return sum([phase.candidates * phase.training_phase.max_itr for phase in self.phases])


@dataclass
class Preset:
    """Dummy parent (abstract) class of all encoder presets. An actual preset
    should inherit from this class.

    Encoding preset defines how we encode each frame. They are similar to
    conventional codecs presets *e.g* x264 ``--slow`` preset offers better
    compression performance at the expense of a longer encoding.

    Here a preset defines two things: how the :doc:`warm-up <../training/warmup>`
    is done, and how the subsequent :doc:`training <../training/train>` is done.

    Args:
        preset_name (str): Name of the preset.
        lmbda (float): Rate constraint, D + lambda R
        training_phases (List[TrainerPhase]): The successive (post warm-up) training
            phase. Defaults to ``[]``.
        warmup (Warmup): The warm-up parameters. Defaults to ``Warmup()``.
    """

    lmbda: float
    start_lr: float
    itr_main_training: int
    precondition_frequency_model: int = 10
    preset_name: str = ""
    itr_motion_pretrain: int = 0

    dist_weight: Dict[DISTORTION_METRIC, float] = field(default_factory=lambda: {"mse": 1.0})
    # ----- The different training phases and warm-up
    motion_pretrain_phase: List[TrainerPhase] = field(default_factory=lambda: [], init=False)
    warmup: Warmup = field(default_factory=lambda: Warmup(), init=False)
    training_phases: List[TrainerPhase] = field(default_factory=lambda: [], init=False)

    def __post_init__(self):
        # Check that we do quantize the model at least once during the training
        flag_quantize_model = False
        for training_phase in self.training_phases:
            if training_phase.quantize_model:
                flag_quantize_model = True

        # Ignore this assertion if there is no self.training_phases described
        assert flag_quantize_model or len(self.training_phases) == 0, (
            f"The selected preset ({self.preset_name}) does not include "
            f" a training phase with neural network quantization.\n"
            f"{self.pretty_string()}"
        )

    def _get_total_training_iterations(
        self, what: Literal["warmup", "motion-pretrain", "main", "all"] = "all"
    ) -> int:
        """
        Return the total number of iterations for some specific training stages:
            - warmup: the total training iterations for the warm up
            - motion-pretrain: the total training iterations for the motion pre training
            - motion-pretrain: the total training iterations for the main training stage
            - all: sum of the above
        """
        if what not in ["warmup", "motion-pretrain", "main", "all"]:
            raise ValueError(
                f"what must be in ['warmup', 'motion-pretrain', 'main', 'all']. Found what={what}"
            )

        n_itr = 0
        if what in ["warmup", "all"]:
            n_itr += self.warmup._get_total_iterations()

        if what in ["motion-pretrain", "all"]:
            n_itr += sum([phase.max_itr for phase in self.motion_pretrain_phase])

        if what in ["main", "all"]:
            n_itr += sum([phase.max_itr for phase in self.training_phases])

        return n_itr

    def pretty_string(self) -> str:
        """Return a pretty string describing a warm-up phase"""
        s = f"Preset: {self.preset_name:<10}\n"
        s += "-------\n"

        if len(self.motion_pretrain_phase) > 0:
            s += "\nMotion pre-training\n"
            s += "-------------------\n"
            s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
            s += f"|{'Phase index':^14}|{TrainerPhase._pretty_string_column_name()}\n"
            s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
            for idx, training_phase in enumerate(self.motion_pretrain_phase):
                s += f"|{idx:^14}|{training_phase.pretty_string()}\n"
            s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"

        s += "\nWarm-up\n"
        s += "-------\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        s += WarmupPhase._pretty_string_column_name() + "\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        for warmup_phase in self.warmup.phases:
            s += warmup_phase.pretty_string() + "\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"

        s += "\nMain training\n"
        s += "-------------\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        s += f"|{'Phase index':^14}|{TrainerPhase._pretty_string_column_name()}\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"
        for idx, training_phase in enumerate(self.training_phases):
            s += f"|{idx:^14}|{training_phase.pretty_string()}\n"
        s += "+" + "-" * 14 + "+" + TrainerPhase._vertical_line_array() + "\n"

        s += "\nMaximum number of iterations (motion / warm-up / training / total):"
        motion_max_itr = self._get_total_training_iterations(what="motion-pretrain")
        warmup_max_itr = self._get_total_training_iterations(what="warmup")
        training_max_itr = self._get_total_training_iterations(what="main")
        total_max_itr = self._get_total_training_iterations(what="all")
        s += (
            f"{motion_max_itr:^8} / "
            f"{warmup_max_itr:^8} / "
            f"{training_max_itr:^8} / "
            f"{total_max_itr:^8}\n"
        )
        return s


class PresetIntra(Preset):
    def __post_init__(self):

        self.preset_name = "intra"

        if self.itr_main_training < 2000:
            raise ValueError(
                f"Preset {self.preset_name} requires at least 2000 iterations. "
                f"Found --n_itr={self.itr_main_training}."
            )

        log_lambda = math.log10(self.lmbda)
        init_noise_level = (-0.432 * log_lambda + 0.747) / 10.0

        iter_ste_training = 500
        iter_core_training = self.itr_main_training - iter_ste_training

        # ---- Create the 2 warm-up stages, with wu_n_cand initial candidates
        wu_n_stage = 2
        wu_n_cand = 5

        wu_n_iter = 400  # default when plenty of iterations
        if self.itr_main_training < 4000:
            wu_n_iter = 50
        elif self.itr_main_training < 9000:
            wu_n_iter = 100
        elif self.itr_main_training < 12000:
            wu_n_iter = 200

        wu_stages = []
        candidates = wu_n_cand
        for stage in range(min(wu_n_stage, 2)):
            wu = WarmupPhase(
                candidates=candidates,
                training_phase=TrainerPhase(
                    lr=self.start_lr,
                    max_itr=wu_n_iter,
                    freq_valid=100,
                    patience=wu_n_iter,
                    quantize_model=False,
                    schedule_lr=False,
                    softround_temperature=(0.35, 0.35),
                    noise_parameter=(init_noise_level, init_noise_level),
                    quantizer_noise_type="gaussian",
                    quantizer_type="softround",
                    lmbda=self.lmbda,
                    # Warm-up is always pure mse
                    dist_weight={"mse": 1.0},
                    betas_latent=(0.725, 0.97),
                    betas_model=(0.95, 0.95),
                    precondition_frequency_model=1,
                ),
            )
            wu_stages.append(wu)

            iter_core_training -= candidates * wu_n_iter
            candidates = 2  # for eventual second stage
        self.warmup = Warmup(wu_stages)

        print(
            f"wu_n_stage:{wu_n_stage} "
            f"wu_n_cand:{wu_n_cand} "
            f"wu_n_iter:{wu_n_iter} "
            f"core_training={iter_core_training} "
            f"({100.0 * iter_core_training / self.itr_main_training:.0f}% overall)"
        )

        self.training_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=self.start_lr,
                max_itr=iter_core_training,
                patience=5000,
                schedule_lr=True,
                quantizer_type="softround",
                quantizer_noise_type="gaussian",
                softround_temperature=(0.35, 0.08),
                noise_parameter=(0.22, 0.15),
                lmbda=self.lmbda,
                dist_weight=self.dist_weight,
                betas_latent=(0.9, 0.999),
                betas_model=(0.95, 0.95),
                precondition_frequency_model=10,
            ),
            TrainerPhase(
                lr=1.0e-4,
                max_itr=iter_ste_training,
                schedule_lr=True,
                quantizer_type="hardround",
                quantizer_noise_type="none",
                quantize_model=True,
                lmbda=self.lmbda,
                dist_weight=self.dist_weight,
                betas_latent=(0.9, 0.999),
                betas_model=(0.95, 0.95),
                precondition_frequency_model=10,
            ),
        ]

        # Run post init to carry out some checks
        super().__post_init__()


class PresetInter(Preset):
    def __post_init__(self):
        self.preset_name = "inter"

        self.training_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=self.start_lr,
                max_itr=self.itr_main_training,
                patience=5000,
                schedule_lr=True,
                quantizer_type="softround",
                quantizer_noise_type="gaussian",
                softround_temperature=(0.3, 0.1),
                noise_parameter=(0.25, 0.1),
                quantize_model=True,
                lmbda=self.lmbda,
                dist_weight=self.dist_weight,
                betas_latent=(0.9, 0.999),
                betas_model=(0.95, 0.95),
                precondition_frequency_model=10,
            ),
        ]

        self.warmup = Warmup(
            [
                WarmupPhase(
                    candidates=2,
                    training_phase=TrainerPhase(
                        lr=self.start_lr,
                        max_itr=600,
                        freq_valid=600,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type="kumaraswamy",
                        quantizer_type="softround",
                        lmbda=self.lmbda,
                        # Warm-up is always pure mse
                        dist_weight={"mse": 1.0},
                        betas_latent=(0.725, 0.97),
                        betas_model=(0.95, 0.95),
                        precondition_frequency_model=1,
                    ),
                )
            ]
        )

        self.motion_pretrain_phase: List[TrainerPhase] = [
            TrainerPhase(
                lr=1e-2,
                max_itr=self.itr_motion_pretrain,
                patience=self.itr_motion_pretrain,
                schedule_lr=False,
                softround_temperature=(0.3, 0.3),
                noise_parameter=(2.0, 2.0),
                quantizer_noise_type="kumaraswamy",
                quantizer_type="softround",
                # ! Should we still do 20 * lambda?
                lmbda=20 * self.lmbda,
                # Motion pre training is always pure mse
                dist_weight={"mse": 1.0},
                betas_latent=(0.9, 0.999),
                betas_model=(0.95, 0.95),
                precondition_frequency_model=10,
            ),
        ]


class PresetDebug(Preset):
    """Very fast training schedule, should only be used to ensure that the code works properly!"""

    def __post_init__(self):
        self.preset_name = "debug"

        self.training_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=self.start_lr,
                max_itr=50,
                patience=100000,
                schedule_lr=True,
                quantizer_type="softround",
                quantizer_noise_type="gaussian",
                softround_temperature=(0.35, 0.08),
                noise_parameter=(0.22, 0.15),
                lmbda=self.lmbda,
                dist_weight=self.dist_weight,
            ),
            TrainerPhase(
                lr=1e-4,
                max_itr=10,
                patience=10,
                quantizer_type="ste",
                quantizer_noise_type="none",
                quantize_model=True,
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),  # not used since quantizer type is "ste",
                lmbda=self.lmbda,
                dist_weight=self.dist_weight,
            ),
            # Re-tune the latent
            TrainerPhase(
                lr=1.0e-4,
                max_itr=10,
                patience=5,
                quantizer_type="hardround",
                quantizer_noise_type="none",
                freq_valid=10,
                softround_temperature=(1e-4, 1e-4),
                noise_parameter=(1.0, 1.0),  # not used since quantizer type is "ste"
                lmbda=self.lmbda,
                dist_weight=self.dist_weight,
            ),
        ]

        self.warmup = Warmup(
            [
                WarmupPhase(
                    candidates=3, training_phase=TrainerPhase(max_itr=10, lmbda=self.lmbda)
                ),
                WarmupPhase(
                    candidates=2, training_phase=TrainerPhase(max_itr=10, lmbda=self.lmbda)
                ),
            ]
        )

        self.motion_pretrain_phase: List[TrainerPhase] = [
            TrainerPhase(
                lr=self.start_lr,
                max_itr=50,
                patience=50,
                schedule_lr=True,
                quantizer_type="softround",
                quantizer_noise_type="gaussian",
                softround_temperature=(0.3, 0.1),
                noise_parameter=(0.25, 0.1),
                lmbda=self.lmbda,
                dist_weight=self.dist_weight,
            ),
        ]

        # Run post init to carry out some checks
        super().__post_init__()


class PresetMeasureSpeed(Preset):
    def __post_init__(self):
        self.preset_name = "measure_speed"

        # Single stage model with the shortest warm-up ever!
        self.training_phases: List[TrainerPhase] = [
            TrainerPhase(
                lr=self.start_lr,
                max_itr=self.itr_main_training,
                patience=5000,
                schedule_lr=True,
                quantizer_type="softround",
                quantizer_noise_type="gaussian",
                softround_temperature=(0.3, 0.1),
                noise_parameter=(0.25, 0.1),
                quantize_model=True,  # ! This is an important parameter
                lmbda=self.lmbda,
                dist_weight=self.dist_weight,
            ),
        ]

        self.warmup = Warmup(
            [
                WarmupPhase(
                    candidates=1,
                    training_phase=TrainerPhase(
                        lr=self.start_lr,
                        max_itr=1,
                        freq_valid=1,
                        patience=100000,
                        quantize_model=False,
                        schedule_lr=False,
                        softround_temperature=(0.3, 0.3),
                        noise_parameter=(2.0, 2.0),
                        quantizer_noise_type="kumaraswamy",
                        quantizer_type="softround",
                        lmbda=self.lmbda,
                        # Warm-up is always pure mse
                        dist_weight={"mse": 1.0},
                    ),
                )
            ]
        )
        # Run post init to carry out some checks
        super().__post_init__()


AVAILABLE_PRESETS: Dict[str, Preset] = {
    "intra": PresetIntra,
    "inter": PresetInter,
    "debug": PresetDebug,
    "measure_speed": PresetMeasureSpeed,
}
