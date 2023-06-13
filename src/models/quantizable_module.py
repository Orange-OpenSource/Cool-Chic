# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import copy
import math
from typing import Optional, OrderedDict
import torch
from torch import nn, Tensor
from torch.distributions import Laplace

from utils.constants import MAX_AC_MAX_VAL, MIN_SCALE_NN_WEIGHTS_BIAS
from utils.data_structure import DescriptorNN

class QuantizableModule(nn.Module):
    """This class is **not** made to be instantiated. It is thought as an interface
    from which all the modules should inherit. It implements all the mechanism to
    quantize, entropy code and measure the rate of the Module.
    """

    def __init__(self, possible_q_steps: Tensor):
        """Instantiate a quantizable module with a list of available
        quantization steps.

        Args:
            possible_q_steps (Tensor): A list of the available quantization step for this module
        """
        super().__init__()
        # List of the available quantization steps
        self._POSSIBLE_Q_STEP = possible_q_steps

        # Store the full precision here by calling self.save_full_precision_param()
        self._full_precision_param: Optional[OrderedDict[str, Tensor]] = None

        # Store the quantization step info for the weight and biases
        self._q_step: Optional[DescriptorNN] = None

    def get_param(self) -> OrderedDict[str, Tensor]:
        """Return the weights and biases **currently** inside the Linear modules.

        Returns:
            OrderedDict[str, Tensor]: All the weights and biases inside the layers.
        """
        return OrderedDict({k: v for k, v in self.named_parameters()})

    def set_param(self, param: OrderedDict[str, Tensor]):
        """Set the current parameters of the ARM.

        Args:
            param (OrderedDict[str, Tensor]): Parameters to be set.
        """
        self.load_state_dict(param)

    def save_full_precision_param(self):
        """Store the full precision parameters inside an internal attribute
            self._full_precision_param.
        The current parameters **must** be the full precision parameters.
        """
        self._full_precision_param = copy.deepcopy(self.get_param())

    def get_full_precision_param(self) -> Optional[OrderedDict[str, Tensor]]:
        """Return the **already saved** full precision parameters.
        They must have been saved with self.save_full_precision_param() beforehand!

        Returns:
            Optional[OrderedDict[str, Tensor]]: The full precision parameters if available,
                None otherwise.
        """
        return self._full_precision_param

    def save_q_step(self, q_step: DescriptorNN):
        """Save a quantization step into an internal attribute self._q_step."""
        self._q_step = q_step

    def get_q_step(self) -> Optional[DescriptorNN]:
        """Return the quantization used to go from self._full_precision_param to the
        current parameters.

        Returns:
            Optional[DescriptorNN]: The quantization step which has been used.
        """
        return self._q_step

    def measure_laplace_rate(self) -> DescriptorNN:
        """Get the rate associated with the current parameters.
        # ! No back propagation is possible in this method as we work with float,
        # ! not with tensor.

        Returns:
            DescriptorNN: The rate of the different modules wrapped inside a dictionary
                of float. It does **not** return tensor so no back propagation is possible
        """
        # Concatenate the sent parameters here to measure the entropy later
        sent_param: DescriptorNN = {'bias': [], 'weight': []}
        rate_param: DescriptorNN = {'bias': 0., 'weight': 0.}

        # We don't have a quantization step loaded which means that the parameters are
        # not yet quantized. Return zero rate.
        if self.get_q_step() is None:
            return rate_param

        param = self.get_param()
        # Retrieve all the sent item
        for parameter_name, parameter_value in param.items():
            current_q_step = self.get_q_step_from_parameter_name(parameter_name)
            # Quantization is round(parameter_value / q_step) * q_step so we divide by q_step
            # to obtain the sent latent.
            current_sent_param = (parameter_value / current_q_step).view(-1)

            if self._is_weight(parameter_name):
                sent_param['weight'].append(current_sent_param)
            elif self._is_bias(parameter_name):
                sent_param['bias'].append(current_sent_param)
            else:
                print('Parameter name should end with ".w", ".weight", ".b" or ".bias"')
                print(f'Found: {parameter_name}')
                print('Exiting!')
                return None

        # For each sent parameters (e.g. all biases and all weights)
        # compute their entropy.
        for k, v in sent_param.items():
            # If we do not have any parameter, there is no rate associated.
            # This can happens for the upsampling biases for instance
            if len(v) == 0:
                rate_param[k] = 0.
                continue

            # Concatenate the list of parameters as a big one dimensional tensor
            v = torch.cat(v)
            distrib = Laplace(
                0., max(v.std().item() / math.sqrt(2), MIN_SCALE_NN_WEIGHTS_BIAS)
            )
            # No value can cost more than 32 bits
            proba = torch.clamp(
                distrib.cdf(v + 0.5) - distrib.cdf(v - 0.5), min=2 ** -32, max=None
            )
            rate_param[k] = -torch.log2(proba).sum().item()

        return rate_param

    def _is_weight(self, parameter_name: str) -> bool:
        """Return True is a parameter name ends with ".w" or ".weight".

        Args:
            parameter_name (str): The parameter name

        Returns:
            bool: flag set to true if a parameter is a weight
        """
        return parameter_name.endswith('.weight')

    def _is_bias(self, parameter_name: str) -> bool:
        """Return True is a parameter name ends with ".b" or ".bias".

        Args:
            parameter_name (str): The parameter name

        Returns:
            bool: flag set to true if a parameter is a bias
        """
        return parameter_name.endswith('.bias')

    def get_q_step_from_parameter_name(self, parameter_name: str) -> Optional[float]:
        """Return the specific quantization step from self.q_step (a dictionary with
        several quantization steps). The specific quantization step is selected through
        the parameter name.

        Args:
            parameter_name (str): Name of the parameter in the state dict.

        Returns:
            Optional[float]: The quantization step associated to the parameter. Return None
                if nothing is found.
        """
        # Retrieve the quantization step selected for the model
        q_step = self.get_q_step()

        assert q_step is not None, 'No quantization step associated to the current model.'\
            ' Model must have been quantized with model.quantize(q_step) beforehand.'

        if self._is_weight(parameter_name):
            current_q_step = q_step.get('weight')
        elif self._is_bias(parameter_name):
            current_q_step = q_step.get('bias')
        else:
            print('Parameter name should end with ".w", ".weight", ".b" or ".bias"')
            print(f'Found: {parameter_name}')
            print('Exiting!')
            return None

        return current_q_step

    def quantize(self, q_step: DescriptorNN) -> bool:
        """Quantize **in place** the model with a given quantization step q_step.
        The current model parameters are replaced by the quantized one.

        This methods save the q_step parameter as an attribute of the class

        Args:
            q_step (DescriptorNN): Quantization step for weights and biases.

        Return:
            bool: True if everything went well, False otherwise
        """
        fp_param = self.get_full_precision_param()

        assert fp_param is not None, 'You must save the full precision parameters '\
            'before quantizing the model. Use model.save_full_precision_param().'

        self.save_q_step(q_step)
        q_param = OrderedDict()
        for k, v in fp_param.items():
            current_q_step = self.get_q_step_from_parameter_name(k)
            sent_param = torch.round(v / current_q_step)

            if sent_param.abs().max() > MAX_AC_MAX_VAL:
                print(f'Sent param {k} exceed MAX_AC_MAX_VAL! Q step {current_q_step} too small.')
                return False

            q_param[k] = sent_param * current_q_step
        self.set_param(q_param)

        return True
