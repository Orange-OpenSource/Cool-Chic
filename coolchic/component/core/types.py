# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional, Union


@dataclass
class DescriptorNN:
    """Contains information (scale, weight, quantization step, ...) about the
    weights and biases of a neural network."""

    weight: Optional[Any] = None
    bias: Optional[Any] = None

    def multiply(self, factor: float = 1.0) -> Any:
        """Apply a scalar factor to all values within a DescriptorNN.
        It might fail if the values within the DescriptorNN do not support multiplication.

        The operation is **not** done in place.

        Args:
            factor (float, optional): Multiplication factor applied. Defaults to 1.0.

        Returns:
            Self: A new, scaled, DescriptorNN
        """
        results = DescriptorNN()
        for field_wb in fields(DescriptorNN):
            results.set_value(
                self.get_value(field_wb.name) * factor,
                field_wb.name,
            )
        return results

    def pretty_string(self) -> str:
        return f"weight={self.weight:<10}; bias={self.bias:<10}"

    def get_value(self, weight_or_bias: str) -> Any:
        """Return the value of either the weight or bias fields.

        Args:
            weight_or_bias (str): Either "weight" or "bias"

        Returns:
            Any: The value of the requested attributes.
        """
        available_name = [x.name for x in fields(DescriptorNN)]
        if weight_or_bias not in available_name:
            raise ValueError(
                f"Can not get value for weight_or_bias={weight_or_bias}. Available "
                f"names are {available_name}"
            )
        return self.__getattribute__(weight_or_bias)

    def set_value(self, value: Any, weight_or_bias: str) -> None:
        """Set the value of either the weight or bias fields.

        Args:
            value (Any): The value being set.
            weight_or_bias (str): Either "weight" or "bias".

        Returns:
            Any: The value of the requested attributes.
        """
        available_name = [x.name for x in fields(DescriptorNN)]
        if weight_or_bias not in available_name:
            raise ValueError(
                f"Can not get value for weight_or_bias={weight_or_bias}. Available "
                f"names are {available_name}"
            )
        self.__setattr__(weight_or_bias, value)

    def sum(self) -> float:
        """Sum weight and bias attributes if present.
        Might fail if attribute does not support addition
        """
        res = 0.0
        for field_wb in fields(DescriptorNN):
            val = self.get_value(field_wb.name)
            if val is not None:
                res += val
        return res


@dataclass
class DescriptorCoolChic:
    """
    Contains information about the different sub-networks of Cool-chic.
    The default initialization fills all descriptor NN object with None
    """

    arm: DescriptorNN = field(default_factory=lambda: DescriptorNN())
    ifce: DescriptorNN = field(default_factory=lambda: DescriptorNN())
    upsampling: DescriptorNN = field(default_factory=lambda: DescriptorNN())
    synthesis: DescriptorNN = field(default_factory=lambda: DescriptorNN())

    def multiply(self, factor: float = 1.0) -> Any:
        """Apply a scalar factor to all values within a descriptor cool-chic.
        It might fail if the values within all DescriptorNN do not support multiplication.

        The operation is **not** done in place.

        Args:
            factor (float, optional): _description_. Defaults to 1.0.

        Returns:
            Self: A new, scaled, DescriptorCoolChic
        """
        results = DescriptorCoolChic()
        for field_nn in fields(DescriptorCoolChic):
            results.set_value(
                self.get_value(field_nn.name).multiply(factor),
                field_nn.name,
            )
        return results

    def pretty_string(self) -> str:
        """Return a string exposing the content of the DescriptorCoolChic."""
        s = ""
        for field_nn in fields(DescriptorCoolChic):
            name = field_nn.name
            s += f"{name} {self.get_value(field_nn.name).pretty_string()} "
        return s

    def get_value(
        self, module_name: str, weight_or_bias: Optional[str] = None
    ) -> Union[Any, DescriptorNN]:
        """Get the value of a specific field of a DescriptorCoolChic. Return either
        a whole DescriptorNN if weight_or_bias is None or a specific field within
        this descriptorNN.

        Args:
            module_name (str): The module (arm, synthesis, upsampling) we want to
                get the value from.
            weight_or_bias (Optional[str], optional): Either weight or bias to return one
                specific field of the DescriptorNN. Let it to None to return the whole
                DescriptorNN. Defaults to None.

        Returns:
            Union[Any, DescriptorNN]: The desired value
        """

        available_module_name = [x.name for x in fields(DescriptorCoolChic)]

        if module_name not in available_module_name:
            raise ValueError(
                f"Can not get value for module_name={module_name}. Available module name "
                f"is {available_module_name}"
            )

        # Return the whole DescriptorNN
        if weight_or_bias is None:
            return self.__getattribute__(module_name)
        # Return the requested field from the DescriptorNN
        else:
            return self.__getattribute__(module_name).get_value(weight_or_bias)

    def set_value(
        self,
        value: Union[Any, DescriptorNN],
        module_name: str,
        weight_or_bias: Optional[str] = None,
    ) -> None:
        """Set the value of a specific field of a DescriptorCoolChic. Value can be
        a DescriptorNN if weight_or_bias is None i.e. setting values for all fields
        of a DescriptorNN. Otherwise, we set the value of one field of a DescriptorNN
        within the current DescriptorCoolChic.

        Args:
            value (Any, DescriptorNN): The value to set.
            module_name (str): The module (arm, synthesis, upsampling) for which we want to
                set the value.
            weight_or_bias (Optional[str], optional): Either weight or bias to set one
                specific field of the DescriptorNN. Let it to None to set the whole
                DescriptorNN. Defaults to None.
        """

        available_module_name = [x.name for x in fields(DescriptorCoolChic)]

        if module_name not in available_module_name:
            raise ValueError(
                f"Can not set value for module_name={module_name}. Available module name "
                f"is {available_module_name}"
            )

        if weight_or_bias is None and not isinstance(value, DescriptorNN):
            raise ValueError(
                f"Can not set a value of type {type(value)} into a DescriptorNN. "
                f"To only set the value for weight or bias use weight_or_bias=weight (or bias)."
            )

        # Set the whole DescriptorNN
        if weight_or_bias is None:
            self.__setattr__(module_name, value)
        else:
            self.get_value(module_name).set_value(value, weight_or_bias)

    def sum(self) -> float:
        res = 0.0
        for field_nn in fields(DescriptorCoolChic):
            val = self.get_value(field_nn.name)
            if val is not None:
                res += val.sum()
        return res


# For now, it is only possible to have a Cool-chic encoder
# with this name i.e. this key in frame_encoder.coolchic_enc
NAME_COOLCHIC_ENC = Literal["residue", "motion"]


if __name__ == "__main__":
    a = DescriptorCoolChic()

    desc_nn = DescriptorNN()
    desc_nn.set_value(4, "weight")
    desc_nn.set_value(10, "bias")

    a.set_value(2, "synthesis", "weight")
    a.set_value(1, "synthesis", "bias")
    a.set_value(desc_nn, "arm")
    a.set_value(-2, "upsampling", "weight")
    a.set_value(-3, "upsampling", "bias")

    print(a.pretty_string())
    print(a.multiply(2.0).pretty_string())
