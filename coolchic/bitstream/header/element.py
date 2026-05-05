# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>

import math
from dataclasses import dataclass, field, fields
from typing import Any, List, Optional, Union

from coolchic.component.core.synthesis import Synthesis
from coolchic.component.core.types import DescriptorCoolChic, DescriptorNN


def _check_min_max_value(value: int, n_bits: int, signed: bool, name: str = "") -> None:
    """Check if a value can be saved in n_bits.

    Args:
        value (int): The value to write
        n_bits (int): The number of bits available
        signed (bool): Is the value signed?
        name (str, optional): Name of the value, logging purpose only..

    Raises:
        ValueError: Throw an error if the value can not be written
    """
    if signed:
        max_value = 2 ** (n_bits - 1) - 1
        min_value = -(2 ** (n_bits - 1))

    else:
        max_value = 2 ** (n_bits) - 1
        min_value = 0

    if value > max_value or value < min_value:
        raise ValueError(
            f"Trying to convert value {name}={value} to bytes using  "
            f"{n_bits} bits with signed={signed}. Value should be in "
            f"[{min_value}, {max_value}]."
        )


def to_bits(value: int, signed: bool, n_bits: int) -> str:
    """Return a value, expressed as a string of n_bits including
    an optional signed bit. <sign><bits>

    Args:
        value (int): Value to write
        signed (bool): Is it signed?
        n_bits (int): Number of bits

    Returns:
        str: A binary string e.g. 001101
    """
    sign = f"{value < 0:b}" if signed else ""
    # We have one less bit in case of signed value
    n_bits = (n_bits - 1) if signed else n_bits

    return f"{sign}{abs(value):>0{n_bits}b}"


def from_bits(bits: str, signed: bool) -> int:
    """Convert a string of bits into an integer.

    Args:
        bits (str): bits to read.
        signed (bool): Is the first bit a signed bit?

    Returns:
        int: The integer read from the bits
    """
    if signed:
        is_negative = int(bits[0], base=2) == 1
        bits = bits[1:]
    else:
        is_negative = False
        bits = bits
    value = int(bits, base=2)
    if is_negative:
        value *= -1

    return value


@dataclass
class HeaderElement:
    value: int = 0
    n_bits: int = 0
    bits: str = ""

    name: str = ""
    signed: bool = False

    # If true, we write/read indices indexing the list(s) contained in possible_values.
    idx_to_value: bool = False
    possible_values: Optional[Union[List, DescriptorCoolChic]] = None

    def __post_init__(self):

        if self.idx_to_value and self.possible_values is None:
            raise ValueError(
                f"HeaderElement object with idx_to_value={self.idx_to_value} "
                f"must also have an attribute possible_values different from None."
            )

        if self.idx_to_value and self.signed:
            raise ValueError(
                f"HeaderElement object with idx_to_value={self.idx_to_value} "
                f"can not be signed! Set signed=False to use index to value mapping."
            )

        if not (self.idx_to_value) and self.possible_values is not None:
            raise ValueError(
                f"HeaderElement object with idx_to_value={self.idx_to_value} "
                f"can not have an attributes possible_values different from None"
            )

        if self.idx_to_value and isinstance(self.possible_values, list):
            if math.log2(len(self.possible_values)) > self.n_bits:
                raise ValueError(
                    f"Trying to signal the index of {self.name} in a list of length "
                    f"{len(self.possible_values)} with only {self.n_bits} bits."
                )

    def set_value(self, v: int) -> None:
        """Store a value and also set the self.bits attribute based on the value

        Args:
            v (int): The value to write
        """
        self.value = v

        if self.idx_to_value:
            value = self.possible_values.index(self.value)
        else:
            value = self.value

        _check_min_max_value(value, self.n_bits, self.signed, self.name)
        self.bits = to_bits(value, self.signed, self.n_bits)

    def set_bits(self, bits: str) -> str:
        """Store the bits and also compute the corresponding value, stored inside self.value

        Args:
            bits (str): (Potentially longer) binary string. We read only the required number
                of bits from this string and return the rest.

        Returns:
            str: The remaining of the binary string.
        """

        self.bits = bits[: self.n_bits]

        value = from_bits(self.bits, signed=self.signed)

        if self.idx_to_value:
            value = self.possible_values[value]

        self.value = value

        return bits[self.n_bits :]

    def get_value(self) -> Any:
        return self.value


@dataclass
class HeaderElementList(HeaderElement):
    n_bits_per_val: int = 1
    n_val: int = 1

    def __post_init__(self):
        self.n_bits = self.n_val * self.n_bits_per_val
        super().__post_init__()

    def set_value(self, v: List[int]) -> None:
        """Store a value and also set the self.bits attribute based on the value

        Args:
            v (List[int]): The value to write
        """
        self.value = v
        bits = ""
        for v_i in self.value:
            if self.idx_to_value:
                v_i = self.possible_values.index(v_i)

            _check_min_max_value(v_i, self.n_bits_per_val, self.signed, self.name)
            bits += to_bits(v_i, self.signed, self.n_bits_per_val)

        self.bits = bits

    def set_bits(self, bits: str) -> str:
        """Store the bits and also compute the corresponding value, stored inside self.value

        Args:
            bits (str): (Potentially longer) binary string. We read only the required number
                of bits from this string and return the rest.

        Returns:
            str: The remaining of the binary string.
        """

        self.bits = bits[: self.n_bits]
        value = []

        for i in range(self.n_val):
            bits_i = self.bits[i * self.n_bits_per_val : (i + 1) * self.n_bits_per_val]
            v_i = from_bits(bits_i, signed=self.signed)
            if self.idx_to_value:
                v_i = self.possible_values[v_i]
            value.append(v_i)

        self.value = value

        return bits[self.n_bits :]


@dataclass
class HeaderElementDescriptorCoolChic(HeaderElement):
    n_bits_per_val: int = 1
    n_val: int = field(init=False)

    def __post_init__(self):
        self.n_val = len(fields(DescriptorCoolChic)) * len(fields(DescriptorNN))
        self.n_bits = self.n_val * self.n_bits_per_val
        super().__post_init__()

    def set_value(self, descript_cc: DescriptorCoolChic) -> None:
        """Store a value and also set the self.bits attribute based on the value

        Args:
            v (DescriptorCoolChic): The value to write
        """

        self.value = descript_cc
        bits = ""

        for field_nn in fields(DescriptorCoolChic):
            for field_wb in fields(DescriptorNN):
                v_i = descript_cc.get_value(field_nn.name, field_wb.name)

                if self.idx_to_value:
                    possible_val = self.possible_values.get_value(
                        field_nn.name, field_wb.name
                    ).tolist()
                    v_i = possible_val.index(v_i)

                _check_min_max_value(
                    v_i,
                    self.n_bits_per_val,
                    signed=self.signed,
                    # Only for logging in case of error
                    name=f"{self.name}-{field_nn.name}-{field_wb.name}",
                )
                bits += to_bits(v_i, self.signed, self.n_bits_per_val)
        self.bits = bits

    def set_bits(self, bits: str) -> str:
        """Store the bits and also compute the corresponding value, stored inside self.value

        Args:
            bits (str): (Potentially longer) binary string. We read only the required number
                of bits from this string and return the rest.

        Returns:
            str: The remaining of the binary string.
        """

        self.bits = bits[: self.n_bits]
        value = DescriptorCoolChic()

        ptr = 0
        for field_nn in fields(DescriptorCoolChic):
            for field_wb in fields(DescriptorNN):
                bits_i = self.bits[ptr : ptr + self.n_bits_per_val]

                v_i = from_bits(bits_i, signed=self.signed)
                if self.idx_to_value:
                    possible_val = self.possible_values.get_value(
                        field_nn.name, field_wb.name
                    ).tolist()
                    if v_i >= 0 and v_i < len(possible_val):
                        v_i = possible_val[v_i]
                    else:
                        raise ValueError(
                            f"Try to read list of length {len(possible_val)} at index {v_i}."
                            f"Variable name is {self.name}"
                        )
                value.set_value(v_i, field_nn.name, weight_or_bias=field_wb.name)
                ptr += self.n_bits_per_val

        self.value = value

        return bits[self.n_bits :]


@dataclass
class HeaderElementSynLayer(HeaderElement):
    n_bits_out_ft: int = field(init=False, default=7)
    n_bits_k_size: int = field(init=False, default=4)
    n_bits_mode: int = field(init=False, default=1)
    n_bits_non_linearity: int = field(init=False, default=1)

    def __post_init__(self):
        self.n_bits = (
            self.n_bits_out_ft + self.n_bits_k_size + self.n_bits_mode + self.n_bits_non_linearity
        )
        super().__post_init__()

    def set_value(self, layer_description: str) -> None:
        """Store a value and also set the self.bits attribute based on the value

        Args:
            v (layer_description): The value to write
        """
        self.value = layer_description
        out_ft, k_size, mode, non_linearity = Synthesis._parse_layer_syntax(layer_description)
        idx_mode = Synthesis.possible_mode.index(mode)
        idx_nl = list(Synthesis.possible_non_linearity.keys()).index(non_linearity)

        _check_min_max_value(value=out_ft, n_bits=self.n_bits_out_ft, signed=False, name="out_ft")
        _check_min_max_value(value=k_size, n_bits=self.n_bits_k_size, signed=False, name="k_size")
        _check_min_max_value(value=idx_mode, n_bits=self.n_bits_mode, signed=False, name="k_size")
        _check_min_max_value(
            value=idx_nl, n_bits=self.n_bits_non_linearity, signed=False, name="k_size"
        )

        bits = ""
        bits += to_bits(out_ft, False, self.n_bits_out_ft)
        bits += to_bits(k_size, False, self.n_bits_k_size)
        bits += to_bits(idx_mode, False, self.n_bits_mode)
        bits += to_bits(idx_nl, False, self.n_bits_non_linearity)
        self.bits = bits

    def set_bits(self, bits: str) -> str:
        """Store the bits and also compute the corresponding value, stored inside self.value

        Args:
            bits (str): (Potentially longer) binary string. We read only the required number
                of bits from this string and return the rest.

        Returns:
            str: The remaining of the binary string.
        """

        self.bits = bits[: self.n_bits]

        ptr = 0
        bits_i = self.bits[ptr : ptr + self.n_bits_out_ft]
        ptr += self.n_bits_out_ft
        out_ft = from_bits(bits_i, signed=False)

        bits_i = self.bits[ptr : ptr + self.n_bits_k_size]
        ptr += self.n_bits_k_size
        k_size = from_bits(bits_i, signed=False)

        bits_i = self.bits[ptr : ptr + self.n_bits_mode]
        ptr += self.n_bits_mode
        mode = Synthesis.possible_mode[from_bits(bits_i, signed=False)]

        bits_i = self.bits[ptr : ptr + self.n_bits_non_linearity]
        ptr += self.n_bits_non_linearity
        non_linearity = list(Synthesis.possible_non_linearity.keys())[
            from_bits(bits_i, signed=False)
        ]

        self.value = f"{out_ft}-{k_size}-{mode}-{non_linearity}"

        return bits[self.n_bits :]
