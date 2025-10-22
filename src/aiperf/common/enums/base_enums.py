# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from functools import cached_property

from pydantic import BaseModel, Field
from typing_extensions import Self


class CaseInsensitiveStrEnum(str, Enum):
    """
    CaseInsensitiveStrEnum is a custom enumeration class that extends `str` and `Enum` to provide case-insensitive
    lookup functionality for its members.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        if isinstance(other, Enum):
            return self.value.lower() == other.value.lower()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.value.lower())

    @classmethod
    def _missing_(cls, value):
        """
        Handles cases where a value is not directly found in the enumeration.

        This method is called when an attempt is made to access an enumeration
        member using a value that does not directly match any of the defined
        members. It provides custom logic to handle such cases.

        Returns:
            The matching enumeration member if a case-insensitive match is found
            for string values; otherwise, returns None.
        """
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


class BasePydanticEnumInfo(BaseModel):
    """Base class for all enum info classes that extend `BasePydanticBackedStrEnum`. By default, it
    provides a `tag` for the enum member, which is used for lookup and string comparison,
    and the subclass can provide additional information as needed."""

    tag: str = Field(
        ...,
        min_length=1,
        description="The string value of the enum member used for lookup, serialization, and string insensitive comparison.",
    )

    def __str__(self) -> str:
        return self.tag


class BasePydanticBackedStrEnum(CaseInsensitiveStrEnum):
    """
    Custom enumeration class that extends `CaseInsensitiveStrEnum`
    and is backed by a `BasePydanticEnumInfo` that contains the `tag`, and any other information that is needed
    to represent the enum member.
    """

    # Override the __new__ method to store the `BasePydanticEnumInfo` subclass model as an attribute. This is a python feature that
    # allows us to modify the behavior of the enum class's constructor. We use this to ensure the the enums still look like
    # a regular string enum, but also have the additional information stored as an attribute.
    def __new__(cls, info: BasePydanticEnumInfo) -> Self:
        # Create a new string object based on this class and the tag value.
        obj = str.__new__(cls, info.tag)
        # Ensure string value is set for comparison. This is how enums work internally.
        obj._value_ = info.tag
        # Store the Pydantic model as an attribute.
        obj._info: BasePydanticEnumInfo = info  # type: ignore
        return obj

    @cached_property
    def info(self) -> BasePydanticEnumInfo:
        """Get the enum info for the enum member."""
        # This is the Pydantic model that was stored as an attribute in the __new__ method.
        return self._info  # type: ignore
