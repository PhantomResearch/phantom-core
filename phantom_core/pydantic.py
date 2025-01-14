from abc import ABC, abstractmethod
from typing import Annotated, Any

import pandas as pd
from pydantic import PlainValidator, WithJsonSchema
from pydantic_core import core_schema

from .utils import flatten_dict

pdTimestamp = Annotated[
    pd.Timestamp,
    PlainValidator(lambda x: pd.Timestamp(x)),
    WithJsonSchema({"type": 'date-time'})
]

class PydanticFlattnerMixin(ABC):
    """
    A mixin class that provides functionality to flatten Pydantic models.
    """

    @abstractmethod
    def model_dump(self) -> dict:
        """
        Abstract method to dump the model into a dictionary.

        Returns:
            dict: A dictionary representation of the model.
        """
        ...

    def flatten(self) -> dict:
        """
        Flattens the model's dictionary representation.

        This method first dumps the model into a dictionary using `model_dump()`,
        then flattens the resulting dictionary using the `flatten_dict` function.

        Returns:
            dict: A flattened dictionary representation of the model.
        """
        return flatten_dict(self.model_dump())


class StrSubclassMixin:
    """
    Mixin making a str subclass compatible with pydantic.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type: Any, _handler: Any) -> core_schema.CoreSchema:
        """
        Defines the Pydantic core schema for the Ticker class.

        This method is used by Pydantic for validation and serialization.

        Args:
            _source_type: The source type (unused in this implementation).
            _handler: The schema handler (unused in this implementation).

        Returns:
            A CoreSchema object defining the validation and serialization behavior.
        """
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )

