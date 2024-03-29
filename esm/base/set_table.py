from typing import Any, Dict, Iterator, List, Tuple
import pandas as pd

from esm import constants
from esm.log_exc.logger import Logger


class SetTable:
    """
    Represents a set with associated attributes and methods.

    Args:
        logger (Logger): An instance of the Logger class for logging purposes.
        data (pd.DataFrame, optional): DataFrame containing set data. 
            Defaults to None.
        **kwargs: Additional keyword arguments representing attributes of 
            the set.

    Attributes:
        logger (Logger): A Logger instance for logging.
        symbol (str): Symbol representing the set.
        table_name (str): Name of the SQLite table associated with the set.
        table_headers (Dict[str, Any]): Headers of the SQLite table associated 
            with the set.
        set_categories (Dict[str, Any]): Categories of the set.
        split_problem (bool): If True, the set is defining multiple numerical 
            problems.
        data (pd.DataFrame): DataFrame containing set data.

    Methods:
        __repr__: Representation of the Set instance.
        __iter__: Iterator over the Set instance.
    """

    def __init__(
            self,
            logger: Logger,
            data: pd.DataFrame = None,
            **kwargs,
    ) -> None:

        self.logger = logger.getChild(__name__)

        self.symbol: str = None
        self.table_name: str = None
        self.table_headers: Dict[str, Any] = None
        self.set_categories: Dict[str, Any] = None
        self.split_problem: bool = False
        self.data = data

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def set_header(self) -> str:
        return self.table_headers[constants._STD_TABLE_HEADER][0]

    @property
    def set_values(self) -> List[str]:
        return list(self.data[self.set_header])

    def __repr__(self) -> str:
        output = ''
        for key, value in self.__dict__.items():
            if key in ('data', 'logger'):
                pass
            elif key != 'values':
                output += f'\n{key}: {value}'
            else:
                output += f'\n{key}: \n{value}'
        return output

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        for key, value in self.__dict__.items():
            if key not in ('data', 'logger'):
                yield key, value
