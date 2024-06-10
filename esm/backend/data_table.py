"""
data_table.py 

@author: Matteo V. Rocco
@institution: Politecnico di Milano

This module provides the DataTable class which is designed to handle and 
manipulate data tables within the package. DataTable encapsulates the data 
storage and transformation functionalities required for complex data operations, 
including handling of CVXPY variables for optimization problems, dynamic data 
structures for coordinates, and integration with pandas for data manipulation.

DataTable is instrumental in structuring and interfacing data for use in 
mathematical models and data analysis within the application.
"""
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cvxpy as cp
import pandas as pd

from esm.log_exc import exceptions as exc
from esm.log_exc.logger import Logger
from esm.support import util


class DataTable:
    """
    A class for managing and interfacing table data for analysis and modeling 
    within the application.

    Attributes:
        logger (Logger): Logger object for logging information, warnings, and errors.
        name (Optional[str]): Name of the data table.
        type (Optional[str]): Type of data table (e.g., 'endogenous', 'exogenous').
        coordinates (List[str]): List of coordinates that define the data structure.
        coordinates_headers (Dict[str, str]): Dictionary mapping coordinates to 
            their headers.
        coordinates_values (Dict[str, Any]): Dictionary mapping coordinates to 
            their values.
        coordinates_dataframe (Optional[pd.DataFrame]): DataFrame representation 
            of coordinates.
        table_headers (Dict[str, Any]): Dictionary of table headers and their types.
        variables_info (Dict[str, Any]): Dictionary containing information about 
            variables.
        foreign_keys (Dict[str, Any]): Dictionary defining foreign key relationships.
        cvxpy_var (Optional[cp.Variable | cp.Parameter | cp.Constant]): CVXPY 
            variable associated with the data table for optimization modeling.
        variables_list (List[str]): List of variables derived from variables_info.

    Methods:
        table_length: Property that returns the number of rows in the 
            coordinates dataframe.
        generate_coordinates_dataframe: Generates a dataframe from coordinates 
            values.
    """

    def __init__(
            self,
            logger: Logger,
            **kwargs,
    ) -> None:
        """
        Initializes a new instance of the DataTable class.

        Args:
            logger (Logger): Logger object for logging operations.
            **kwargs: Arbitrary keyword arguments that can set any attribute 
                of the class.
        """
        self.logger = logger.get_child(__name__)

        self.name: Optional[str] = None
        self.type: Optional[str | dict] = None
        self.coordinates: List[str] = []
        self.coordinates_headers: Dict[str, str] = {}
        self.coordinates_values: Dict[str, Any] = {}
        self.coordinates_dataframe: Optional[dict | pd.DataFrame] = None
        self.table_headers: Dict[str, Any] = {}
        self.variables_info: Dict[str, Any] = {}
        self.foreign_keys: Dict[str, Any] = {}
        self.cvxpy_var: Optional[
            pd.DataFrame[Any, cp.Variable] | cp.Variable] = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.variables_list: List[str] = list(self.variables_info.keys())

    def __repr__(self) -> str:
        avoid_representation = ('logger', 'data', 'coordinates_dataframe')
        output = ''
        for key, value in self.__dict__.items():
            if key in avoid_representation:
                continue
            output += f'\n{key}: {value}'
        return output

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        avoid_iteration = ('logger', 'data', 'coordinates_dataframe')
        for key, value in self.__dict__.items():
            if key in avoid_iteration:
                continue
            yield key, value

    @property
    def table_length(self) -> int:
        """
        Returns the number of rows in the coordinates dataframe.

        Returns:
            int: The number of rows in the dataframe.

        Raises:
            MissingDataError: If the coordinates dataframe is not initialized.
        """
        if self.coordinates_dataframe is not None:
            return len(self.coordinates_dataframe)
        else:
            msg = f"Lenght of data table '{self.name}' unknown."
            self.logger.error(msg)
            raise exc.MissingDataError(msg)

    def generate_coordinates_dataframes(
            self,
            sets_split_problems: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Generates data structure for data tables. 
        This method requires both 'self.coordinates' and 'self.coordinates_values' 
        be predefined. If 'sets_split_problems' is provided, it filters the 
        coordinates values based on the keys in 'sets_split_problems', unpivots 
        the filtered dictionary to a DataFrame, and then merges each row of the 
        filtered DataFrame with the original coordinates DataFrame.
        The result is a dictionary of DataFrames, each corresponding to a key 
        in 'sets_split_problems'. If `sets_split_problems` is not provided, it 
        simply assigns the unpivoted coordinates DataFrame to `self.coordinates_dataframe`.

        Parameters:
            sets_split_problems (dict, optional): A dictionary of keys to filter 
                the coordinates values. Defaults to None.

        Raises:
            MissingDataError: If 'self.coordinates' or 'self.coordinates_values' 
                are not defined.
            TypeError: If 'self.coordinates_values' is not a dictionary.
        """
        if not self.coordinates or not self.coordinates_values:
            msg = "Coordinates and related values must be defined for " \
                f"data table '{self.name}'."
            self.logger.error(msg)
            raise exc.MissingDataError(msg)

        if not isinstance(self.coordinates_values, dict):
            msg = "Coordinates values must be a dictionary."
            self.logger.error(msg)
            raise TypeError(msg)

        coordinates_df = util.unpivot_dict_to_dataframe(
            self.coordinates_values
        )

        if not sets_split_problems:
            self.coordinates_dataframe = coordinates_df

        else:
            coords_split_problems = {
                key: value
                for key, value in self.coordinates_values.items()
                if key in sets_split_problems.values()
            }
            coords_split_problems_df = util.unpivot_dict_to_dataframe(
                coords_split_problems
            )

            coordinates_dataframe_dict = {}

            for set_split_problem in coords_split_problems_df.index:
                coords_filter: pd.Series = \
                    coords_split_problems_df.loc[set_split_problem]
                coords_filter_df = coords_filter.to_frame().T

                coordinates_dataframe_dict[set_split_problem] = pd.merge(
                    left=coords_filter_df,
                    right=coordinates_df,
                    on=coords_filter_df.columns.tolist(),
                )

            self.coordinates_dataframe = coordinates_dataframe_dict
