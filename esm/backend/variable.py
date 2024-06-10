"""
variable.py 

@author: Matteo V. Rocco
@institution: Politecnico di Milano

This module defines the Variable class which manages the characteristics and 
behaviors of variables used in optimization models. It facilitates operations 
such as defining constants, reshaping data, and interfacing with SQL database 
tables for data manipulation.

The class incorporates functionality to handle complex variable structures 
that may include dimensions, mapping of related tables, and operations that 
convert SQL data to formats usable by optimization tools like CVXPY.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import cvxpy as cp
import pandas as pd

from esm.constants import Constants
from esm.log_exc import exceptions as exc
from esm.log_exc.logger import Logger
from esm.support import util


class Variable:
    """
    Manages the definition and operations of variables used in optimization models.

    This class supports handling different aspects of a variable, including 
    its dimensions, related database table, type, and the transformation of its 
    data from database formats to structures suitable for optimization calculations.

    Attributes:
        logger (Logger): Logger object for logging messages.
        rows (Dict[str, Any]): Information about the rows dimension of the 
            variable.
        cols (Dict[str, Any]): Information about the columns dimension of the 
            variable.
        value (Optional[str]): The specific value or type of the variable.
        related_table (Optional[str]): The database table associated with the 
            variable.
        related_dims_map (Optional[pd.DataFrame]): Mapping for dimensions if 
            required for complex variables.
        type (Optional[str]): The type of the variable, such as 'endogenous' 
            or 'exogenous'.
        coordinates_info (Dict[str, Any]): Information about the coordinates of 
            the variable.
        coordinates (Dict[str, Any]): Specific coordinates data for the variable.
        data (Optional[pd.DataFrame]): Data associated with the variable, 
            typically fetched from a database.
        cvxpy_var (Optional[Union[cp.Variable, cp.Parameter, cp.Constant]]): 
            The CVXPY object associated with the variable.

    Methods:
        shape: Property that returns the number of dimensions of the variable.
        shape_size: Property that computes the size of each dimension of the 
            variable.
        dims_labels: Property that retrieves the labels for each dimension of 
            the variable.
        dims_items: Property that retrieves the items for each dimension of 
            the variable.
        dims_labels_items: Property that combines labels and items for each 
            dimension.
        dims_sets: Property that retrieves set names for each dimension if 
            available.
        is_square: Property that checks if the variable's matrix is square.
        is_vector: Property that checks if the variable is a vector.
        sets_parsing_hierarchy: Property that retrieves the parsing hierarchy 
            for sets.
        sets_parsing_hierarchy_values: Property that retrieves values based on 
            the parsing hierarchy.
        all_coordinates: Property that retrieves all coordinates associated 
            with the variable.
        none_data_coordinates: Checks for None data values in CVXPY variables 
            and returns related coordinates.
        reshaping_sqlite_table_data: Reshapes data fetched from SQLite to the 
            required format for CVXPY.
        reshaping_variable_data: Adjusts CVXPY variable data to match SQLite 
            table format.
        define_constant: Defines a constant based on the specified type and 
            validates it against allowed types.
    """

    def __init__(
            self,
            logger: Logger,
            **kwargs,
    ) -> None:
        """
        Initializes a new instance of the Variable class with optional settings 
        for various attributes.

        Args:
            logger (Logger): Logger object for logging operations within the class.
            **kwargs: Arbitrary keyword arguments that set attributes such as 
                rows, cols, value, etc.

        The **kwargs parameter allows for dynamic attribute assignment, which 
        means any property of the class can be initialized via keywords, making 
        the class flexible in handling different types of variables for 
        optimization models.
        """
        self.logger = logger.get_child(__name__)

        self.rows: Dict[str, Any] = {}
        self.cols: Dict[str, Any] = {}
        self.value: Optional[str] = None
        self.related_table: Optional[str] = None
        self.related_dims_map: Optional[pd.DataFrame] = None
        self.type: Optional[str] = None

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.coordinates_info: Dict[str, Any] = {}
        self.coordinates: Dict[str, Any] = {}
        self.data: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        output = ''
        for key, value in self.__dict__.items():
            if key not in ('data', 'logger'):
                output += f'\n{key}: {value}'
        return output

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        for key, value in self.__dict__.items():
            if key not in ('data', 'logger'):
                yield key, value

    @property
    def shape(self) -> List[str | int]:
        """
        Determines the shape of the variable in terms of rows and columns dimensions.

        Returns:
            List[Union[str, int]]: A list containing dimension identifiers or 
                sizes, such as ['set1', 1].
        """
        rows_shape = self.rows['set'] if 'set' in self.rows else 1
        cols_shape = self.cols['set'] if 'set' in self.cols else 1
        return [rows_shape, cols_shape]

    @property
    def shape_size(self) -> Tuple[int]:
        """
        Computes and returns the size of each dimension in the variable. 
        This is useful for determining the dimensionality of the data associated 
        with the variable.

        Returns:
            Tuple[int]: A tuple containing the size of each dimension.
        """
        shape_size = []

        for dimension in ['rows', 'cols']:
            if self.coordinates[dimension]:
                shape_size.append(len(*self.coordinates[dimension].values()))
            else:
                shape_size.append(1)

        return tuple(shape_size)

    @property
    def dims_labels(self) -> List[str]:
        """
        Retrieves the labels for each dimension of the variable, typically used 
        for identifying matrix dimensions.

        Returns:
            List[str]: A list containing labels for each dimension of the variable.
        """
        dim_labels = []

        for dimension in ['rows', 'cols']:
            if self.coordinates_info[dimension]:
                self.coordinates_info: Dict[str, Dict]
                dim_labels.append(
                    list(self.coordinates_info[dimension].values())[0])
            else:
                dim_labels.append(None)

        return dim_labels

    @property
    def dims_items(self) -> List[List[str]]:
        """
        Retrieves the items for each dimension of the variable, which are the 
        specific values that define the dimension.

        Returns:
            List[List[str]]: Lists of items for each dimension.
        """
        dim_items = []

        for dimension in ['rows', 'cols']:
            if self.coordinates[dimension]:
                self.coordinates: Dict[str, Dict]
                dim_items.append(
                    list(*self.coordinates[dimension].values()))
            else:
                dim_items.append(None)

        return dim_items

    @property
    def dims_labels_items(self) -> Dict[str, List[str]]:
        """
        Combines the labels and items for each dimension of the variable into 
        a dictionary.

        Returns:
            Dict[str, List[str]]: Dictionary with dimension labels as keys and 
            the corresponding items as values.
        """
        return {
            self.dims_labels[dim]: self.dims_items[dim]
            for dim in [0, 1]
        }

    @property
    def dims_sets(self) -> Dict[str, str]:
        """Get the sets names corresponding to the rows and cols dimension of
        the variable. If a dimension has not a corrsponding set, it returns
        None as a value of the resulting dictionary.

        Returns:
            Dict[str]: A dictionary containing dimension name as keys and 
                corresponding set name as values.
        """
        dims_sets = {}

        for dim in ['rows', 'cols']:
            if dim in self.coordinates_info:
                dim_set: dict = self.coordinates_info[dim]

                if dim_set:
                    dims_sets[dim] = next(iter(dim_set.keys()), None)
                else:
                    dims_sets[dim] = None

        return dims_sets

    @property
    def is_square(self) -> bool:
        """Checks if the variable matrix is square.

        Returns:
            bool: True if the variable matrix is square, False otherwise.
        """

        if len(self.shape) != 2:
            return False

        if len(self.shape_size) == 2 and \
                self.shape_size[0] == self.shape_size[1]:
            return True
        else:
            return False

    @property
    def is_vector(self) -> bool:
        """Checks if the variable matrix is a vector.

        Returns:
            bool: True if the variable is a vector, False otherwise.
        """

        if len(self.shape_size) == 1 or 1 in self.shape_size:
            return True
        return False

    @property
    def sets_parsing_hierarchy(self) -> Dict[str, str]:
        """
        Retrieves the hierarchical structure of sets parsing for the variable.
        Specifically, it returns inter-problem and intra-problem sets.

        Returns:
            Dict[str, str]: Dictionary representing the hierarchy of sets parsing.
        """
        return {
            **self.coordinates_info['inter'],
            **self.coordinates_info['intra'],
        }

    @property
    def sets_parsing_hierarchy_values(self) -> Dict[str, str]:
        """
        Retrieves the set values based on the hierarchical parsing of sets for the 
        variable.

        Returns:
            Dict[str, str]: Dictionary with parsing hierarchy values.
        """
        return {
            **self.coordinates['intra'],
            **self.coordinates['inter'],
        }

    @property
    def all_coordinates(self) -> Dict[str, List[str] | None]:
        """
        Compiles all coordinates related to the variable into a single dictionary.

        Returns:
            Dict[str, List[str] | None]: Dictionary containing all coordinate 
                values, grouped by dimension.
        """
        # attention: in case a variable has same coordinates in different
        # dimensions, only one of them is reported (rare case of a variable
        # with same rows and cols).
        all_coordinates = {}
        for coordinates in self.coordinates.values():
            all_coordinates.update(coordinates)
        return all_coordinates

    def none_data_coordinates(self, row: int) -> Dict[str, Any] | None:
        """
        Checks if there are any None data values in the CVXPY variables and 
        returns the related coordinates (row in Variable.data and related 
        hierarchy coordinates).

        Args:
            row (int): Identifies the row of Variable.data item (i.e., one 
                specific CVXPY variable).

        Returns:
            Optional[Dict[str, Any]]: Dictionary with keys being the rows where 
                CVXPY variable values are None and values being the names of 
                the sets that identify the variable. Returns None if all data 
                is present.

        Raises:
            KeyError: If the passed row number is out of bounds.
        """
        cvxpy_var_header = Constants.get('_CVXPY_VAR_HEADER')

        if self.data is None \
                or not isinstance(self.data, pd.DataFrame) \
                or cvxpy_var_header not in self.data.columns:
            msg = "Data is not initialized correctly or CVXPY variable header is missing."
            self.logger.error(msg)
            raise ValueError(msg)

        if row < 0 or row > len(self.data):
            msg = f"Passed row number out of bound for variable " \
                f"table '{self.related_table}'. Valid rows between " \
                f"0 and {len(self.data)}."
            self.logger.error(msg)
            raise KeyError(msg)

        cvxpy_var: cp.Variable | cp.Parameter | cp.Constant = \
            self.data.at[row, cvxpy_var_header]

        if cvxpy_var.value is None:
            return {
                key: self.data.loc[row, value]
                for key, value in self.sets_parsing_hierarchy.items()
            }

        return None

    def reshaping_sqlite_table_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        It takes a dataframe with data fetched from SQLite database variable
        table, in the form of a Pandas DataFrame, and elaborate it to get 
        the shape required by the cvxpy variable (two-dimensions matrix).

        Args:
            data (pd.DataFrame): data filtered from the SQLite variable table,
            related to a unique cvxpy variable.

        Returns:
            pd.DataFrame: data reshaped and pivoted to be used as cvxpy values.
        """
        values_header = Constants.get('_STD_VALUES_FIELD')['values'][0]

        # case of a scalar with no rows/cols labels (scalars)
        if all(item is None for item in self.dims_labels):
            index = ''
            columns = None

        # all other variables with rows/cols labels (scalars, vectors/matrices)
        else:
            index = self.dims_labels[0]
            columns = self.dims_labels[1]

        pivoted_data = data.pivot_table(
            index=index,
            columns=columns,
            values=values_header,
            aggfunc='first'
        )

        pivoted_data = pivoted_data.reindex(
            index=self.dims_items[0],
            columns=self.dims_items[1]
        )

        return pivoted_data

    def define_constant(
            self,
            value_type: str,
    ) -> None:
        """
        Defines a constant of a specific type. This method validates the 
        provided value type against a set of allowed values. Depending on the 
        value type, it either creates a constant of the specified type or raises 
        an error if the value type is not supported.

        Constants allowed:
            'identity': identity matrix.
            'sum_vector': summation vector (vector of 1s).
            'lower_triangular': lower triangular matrix of 1s(inc. diagonal)
            'identity_rcot': special identity matrix for rcot problems
            'arange_0': vector/matrix with a range from 0 up to dimension size
            'arange_1': vector/matrix with a range from 1 up to dimension size

        Args:
            value_type (str): The type of the constant to be created. 

        Returns:
            None. The method modifies the state of the object.

        Raises:
            exc.SettingsError: If the provided value type is not supported.
            exc.ConceptualModelError: If the shape of the variable is not 
                suitable for creating the constant.
        """
        util.validate_selection(
            valid_selections=list(Constants.get('_ALLOWED_CONSTANTS').keys()),
            selection=value_type,
        )

        factory_function, args = \
            Constants.get('_ALLOWED_CONSTANTS')[value_type]

        if value_type == 'identity':
            if self.is_square:
                return factory_function(self.shape_size[0], **args)
            else:
                msg = 'Identity matrix must be square. Check variable shape.'

        elif value_type == 'sum_vector':
            if self.is_vector:
                return factory_function(self.shape_size, **args)
            else:
                msg = 'Summation vector must be a vector (one dimension). ' \
                    'Check variable shape.'

        elif value_type in ['arange_0', 'arange_1']:
            return factory_function(self.shape_size, **args)

        elif value_type == 'lower_triangular':
            if self.is_square:
                return factory_function(self.shape_size[0], **args)
            else:
                msg = 'Lower triangular matrix must be square. ' \
                    'Check variable shape.'

        elif value_type == 'identity_rcot':
            if self.related_dims_map is not None and \
                    not self.related_dims_map.empty:
                return factory_function(self.related_dims_map, **args)
            else:
                msg = 'Identity_rcot matrix supported only for variables ' \
                    'with dimensions defined by the same set, or when one set ' \
                    'is defined by items defined as aggregation of items of ' \
                    'another set.'

        else:
            msg = f"Variable 'value': '{value_type}' not supported. " \
                f"Supported value types: {Constants.get('_ALLOWED_CONSTANTS').keys()}"
            self.logger.error(msg)
            raise exc.SettingsError(msg)

        if msg:
            self.logger.error(msg)
            raise exc.ConceptualModelError(msg)
