"""
index.py 

@author: Matteo V. Rocco
@institution: Politecnico di Milano

This module defines the Index class, which acts as a central registry for managing 
sets, data tables, and variables within the modeling system. It facilitates access 
and manipulation of these entities throughout the application.

The Index class loads and initializes set tables, data tables, and variable 
objects from configured sources and provides properties to access metadata and 
operational characteristics related to these entities.
"""
from pathlib import Path
from typing import Dict, List, Optional, Type

import pandas as pd

from esm.constants import Constants
from esm.backend.data_table import DataTable
from esm.backend.set_table import SetTable
from esm.backend.variable import Variable
from esm.log_exc import exceptions as exc
from esm.log_exc.logger import Logger
from esm.support import util
from esm.support.file_manager import FileManager
from esm.support.dotdict import DotDict


class Index:
    """
    A central index for managing and accessing sets, data tables, and variables 
    within the system.

    Attributes:
        logger (Logger): Logger instance for logging activity within the class.
        files (FileManager): FileManager instance for file operations.
        paths (Dict[str, Path]): Dictionary mapping of paths used in file operations.
        sets (Dict[str, SetTable]): Dictionary of set tables loaded upon initialization.
        data (Dict[str, DataTable]): Dictionary of data tables loaded upon initialization.
        variables (Dict[str, Variable]): Dictionary of variables fetched upon initialization.

    Parameters:
        logger (Logger): Logger instance from the parent or main handler.
        files (FileManager): FileManager instance for handling file operations.
        paths (Dict[str, Path]): Dictionary containing necessary path configurations.
    """

    def __init__(
            self,
            logger: Logger,
            files: FileManager,
            paths: Dict[str, Path],
    ) -> None:
        """
        Initializes the Index object, loads sets, data tables, and variables.
        """
        self.logger = logger.get_child(__name__)
        self.logger.debug("Object initialization...")

        self.files = files
        self.paths = paths

        self.sets: Dict[str, SetTable] = self.load_sets_tables()
        self.data: Dict[str, DataTable] = self.load_data_tables()
        self.variables: Dict[str, Variable] = self.fetch_variables()

        self.fetch_vars_coordinates_info()

        self.logger.debug("Object initialized.")

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}'

    @property
    def sets_split_problem_dict(self) -> Dict[str, str]:
        """
        Provides a dictionary of sets that have a split problem, mapped to 
        their respective headers. Returns an empty dictionary if no sets are 
        identified with a split problem.

        Returns:
            Dict[str, str]: A dictionary where keys are set identifiers and values 
                are headers associated with split problems. Returns an empty 
                dictionary if the required information is not available or applicable.
        """
        sets_split_problem_list = {}
        name_header = Constants.get('_STD_NAME_HEADER')

        for key, set_table in self.sets.items():
            if getattr(set_table, 'split_problem', False) and \
                    set_table.table_headers is not None:

                headers = set_table.table_headers.get(name_header, [])

                if headers:
                    sets_split_problem_list[key] = headers[0]

        return sets_split_problem_list

    @property
    def list_sets(self) -> List[str]:
        """
        Returns a list of all set identifiers currently loaded in the index.
        Returns an empty list if no sets are loaded.

        Returns:
            List[str]: List of set identifiers.
        """
        return list(self.sets.keys()) if self.sets else []

    @property
    def list_data_tables(self) -> List[str]:
        """
        Returns a list of all data table identifiers currently loaded in the index.
        Returns an empty list if no data tables are loaded.

        Returns:
            List[str]: List of data table identifiers.
        """
        return list(self.data.keys()) if self.sets else []

    @property
    def list_variables(self) -> List[str]:
        """
        Returns a list of all variable identifiers currently loaded in the index.
        Returns an empty list if no variables are loaded.

        Returns:
            List[str]: List of variable identifiers.
        """
        return list(self.variables.keys()) if self.variables else []

    def _load_and_validate(
            self,
            file_key: int,
            validation_structure: Dict,
            object_class: Type[SetTable | DataTable],
    ) -> DotDict:
        """
        Loads data from a specified file and validates it against a given structure.
        If validation passes, it creates instances of the specified class for 
        each entry.

        Parameters:
            file_key (int): Key to retrieve the file name from constants.
            validation_structure (Dict): The structure against which to validate 
                the data.
            object_class (Type[SetTable | DataTable]): The class to instantiate 
                with the validated data.

        Returns:
            DotDict: A DotDict of instantiated objects, keyed by their original 
                identifiers in the data file.

        Raises:
            SettingsError: If any dictionary in the loaded data does not conform 
                to the validation structure.

        Notes:
            This method assumes that the data file is structured as a dictionary 
                of dictionaries.
        """
        self.logger.debug(
            "Loading and validating data from file, "
            f"generating '{object_class.__name__}' objects.")

        data = self.files.load_file(
            file_name=Constants.get('_SETUP_FILES')[file_key],
            dir_path=self.paths['model_dir']
        )

        invalid_entries = {}
        for key, dictionary in data.items():
            if not util.validate_dict_structure(dictionary, validation_structure):
                invalid_entries[key] = dictionary

        if invalid_entries:
            msg = f"'{object_class.__name__}' data validation not successful " \
                f"for entries {list(invalid_entries.keys())}." \
                "Input data must comply with default structure. " \
                f"Check setup file: '{Constants.get('_SETUP_FILES')[file_key]}'"
            self.logger.error(msg)
            raise exc.SettingsError(msg)

        return DotDict({
            key: object_class(logger=self.logger, **value)
            for key, value in data.items()
        })

    def load_sets_tables(self) -> Dict[str, SetTable]:
        """
        Loads and validates set tables from a designated configuration file.

        Returns:
            Dict[str, SetTable]: A dictionary of `SetTable` instances keyed by 
                their identifiers.

        Raises:
            SettingsError: If any part of the data fails to validate against 
                the set structure defined in constants.
        """
        return self._load_and_validate(
            file_key=0,
            validation_structure=Constants.get('_SET_DEFAULT_STRUCTURE'),
            object_class=SetTable,
        )

    def load_data_tables(self) -> Dict[str, DataTable]:
        """
        Loads and validates data tables from a designated configuration file, 
        then configures each data table with additional headers derived from 
        the corresponding set tables.

        Returns:
            Dict[str, DataTable]: A dictionary of `DataTable` instances keyed 
                by their identifiers.

        Raises:
            MissingDataError: If any expected set key is missing or if required 
                headers are not available.
        """
        data_tables = self._load_and_validate(
            file_key=1,
            validation_structure=Constants.get(
                '_DATA_TABLE_DEFAULT_STRUCTURE'),
            object_class=DataTable,
        )

        for table in data_tables.values():
            table: DataTable
            set_headers_key = Constants.get('_STD_NAME_HEADER')

            try:
                table.table_headers = {
                    set_key: self.sets[set_key].table_headers[set_headers_key]
                    for set_key in table.coordinates
                    if self.sets.get(set_key) and self.sets[set_key].table_headers
                }
            except KeyError as e:
                msg = f"Set key {e} not found in sets or table_headers is None."
                self.logger.error(msg)
                raise exc.MissingDataError(msg) from e

            table.table_headers = util.add_item_to_dict(
                dictionary=table.table_headers,
                item=Constants.get('_STD_ID_FIELD'),
                position=0,
            )
            table.coordinates_headers = {
                key: value[0] for key, value in table.table_headers.items()
                if key in table.coordinates
            }

        return data_tables

    def fetch_variables(self) -> Dict[str, Variable]:
        """
        Fetches and validates variable information from all loaded data tables, 
        generating 'Variable' objects for each valid entry.
        The method checks each variable's data against a defined structure and 
        initializes 'Variable' instances if the data conforms to the expected format.

        Returns:
            Dict[str, Variable]: A dictionary of 'Variable' objects keyed by 
                their identifiers, collected across all data tables.

        Raises:
            SettingsError: If any variable data does not conform to the required 
                structure, specifying the table and the nature of the validation 
                failure.
        """
        self.logger.debug(
            "Fetching and validating data, generating "
            f"'{Variable.__name__}' objects.")

        variables_info = DotDict({})

        for table_key, data_table in self.data.items():
            data_table: DataTable

            invalid_vars = [
                var_key for var_key, var_info in data_table.variables_info.items()
                if var_info is not None and not util.validate_dict_structure(
                    dictionary=var_info,
                    validation_structure=Constants.get(
                        '_VARIABLE_DEFAULT_STRUCTURE')
                )
            ]

            if invalid_vars:
                msg = f"Invalid variables structures in table '{table_key}' " \
                    f"for variables {invalid_vars}: variable fields do not " \
                    "match the expected default format."
                self.logger.error(msg)
                raise exc.SettingsError(msg)

            variable = DotDict({
                var_key: Variable(
                    logger=self.logger,
                    related_table=table_key,
                    type=data_table.type,
                    **(var_info or {}),
                )
                for var_key, var_info in data_table.variables_info.items()
            })
            variables_info.update(variable)

        return variables_info

    def fetch_vars_coordinates_info(self) -> None:
        """
        Populates each variable in the index with detailed coordinate information,
        categorizing data table headers into rows, columns, intra-problem sets,
        and inter-problem sets based on the variable's shape and configuration.
        It derives this information from the related data tables specified for 
        each variable.

        Raises:
            ConceptualModelError: If there are multiple intra-problem sets found, 
                indicating a configuration error in the variable setup.
        """
        self.logger.debug("Fetching 'coordinates_info' to Index.variables.")

        for var_key, variable in self.variables.items():

            if variable.related_table is None:
                msg = "Variable related table not defined for variable " \
                    f"'{var_key}'."
                self.logger.error(msg)
                raise exc.MissingDataError(msg)

            related_table_data = self.data.get(variable.related_table)

            if not related_table_data:
                msg = f"No data found for related table '{variable.related_table}' "
                self.logger.error(msg)
                raise exc.MissingDataError(msg)

            related_table_headers = related_table_data.table_headers
            rows, cols, intra, inter = {}, {}, {}, {}

            for key, value in related_table_headers.items():
                table_header = value[0]

                if key not in Constants.get('_STD_ID_FIELD'):
                    if key == variable.shape[0]:
                        rows[key] = table_header
                    if key == variable.shape[1]:
                        cols[key] = table_header
                    if key not in variable.shape:
                        if key not in self.sets_split_problem_dict:
                            intra[key] = table_header
                        else:
                            inter[key] = table_header

            if len(intra) > 1:
                msg = "Only one intra-problem set allowed. Current " \
                    f"intra-problem sets: '{intra}'."
                self.logger.error(msg)
                raise exc.ConceptualModelError(msg)

            variable.coordinates_info = {
                'rows': rows,
                'cols': cols,
                'intra': intra,
                'inter': inter,
            }

    def fetch_foreign_keys_to_data_tables(self) -> None:
        """
        Assigns foreign key relationships to the data tables in the index. 
        This is achieved by referencing set tables that are associated with 
        each data table based on coordinate headers.        
        Each foreign key relationship is defined by matching coordinate headers 
        in the data tables with set tables, effectively linking related data 
        entities across the model.
        """
        self.logger.debug(
            "Loading tables 'foreign_keys' to Index.data_tables.")

        for table in self.data.values():
            if not hasattr(table, 'foreign_keys'):
                table.foreign_keys = {}

            for set_key, set_header in table.coordinates_headers.items():
                if set_key in self.sets:
                    table.foreign_keys[set_header] = \
                        (set_header, self.sets[set_key].table_name)
                else:
                    msg = f"Set key '{set_key}' not found in sets when " \
                        "assigning foreign keys."
                    self.logger.error(msg)
                    raise exc.MissingDataError(msg)

    def load_sets_data_to_index(
            self,
            excel_file_name: str,
            excel_file_dir_path: Path | str,
            empty_data_fill='',
    ) -> None:
        """
        Loads data for sets from an Excel file into the Index. If any set already 
        contains data, prompts the user to decide whether to overwrite the existing data.

        Parameters:
            excel_file_name (str): The name of the Excel file to load.
            excel_file_dir_path (Path | str): The directory path where the Excel 
                file is located.
            empty_data_fill (str, optional): The value to use for filling in 
                empty cells in the Excel data.

        Raises:
            MissingDataError: If a table is referenced in a set but not found 
                in the Excel file and not defined to be copied from another set, 
                or if necessary headers are missing.
            SettingsError: If the set to be copied does not exist or its data 
                is improperly formatted.
        """
        if all(
            set_instance.data is None
            for set_instance in self.sets.values()
        ):
            self.logger.debug("Loading Sets data to Index.sets.")
        else:
            self.logger.warning(
                "At least one Set is already defined in Index.")
            user_input = input("Overwrite Sets in Index? (y/[n]): ")
            if user_input.lower() != 'y':
                self.logger.info("Sets not overwritten in Index.")
                return
            self.logger.info("Overwriting Sets in Index.")

        sets_excel_data = self.files.excel_to_dataframes_dict(
            excel_file_name=excel_file_name,
            excel_file_dir_path=excel_file_dir_path,
            empty_data_fill=empty_data_fill,
            dtype=str
        )

        sets_excel_keys = sets_excel_data.keys()

        for set_instance in self.sets.values():
            set_instance: SetTable

            if set_instance.table_name in sets_excel_keys:
                set_instance.data = sets_excel_data[set_instance.table_name]
                continue

            if not set_instance.copy_from:
                msg = f"Table '{set_instance.table_name}' not included in " \
                    "the excel sets file, nor defined as a copy of another " \
                    "existing set. Check sets definition."
                self.logger.error(msg)
                raise exc.MissingDataError(msg)

            if set_instance.table_headers is None:
                msg = f"Headers for table '{set_instance.table_name}' not defined."
                self.logger.error(msg)
                raise exc.MissingDataError(msg)

            set_to_be_copied = set_instance.copy_from

            if set_to_be_copied in self.sets and \
                    self.sets[set_to_be_copied].data is not None and \
                    isinstance(self.sets[set_to_be_copied].data, pd.DataFrame):

                set_instance.data = self.sets[set_to_be_copied].data.copy()
                set_instance.data.columns = [
                    header[0]
                    for header in set_instance.table_headers.values()
                ]
            else:
                msg = f"Table '{set_to_be_copied}' not included in " \
                    "the defined Sets, or data not defined or defined in the " \
                    "wrong format. Check set name and data."
                self.logger.error(msg)
                raise exc.SettingsError(msg)

    def load_coordinates_to_data_index(self) -> None:
        """
        Populates the 'coordinates_values' dictionary of each data table in the 
        index with coordinate items from corresponding sets based on headers 
        defined in 'coordinates_headers'.
        This method essentially maps set items to their respective data tables, 
        facilitating direct access to these items for operations that require 
        context-specific data, such as data processing or analysis tasks.
        Ensures that each data table's coordinates are updated with actual items 
        from the sets as specified in the table's coordinate headers.
        """
        self.logger.debug("Loading variable coordinates to Index.data.")

        for table in self.data.values():
            for set_key, set_header in table.coordinates_headers.items():
                if set_key in self.sets:
                    table.coordinates_values[set_header] = self.sets[set_key].set_items
                else:
                    msg = f"Set key '{set_key}' not found in sets while " \
                        "loading coordinates"
                    self.logger.error(msg)
                    raise exc.MissingDataError(msg)

    def load_all_coordinates_to_variables_index(self) -> None:
        """
        Populates the 'coordinates' attribute of each variable in the index with 
        actual set items. This method utilizes 'coordinates_info' from each variable 
        to retrieve and assign corresponding set items from the index's sets.
        The method ensures that all variables are enriched with complete and correct 
        coordinate data, linking directly to the related set items based on 
        specified coordinates headers.

        Raises:
            SettingsError: If a set key specified in the variable's coordinates_info 
                does not exist in the Index sets, indicating a configuration or 
                data entry error.
        """
        self.logger.debug("Loading variable coordinates to Index.variables.")

        for var_key, variable in self.variables.items():

            # Replicate coordinates_info with inner values as None
            # to prepare the structure
            coordinates: Dict[str, Dict[str, Optional[List[str]]]] = {
                category: {key: None for key in coord_values}
                for category, coord_values in variable.coordinates_info.items()
            }

            # Populate the coordinates with actual set names and values
            # from the index's sets
            for category, coord_dict in coordinates.items():
                for coord_key in coord_dict:
                    set_instance = self.sets.get(coord_key)

                    if set_instance:
                        coordinates[category][coord_key] = set_instance.set_items
                    else:
                        msg = f"Set key '{coord_key}' not found in Index set for " \
                            f"variable '{var_key}'."
                        self.logger.error(msg)
                        raise exc.SettingsError(msg)

            variable.coordinates = coordinates

    def filter_coordinates_in_variables_index(self) -> None:
        """
        Filters the coordinate data for variables based on predefined filter 
        conditions stored in sets. This process adjusts the coordinate values 
        for 'rows', 'cols', and 'intra' categories by applying the filters 
        specified in the sets' filter headers.
        This method modifies the variable's coordinate data directly, ensuring 
        that only relevant items that meet the filter conditions are retained.
        """
        self.logger.debug(
            "Filtering variables coordinates in Index.variables.")

        for variable in self.variables.values():
            variable: Variable

            for coord_category, coord_dict in variable.coordinates.items():
                assert isinstance(coord_dict, dict), \
                    f"Expected dict, got {type(coord_dict)} instead."

                coord_key = next(iter(coord_dict), None)

                if coord_key is None:
                    continue

                if not self.sets.get(coord_key):
                    msg = f"Set key '{coord_key}' not found in sets."
                    self.logger.error(msg)
                    raise exc.MissingDataError(msg)

                set_filters_headers = self.sets[coord_key].set_filters_headers

                if not set_filters_headers:
                    continue

                var_coord_filter: Dict = getattr(
                    variable, coord_category, {}
                ).get('filters', {})

                var_coord_filter = {
                    set_filters_headers[num]: var_coord_filter[num]
                    for num in set_filters_headers.keys()
                    if num in var_coord_filter.keys()
                }

                # only rows, cols and intra problem sets can be filtered
                if var_coord_filter and coord_category in ['rows', 'cols', 'intra']:
                    set_data = self.sets[coord_key].data.copy()

                    for column, conditions in var_coord_filter.items():
                        if isinstance(conditions, list):
                            set_data = set_data[
                                set_data[column].isin(conditions)
                            ]
                        else:
                            set_data = set_data[
                                set_data[column] == conditions
                            ]

                    items_column_header = self.sets[coord_key].set_name_header
                    variable.coordinates[coord_category][coord_key] = \
                        list(set_data[items_column_header])

    def map_vars_aggregated_dims(self) -> None:
        """
        Maps aggregated dimensions for variables identified as constants. This 
        process involves linking each variable's dimension sets to specific 
        aggregated data mappings based on configurations in the sets.
        For each constant variable, the method looks up the corresponding sets, 
        extracts necessary header information, and applies any defined aggregation 
        logic. If successful, the aggregation map is stored within the variable's 
        attributes, enhancing the variable's utility in aggregated data contexts.
        Variables not marked as 'constant' are skipped. If any referenced set 
        is missing or improperly configured, the mapping process for that variable
        is aborted.
        """
        self.logger.debug(
            "Identifying aggregated dimensions for constants coordinates.")

        for variable in self.variables.values():
            variable: Variable

            if not variable.type == 'constant':
                continue

            set_items_agg_map = pd.DataFrame()
            set_items = pd.DataFrame()

            for dim_key, dim in variable.dims_sets.items():
                dim_set: SetTable = getattr(self.sets, str(dim), None)

                if not dim_set:
                    break

                key_name = Constants.get('_STD_NAME_HEADER')
                key_aggregation = Constants.get('_STD_AGGREGATION_HEADER')

                if dim_set.table_headers is not None:
                    name_header_filter = dim_set.table_headers.get(
                        key_name, [None])[0]

                    aggregation_header_filter = dim_set.table_headers.get(
                        key_aggregation, [None])[0]
                else:
                    name_header_filter = None
                    aggregation_header_filter = None

                if dim_set.data is None:
                    msg = f"Data for set '{dim_set}' are missing."
                    self.logger.error(msg)
                    raise exc.MissingDataError(msg)

                if key_aggregation in dim_set.table_headers:
                    set_items_agg_map = dim_set.data[[
                        name_header_filter, aggregation_header_filter]].copy()
                    set_items_agg_map.rename(
                        columns={name_header_filter: dim_key},
                        inplace=True,
                    )
                    # renaming column representing filtered dimension
                else:
                    set_items = dim_set.data[[name_header_filter]].copy()
                    set_items.rename(
                        columns={name_header_filter: dim_key},
                        inplace=True,
                    )

                if not set_items_agg_map.empty and set_items is not None:
                    if set(set_items_agg_map[aggregation_header_filter]) == \
                            set(set_items.values.flatten()):

                        # renaming column representing non-filtered dimension
                        set_items_agg_map.rename(
                            columns={
                                aggregation_header_filter: set_items.columns[0]
                            },
                            inplace=True,
                        )

                        # filtering rows and cols (in case of filtered vars)
                        filter_dict = {
                            'rows': variable.dims_items[0],
                            'cols': variable.dims_items[1],
                        }
                        set_items_agg_map_filtered = util.filter_dataframe(
                            df_to_filter=set_items_agg_map,
                            filter_dict=filter_dict,
                        )

                        variable.related_dims_map = set_items_agg_map_filtered
                        break
