"""
core.py

@author: Matteo V. Rocco
@institution: Politecnico di Milano

This module embeds the generation of main classes of the package, allowing to 
operate across them. 
It facilitates interactions among data tables, databases, problems, and 
logging mechanisms, enabling comprehensive management and operations within 
the modeling environment.
"""

import os
from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import cvxpy as cp

from esm.backend import problem
from esm.backend.data_table import DataTable
from esm.backend.database import Database
from esm.backend.index import Index, Variable
from esm.backend.problem import Problem
from esm.log_exc import exceptions as exc
from esm.log_exc.logger import Logger
from esm.constants import Constants
from esm.support import util
from esm.support.file_manager import FileManager
from esm.support.sql_manager import SQLManager, db_handler


class Core:
    """
    Core class defines the interactions among various components of the package, 
    such as data tables, databases, and problem-solving mechanisms. It serves 
    as the central management point for initializing, processing, and 
    executing model operations.

    Attributes:
        logger (Logger): Logger object for logging information.
        files (FileManager): FileManager object for file operations.
        settings (Dict[str, str]): Settings for various file paths and configurations.
        paths (Dict[str, Path]): Paths to various directories and files used in the model.
        sqltools (SQLManager): SQLManager object for database interactions.
        index (Index): Index object for managing data table and variable indices.
        database (Database): Database object for database operations.
        problem (Problem): Problem object for problem definitions and operations.

    Args:
        logger (Logger): Logger object for logging information.
        files (FileManager): FileManager object for file operations.
        settings (Dict[str, str]): Settings dictionary containing configuration details.
        paths (Dict[str, Path]): Paths dictionary containing paths used in the model.
    """

    def __init__(
            self,
            logger: Logger,
            files: FileManager,
            settings: Dict[str, str],
            paths: Dict[str, Path],
    ) -> None:
        """
        Initializes the Core class with the necessary components and settings.
        This class serves as the main orchestrator for the application, managing 
        the interactions between the various components.

        Args:
            logger (Logger): An instance of Logger for logging information and 
                error messages.
            files (FileManager): An instance of FileManager for managing 
                file-related operations.
            settings (Dict[str, str]): A dictionary containing configuration 
                settings for the application.
            paths (Dict[str, Path]): A dictionary containing paths used throughout 
                operations, such as for files and directories.

        Returns:
            None

        Notes:
            The logger is initialized with a child logger using the name of the 
                current module.
            The SQLManager, Index, Database, and Problem instances are initialized 
                with the provided logger, files, paths, and settings.
        """
        self.logger = logger.get_child(__name__)
        self.logger.debug(f"'{self}' object initialization...")

        self.files = files
        self.settings = settings
        self.paths = paths

        self.sqltools = SQLManager(
            logger=self.logger,
            database_path=self.paths['sqlite_database'],
            database_name=self.settings['sqlite_database_file'],
        )

        self.index = Index(
            logger=self.logger,
            files=self.files,
            paths=self.paths,
        )

        self.database = Database(
            logger=self.logger,
            files=self.files,
            paths=self.paths,
            sqltools=self.sqltools,
            settings=self.settings,
            index=self.index,
        )

        self.problem = Problem(
            logger=self.logger,
            files=self.files,
            paths=self.paths,
            settings=self.settings,
            index=self.index
        )

        self.logger.debug(f"'{self}' initialized.")

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}'

    def initialize_problems_variables(self) -> None:
        """
        Initializes and generates data structures for handling problem variables.
        This method iterates over each data table and variable in the index. 
        For each endogenous data table or data table with a dictionary type, 
        it generates a coordinates DataFrame and a cvxpy variable. For each 
        constant variable, it generates the variable's data directly. For each 
        exogenous or endogenous variable, it generates a DataFrame and stores it 
        in the variable's data attribute. For each variable with a dictionary 
        type, it generates a DataFrame for each problem and stores them in the 
        variable's data attribute as a dictionary.

        Returns:
            None

        Raises:
            SettingsError: If a variable's type is not 'constant', 'exogenous', 
                'endogenous', or a dictionary.

        Notes:
            The method logs information about the generation process.
            The cvxpy variables for endogenous data tables are created using the 
                'create_cvxpy_variable' method of the Problem instance.
            The data for constant variables is generated using the 
                'generate_constant_data' method of the Problem instance.
            The DataFrames for exogenous and endogenous variables are generated 
                using the 'generate_vars_dataframe' method of the Problem instance.
        """
        self.logger.debug(
            "Generating data structures for endogenous data tables "
            "(cvxpy objects, filters dict for data tables).")

        # generate dataframes and cvxpy var for endogenous data tables
        # and for variables whth type defined by problem linking logic
        for data_table_key, data_table in self.index.data.items():

            if data_table.type == 'endogenous' or \
                    isinstance(data_table.type, dict):

                self.logger.debug(
                    "Generating variable dataframe and cvxpy variable "
                    f"for endogenous data table '{data_table_key}'.")

                data_table.generate_coordinates_dataframes(
                    sets_split_problems=self.index.sets_split_problem_dict
                )

                if isinstance(data_table.coordinates_dataframe, pd.DataFrame):

                    cvxpy_var = self.problem.create_cvxpy_variable(
                        var_type='endogenous',
                        shape=(data_table.table_length, 1),
                        name=data_table_key,
                    )

                # in case of problem with sets split, multiple endogenous variables
                # are created and stored in a dictionary
                elif isinstance(data_table.coordinates_dataframe, dict):

                    cvxpy_var = {}

                    for problem_key, variable_df in data_table.coordinates_dataframe.items():

                        cvxpy_var[problem_key] = self.problem.create_cvxpy_variable(
                            var_type='endogenous',
                            shape=(len(variable_df), 1),
                            name=f"{data_table_key}_{problem_key}",
                        )

                data_table.cvxpy_var = cvxpy_var

        # generating variables dataframes with cvxpy var and filters dictionary
        # (endogenous vars will be sliced from existing cvxpy var in data table)
        self.logger.debug(
            "Generating data structures for all variables and constants.")

        for var_key, variable in self.index.variables.items():

            # for constants, values are directly generated (no dataframes needed)
            if variable.type == 'constant':
                variable.data = self.problem.generate_constant_data(
                    variable_name=var_key,
                    variable=variable
                )

            # for variables whose type is univocally defined, only one data structure
            # is generated and stored in variable.data
            elif variable.type in ['exogenous', 'endogenous']:
                variable.data = self.problem.generate_vars_dataframe(
                    variable_name=var_key,
                    variable=variable
                )

            # for variable whose type varies depending on the problem, both
            # endogenous/exogenous variable dataframes are stored in
            # variable.data defined as a dictionary
            elif isinstance(variable.type, dict):
                variable.data = {}

                for problem_key, problem_var_type in variable.type.items():
                    variable.data[problem_key] = self.problem.generate_vars_dataframe(
                        variable_name=var_key,
                        variable=variable,
                        variable_type=problem_var_type,
                    )

            else:
                setup_file = Constants.get('_SETUP_FILES')[1]
                msg = f"Variable type '{variable.type}' not allowed. Check " \
                    "definition of variable types in the model configuration " \
                    f"file '{setup_file}'."
                self.logger.error(msg)
                raise exc.SettingsError(msg)

    def define_mathematical_problems(
            self,
            force_overwrite: bool = False,
    ) -> None:
        """
        Defines and initializes numerical problems based on the loaded symbolic 
        definitions.
        This method loads the symbolic problem from a file and generates numerical 
        problems based on the symbolic definitions. The method can optionally 
        overwrite existing problem definitions.

        Args:
            force_overwrite (bool, optional): If True, forces the redefinition 
                of problems even if they already exist. Defaults to False.

        Returns:
            None

        Notes:
            The method logs information about the problem definition process.
            The symbolic problem is loaded using the 'load_symbolic_problem_from_file' 
                method of the Problem instance.
            The numerical problems are generated using the 'generate_numerical_problems' 
                method of the Problem instance.
        """
        self.logger.debug(
            "Load symbolic problem, initialize dataframes with cvxpy problem.")

        self.problem.load_symbolic_problem_from_file(force_overwrite)
        self.problem.generate_numerical_problems(force_overwrite)

    def solve_numerical_problems(
            self,
            solver: str,
            verbose: bool,
            integrated_problems: bool,
            force_overwrite: bool,
            maximum_iterations: Optional[int] = None,
            numerical_tolerance: Optional[float] = None,
            **kwargs: Any,
    ) -> None:
        """
        Solves all defined numerical problems using the specified solver and 
        verbosity settings.
        This method checks if numerical problems have been defined and if they 
        have already been solved. If the problems have not been solved or if 
        'force_overwrite' is True, the method solves the problems using the 
        specified solver. The method can solve the problems individually or as 
        an integrated problem, depending on the 'integrated_problems' setting.

        Args:
            solver (str): The solver to use for solving the problems.
            verbose (bool): If True, enables verbose output during problem solving.
            integrated_problems (bool): If True, solves the problems as an 
                integrated problem. If False, solves the problems individually.
            force_overwrite (bool): If True, forces the re-solution of problems 
                even if they have already been solved.
            maximum_iterations (Optional[int], optional): The maximum number of 
                iterations for the solver. Defaults to None.
            numerical_tolerance (Optional[float], optional): The numerical 
                tolerance for the solver. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the solver.

        Returns:
            None

        Raises:
            OperationalError: If numerical problems have not been defined.

        Notes:
            The method logs information about the problem solving process.
            The problems are solved using the 'solve_problems' or 
                'solve_integrated_problems' method of the Problem instance, 
                depending on the 'integrated_problems' setting.
            The method fetches the problem status after solving the problems.
        """
        if self.problem.numerical_problems is None:
            msg = "Numerical problems must be defined first."
            self.logger.warning(msg)
            raise exc.OperationalError(msg)

        if self.problem.problem_status is not None:
            if not force_overwrite:
                self.logger.warning("Numeric problems already solved.")
                user_input = input("Solve again numeric problems? (y/[n]): ")

                if user_input.lower() != 'y':
                    self.logger.info(
                        "Numeric problem NOT solved.")
                    return

            self.logger.info(
                "Solving numeric problem and overwriting existing "
                "variables numerical values.")

        if not integrated_problems:
            self.problem.solve_problems(
                solver=solver,
                verbose=verbose,
                **kwargs
            )
        else:
            self.solve_integrated_problems(
                solver=solver,
                verbose=verbose,
                numerical_tolerance=numerical_tolerance,
                maximum_iterations=maximum_iterations,
                **kwargs,
            )

        self.problem.fetch_problem_status()

    def data_to_cvxpy_exogenous_vars(self) -> None:
        """
        Fetches data from the SQLite database and assigns it to cvxpy exogenous 
        variables.
        This method iterates over each variable in the index. If the variable's 
        type is not 'endogenous' or 'constant', the method fetches the variable's 
        data from the SQLite database and assigns it to the cvxpy variable. 
        The method handles variables whose type is defined by the problem separately.

        Returns:
            None

        Raises:
            TypeError: If a passed item is not an instance of the 'Variable' class.
            MissingDataError: If no data or related table is defined for a variable, 
                or if the data for a variable contains non-allowed values types.

        Notes:
            The method logs information about the data fetching process.
            The method uses a context manager to handle the database connection.
            The data is fetched using the 'filtered_table_to_dataframe' method 
                of the SQLTools instance.
            The data is assigned to the cvxpy variable using the 'data_to_cvxpy_variable' 
                method of the Problem instance.
        """
        self.logger.debug(
            f"Fetching data from '{self.settings['sqlite_database_file']}' "
            "to cvxpy exogenous variables.")

        with db_handler(self.sqltools):
            for var_key, variable in self.index.variables.items():

                if not isinstance(variable, Variable):
                    msg = "Passed item is not a 'Variable' class instance."
                    self.logger.error(msg)
                    raise TypeError(msg)

                if variable.type in ['endogenous', 'constant']:
                    continue

                self.logger.debug(
                    f"Fetching data from table '{var_key}' "
                    "to cvxpy exogenous variable.")

                filter_header = Constants.get('_FILTER_DICT_HEADER')
                cvxpy_var_header = Constants.get('_CVXPY_VAR_HEADER')
                values_header = Constants.get('_STD_VALUES_FIELD')['values'][0]
                id_header = Constants.get('_STD_ID_FIELD')['id'][0]
                allowed_values_types = Constants.get('_ALLOWED_VALUES_TYPES')

                err_msg = []

                if variable.data is None:
                    err_msg.append(
                        f"No data defined for variable '{var_key}'.")

                if variable.related_table is None:
                    err_msg.append(
                        f"No related table defined for variable '{var_key}'.")

                if err_msg:
                    self.logger.error("\n".join(err_msg))
                    raise exc.MissingDataError("\n".join(err_msg))

                # for variables whose type is end/exo depending on the problem,
                # fetch exogenous variable data.
                if isinstance(variable.type, dict):
                    problem_keys = util.find_dict_keys_corresponding_to_value(
                        variable.type, 'exogenous')
                else:
                    problem_keys = [None]

                for problem_key in problem_keys:

                    if problem_key:
                        variable_data = variable.data[problem_key]
                    else:
                        variable_data = variable.data

                    for row in variable_data.index:
                        # get raw data from database
                        raw_data = self.database.sqltools.filtered_table_to_dataframe(
                            table_name=variable.related_table,
                            filters_dict=variable_data[filter_header][row])

                        # check if variable data are int or float
                        non_numeric_ids = util.find_non_allowed_types(
                            dataframe=raw_data,
                            allowed_types=allowed_values_types,
                            target_col_header=values_header,
                            return_col_header=id_header,
                        )

                        if non_numeric_ids:
                            msg = f"Data for variable '{var_key}' in table " \
                                f"'{variable.related_table}' contains " \
                                f"non-allowed values types in rows: " \
                                f"{non_numeric_ids}."
                            self.logger.error(msg)
                            raise exc.MissingDataError(msg)

                        # pivoting and reshaping data to fit variables
                        pivoted_data = variable.reshaping_sqlite_table_data(
                            data=raw_data
                        )

                        self.problem.data_to_cvxpy_variable(
                            cvxpy_var=variable_data[cvxpy_var_header][row],
                            data=pivoted_data
                        )

    def cvxpy_endogenous_data_to_database(
            self,
            operation: str,
            suppress_warnings: bool = False,
    ) -> None:
        """
        Exports data from cvxpy endogenous variables back to data tables in the 
        SQLite database.
        This method iterates over each data table in the index. If the table's 
        type is not 'exogenous' or 'constant', the method exports the data from 
        the cvxpy variable to the corresponding data table in the SQLite database. 

        Parameters:
            operation (str): The type of database operation to perform.
            suppress_warnings (bool, optional): If True, suppresses warnings 
                during the data export process. Defaults to False.

        Returns:
            None

        Raises:
            TypeError: If a passed item is not an instance of the 'DataTable' class.
            OperationalError: If no coordinates DataFrame is defined for a data table.

        Notes:
            The method logs information about the data export process.
            The method uses a context manager to handle the database connection.
            The data is exported using the 'dataframe_to_table' method of the 
                SQLTools instance.
        """
        self.logger.debug(
            "Exporting data from cvxpy endogenous variable (in data table) "
            f"to SQLite database '{self.settings['sqlite_database_file']}' ")

        values_headers = Constants.get('_STD_VALUES_FIELD')['values'][0]

        with db_handler(self.sqltools):
            for data_table_key, data_table in self.index.data.items():

                if not isinstance(data_table, DataTable):
                    msg = "Passed item is not a 'DataTable' class instance."
                    self.logger.error(msg)
                    raise TypeError(msg)

                if data_table.type in ['exogenous', 'constant']:
                    continue

                self.logger.debug(
                    "Exporting data from cvxpy variable to the related "
                    f"data table '{data_table_key}'. ")

                if isinstance(data_table.coordinates_dataframe, pd.DataFrame):
                    data_table_dataframe = data_table.coordinates_dataframe

                elif isinstance(data_table.coordinates_dataframe, dict):
                    data_table_dataframe = pd.concat(
                        data_table.coordinates_dataframe.values(),
                        ignore_index=True
                    )

                if not util.add_column_to_dataframe(
                    dataframe=data_table_dataframe,
                    column_header=values_headers,
                ):
                    if self.settings['log_level'] == 'debug' or \
                            not suppress_warnings:
                        self.logger.warning(
                            f"Column '{values_headers}' already exists in data "
                            f"table '{data_table_key}'")

                if data_table.cvxpy_var is None:
                    if self.settings['log_level'] == 'debug' or \
                            not suppress_warnings:
                        self.logger.warning(
                            f"No data available in cvxpy variable '{data_table_key}'")
                    continue

                if isinstance(data_table.cvxpy_var, dict):

                    cvxpy_var_values_list = []
                    for cvxpy_var in data_table.cvxpy_var.values():
                        cvxpy_var: cp.Variable
                        cvxpy_var_values_list.append(cvxpy_var.value)

                    cvxpy_var_data = np.vstack(cvxpy_var_values_list)

                else:
                    cvxpy_var_data = data_table.cvxpy_var.value

                data_table_dataframe[values_headers] = cvxpy_var_data

                self.sqltools.dataframe_to_table(
                    table_name=data_table_key,
                    dataframe=data_table_dataframe,
                    operation=operation,
                    suppress_warnings=suppress_warnings,
                )

    def check_results_as_expected(
            self,
            values_relative_diff_tolerance: float,
    ) -> None:
        """
        Checks if the results of the current database match the expected results.
        This method uses the 'check_databases_equality' method to compare the 
        current database with a test database. The test database is specified 
        by the 'sqlite_database_file_test' setting and is located in the model 
        directory.

        Args:
            values_relative_diff_tolerance (float): The relative difference 
                tolerance (%) to use when comparing the databases. 

        Raises:
            OperationalError: If the connection or cursor of the database to be 
                checked are not initialized.
            ModelFolderError: If the test database does not exist or is not 
                correctly named.
            ResultsError: If the databases are not identical in terms of table 
                presence, structure, or contents.
        """
        with db_handler(self.sqltools):
            self.sqltools.check_databases_equality(
                other_db_dir_path=self.paths['model_dir'],
                other_db_name=self.settings['sqlite_database_file_test'],
                tolerance_percentage=values_relative_diff_tolerance,
            )

    def solve_integrated_problems(
            self,
            solver: str,
            verbose: bool,
            numerical_tolerance: Optional[float] = None,
            maximum_iterations: Optional[int] = None,
            **kwargs: Any,
    ) -> None:
        """
        Solves all defined numerical problems iteratively using the specified 
        solver and verbosity settings.
        This method iteratively solves the problems until the relative difference 
        between the solutions in consecutive iterations is less than the specified 
        numerical tolerance or until the maximum number of iterations is reached. 
        The method handles the database operations required for each iteration, 
        including updating the data for exogenous variables and exporting the 
        data for endogenous variables.

        Parameters:
            solver (str): The solver to use for solving the problems.
            verbose (bool): If True, enables verbose output during problem solving.
            numerical_tolerance (Optional[float], optional): The numerical tolerance 
                for the solver. Defaults to None.
            maximum_iterations (Optional[int], optional): The maximum number of 
                iterations for the solver. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the solver.

        Returns:
            None

        Notes:
            The method logs information about the problem solving process.
            The problems are solved using the 'solve_problems' method of the 
                Problem instance.
            The data for exogenous variables is updated using the 
                'data_to_cvxpy_exogenous_vars' method.
            The data for endogenous variables is exported using the 
                'cvxpy_endogenous_data_to_database' method.
            The method calculates the relative difference between the solutions 
                in consecutive iterations using the 'get_tables_values_relative_difference' 
                method of the SQLTools instance.
        """
        if maximum_iterations is None:
            maximum_iterations = Constants.get(
                '_MAXIMUM_ITERATIONS_MODEL_COUPLING')

        if numerical_tolerance is None:
            numerical_tolerance = Constants.get(
                '_TOLERANCE_MODEL_COUPLING_CONVERGENCE')

        sqlite_db_path = self.paths['model_dir']
        sqlite_db_file_name = self.settings['sqlite_database_file']

        base_name, extension = os.path.splitext(sqlite_db_file_name)
        sqlite_db_file_name_previous = f"{base_name}_previous_iter{extension}"

        iter_count = 0

        tables_to_check = [
            table_key for table_key in self.index.data.keys()
            if self.index.data[table_key].type not in ['exogenous', 'constant']
        ]

        while True:

            try:
                iter_count += 1
                if iter_count > maximum_iterations:
                    self.logger.warning(
                        f"Maximum number of iterations reached before reaching "
                        "convergence to numerical_tolerance.")
                    break

                self.logger.info(f"=====================")
                self.logger.info(f"Iteration number '{iter_count}'")

                if iter_count > 1:
                    self.data_to_cvxpy_exogenous_vars()

                self.files.copy_file_to_destination(
                    path_destination=sqlite_db_path,
                    path_source=sqlite_db_path,
                    file_name=sqlite_db_file_name,
                    file_new_name=sqlite_db_file_name_previous,
                    force_overwrite=True,
                )

                self.problem.solve_problems(
                    solver=solver,
                    verbose=verbose,
                    **kwargs
                )

                self.cvxpy_endogenous_data_to_database(
                    operation='update',
                    suppress_warnings=True,
                )

                with db_handler(self.sqltools):
                    relative_difference = \
                        self.sqltools.get_tables_values_relative_difference(
                            other_db_dir_path=sqlite_db_path,
                            other_db_name=sqlite_db_file_name_previous,
                            tables_names=tables_to_check,
                        )

                relative_difference_above = {
                    table: value
                    for table, value in relative_difference.items()
                    if value > numerical_tolerance
                }

                if relative_difference_above:
                    self.logger.info(
                        "Data tables with highest relative difference above "
                        f"treshold ({numerical_tolerance}):"
                    )
                    for table, value in relative_difference_above.items():
                        self.logger.info(
                            f"Data table '{table}': {round(value, 5)}")
                else:
                    self.logger.info("Numerical convergence reached.")
                    break

            finally:
                self.files.erase_file(
                    dir_path=sqlite_db_path,
                    file_name=sqlite_db_file_name_previous,
                    force_erase=True,
                    confirm=False,
                )
