from typing import Any, Dict, List
import re

import pandas as pd
import numpy as np
import cvxpy as cp

from src.support import constants
from src.log_exc import exceptions as exc
from src.log_exc.logger import Logger
from src.support import util
from src.support.file_manager import FileManager
from src.support.dotdict import DotDict
from src.backend.index import Index, Variable


class Problem:

    allowed_operators = constants._ALLOWED_OPERATORS

    def __init__(
            self,
            logger: Logger,
            files: FileManager,
            paths: Dict[str, str],
            settings: Dict[str, str],
            index: Index,
    ) -> None:

        self.logger = logger.getChild(__name__)
        self.logger.info(f"'{self}' object initialization...")

        self.files = files
        self.settings = settings
        self.index = index
        self.paths = paths

        self.symbolic_problem = None
        self.numeric_problems = None
        self.model_run = None

        self.logger.info(f"'{self}' object initialized.")

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}'

    def create_cvxpy_variable(
        self,
        type: str,
        shape: List[int],
        name: str,
    ) -> cp.Variable | cp.Parameter:

        if type == 'endogenous':
            return cp.Variable(shape=shape, name=name)
        elif type == 'exogenous':
            return cp.Parameter(shape=shape, name=name)
        elif type == 'constant':
            pass  # tbd
        else:
            error = f"Unsupported variable type: {type}"
            self.logger.error(error)
            raise ValueError(error)

    def data_to_cvxpy_variable(
            self,
            cvxpy_var: cp.Parameter,
            data: pd.DataFrame | np.ndarray,
    ) -> None:

        if not isinstance(cvxpy_var, cp.Parameter):
            error = "Data can only be assigned to exogenous variables."
            self.logger.error(error)
            raise ValueError(error)

        if isinstance(data, pd.DataFrame):
            cvxpy_var.value = data.values
        elif isinstance(data, np.ndarray):
            cvxpy_var.value = data
        else:
            error = "Supported data formats: pandas DataFrame or a numpy array."
            self.logger.error(error)
            raise ValueError(error)

    def generate_vars_dataframe(
            self,
            variable: Variable,
    ) -> pd.DataFrame:
        """For a Variable object, generates a Pandas DataFrame with the  
        hierarchy structure, the related cvxpy variables and the dictionary to
        filter the sql table for fetching data.
        """

        headers = {
            'cvxpy': constants._CVXPY_VAR_HEADER,
            'filter': constants._FILTER_DICT_HEADER,
        }

        self.logger.debug(
            f"Generating variable '{variable.symbol}' dataframe "
            "(cvxpy object, filter dictionary).")

        sets_parsing_hierarchy = variable.sets_parsing_hierarchy.values()

        var_data = util.unpivot_dict_to_dataframe(
            data_dict=variable.coordinates,
            key_order=sets_parsing_hierarchy
        )

        for item in headers.values():
            util.add_column_to_dataframe(
                dataframe=var_data,
                column_header=item,
                column_values=None,
            )

        for row in var_data.index:

            var_data.at[row, headers['cvxpy']] = \
                self.create_cvxpy_variable(
                    type=variable.type,
                    shape=variable.shape_size,
                    name=variable.symbol + str(variable.shape))

            var_filter = {}

            for header in var_data.loc[row].index:

                if sets_parsing_hierarchy is not None and \
                        header in sets_parsing_hierarchy:
                    var_filter[header] = [var_data.loc[row][header]]

                elif header == headers['cvxpy']:
                    for dim in variable.shape:
                        if isinstance(dim, int):
                            pass
                        elif isinstance(dim, str):
                            dim_header = variable.table_headers[dim][0]
                            var_filter[dim_header] = variable.coordinates[dim_header]

                elif header == headers['filter']:
                    pass

                else:
                    msg = f"Variable 'data' dataframe headers mismatch."
                    self.logger.error(msg)
                    raise ValueError(msg)

            var_data.at[row, headers['filter']] = var_filter

        return var_data

    def load_symbolic_problem_from_file(self) -> None:

        problem_file_name = constants._SETUP_FILES['problem']

        if self.symbolic_problem is not None:
            self.logger.warning(f"Symbolic problem already loaded.")
            user_input = input(f"Update symbolic problem? (y/[n]): ")

            if user_input.lower() != 'y':
                self.logger.info(f"Symbolic problem NOT updated.")
                return
            else:
                self.logger.info(f"Symbolic problem updated.")
        else:
            self.logger.info(
                f"Loading symbolic problem from '{problem_file_name}' file.")

        symbolic_problem = self.files.load_file(
            file_name=problem_file_name,
            dir_path=self.paths['model_dir'],
        )

        self.symbolic_problem = DotDict(symbolic_problem)

    def parse_allowed_symbolic_vars(
            self,
            expression: str,
            non_allowed_tokens: List[str] = allowed_operators.keys(),
            standard_pattern: str = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
    ) -> List[str]:

        tokens = re.findall(
            pattern=standard_pattern,
            string=expression,
        )

        allowed_vars = [
            token for token in tokens
            if token not in non_allowed_tokens
        ]

        if not allowed_vars:
            self.logger.warning(
                "Empty list of allowed variables "
                f"for expression: {expression}")

        return allowed_vars

    def check_variables_attribute_equality(
            self,
            variables_subset: DotDict[str, Variable],
            attribute: str,
    ) -> None:

        first_variable = next(iter(variables_subset.values()))
        first_var_attr = getattr(first_variable, attribute)

        all_same_attrs = all(
            getattr(variable, attribute) == first_var_attr
            for variable in variables_subset.values()
        )

        if not all_same_attrs:
            var_subset_symbols = [
                getattr(variable, 'symbol')
                for variable in variables_subset
            ]

            msg = f"Attributes '{attribute}' mismatch in the passed " \
                f"variables subset {var_subset_symbols}."
            self.logger.error(msg)
            raise exc.ConceptualModelError(msg)

    def find_common_sets_intra_problem(
        self,
        variables_subset: DotDict[str, Variable],
    ) -> Dict[str, str]:

        sets_dicts_to_compare = {
            key: getattr(variable, 'sets_intra_problem')
            for key, variable in variables_subset.items()
        }

        sets_dicts_list = list(sets_dicts_to_compare.values())

        if all(
            d == sets_dicts_list[0]
            for d in sets_dicts_list[1:]
        ):
            return sets_dicts_list[0]
        else:
            msg = f"Passed variables subset have not " \
                "the same sets_intra_problem."
            self.logger.error(msg)
            raise exc.ConceptualModelError(msg)

    def generate_problems_dataframe(self):

        if self.numeric_problems is not None:
            self.logger.warning(f"Numeric problem already defined.")
            user_input = input(f"Overwrite numeric problem? (y/[n]): ")

            if user_input.lower() != 'y':
                self.logger.info(f"Numeric problem NOT overwritten.")
                return
            else:
                self.logger.info(f"Numeric problem overwritten.")
        else:
            self.logger.info(
                "Defining numeric problems based on symbolic problem.")

        headers = {
            'info': constants._PROBLEM_INFO_HEADER,
            'objective': constants._OBJECTIVE_HEADER,
            'constraints': constants._CONSTRAINTS_HEADER,
            'problem': constants._PROBLEM_HEADER,
            'status': constants._PROBLEM_STATUS_HEADER,
        }

        # per ora le colonne dei set hanno i nomi degli headers (s_Name, ...)
        # se si vuole mettere i nomi dei set (Scenarios) bisogna cambiarli anche
        # nelle tabelle sei set e delle variabili.
        dict_to_unpivot = {}
        for set_name, set_header in self.index.list_sets_split_problem.items():
            set_values = self.index.sets[set_name].data[set_header]
            dict_to_unpivot[set_header] = list(set_values)

        list_sets_split_problem = self.index.list_sets_split_problem.values()

        problems_data = util.unpivot_dict_to_dataframe(
            data_dict=dict_to_unpivot,
            key_order=list_sets_split_problem,
        )

        for item in headers.values():
            util.add_column_to_dataframe(
                dataframe=problems_data,
                column_header=item,
                column_values=None,
            )

        for problem_num in problems_data.index:

            problem_info = [
                problems_data.loc[problem_num][set_name]
                for set_name in list_sets_split_problem
            ]

            self.logger.debug(
                "Defining numeric problem for combination "
                f"of sets: {problem_info}.")

            problem_filter = problems_data.loc[
                [problem_num],
                list_sets_split_problem
            ]

            constraints = self.define_expressions(
                header_object=headers['constraints'],
                problem_filter=problem_filter
            )

            objective = sum(
                self.define_expressions(
                    header_object=headers['objective'],
                    problem_filter=problem_filter
                )
            )

            problem = cp.Problem(objective, constraints)

            problems_data.at[problem_num, headers['info']] = problem_info
            problems_data.at[problem_num, headers['constraints']] = constraints
            problems_data.at[problem_num, headers['objective']] = objective
            problems_data.at[problem_num, headers['problem']] = problem
            problems_data.at[problem_num, headers['status']] = None

        self.numeric_problems = problems_data

    def fetch_allowed_cvxpy_variables(
            self,
            variables_set_dict: DotDict[str, Variable],
            problem_filter: pd.DataFrame = None,
            set_intra_problem_header: str = None,
            set_intra_problem_value: str = None,
    ) -> Dict[str, cp.Parameter | cp.Variable]:

        allowed_variables = {}

        for variable in variables_set_dict.values():

            if problem_filter is not None:
                variable_data = pd.merge(
                    left=variable.data,
                    right=problem_filter,
                    on=list(problem_filter.columns),
                    how='inner'
                )
            else:
                variable_data = variable.data

            if set_intra_problem_header and set_intra_problem_value:
                cvxpy_variable = variable_data.loc[
                    variable_data[set_intra_problem_header] == set_intra_problem_value,
                    constants._CVXPY_VAR_HEADER,
                ].iloc[0]

            else:
                cvxpy_variable = variable_data.loc[constants._CVXPY_VAR_HEADER]

            allowed_variables[variable.symbol] = cvxpy_variable

        return allowed_variables

    def execute_cvxpy_code(
            self,
            expression: str,
            allowed_variables: Dict[str, cp.Parameter | cp.Variable],
            allowed_operators: Dict[str, str] = constants._ALLOWED_OPERATORS,
    ) -> Any:

        local_vars = {}

        try:
            exec(
                'output = ' + expression,
                {**allowed_operators, **allowed_variables},
                local_vars,
            )
        except SyntaxError:
            msg = "Error in parsing cvxpy expression: " \
                "check allowed variables and operators."
            self.logger.error(msg)
            raise exc.NumericalProblemError(msg)

        return local_vars['output']

    def define_expressions(
            self,
            header_object: str,
            problem_filter: pd.DataFrame,
    ) -> List[Any]:

        expressions = []

        for expression in self.symbolic_problem[header_object]:

            # get variables symbols in expression
            var_symbols_list = self.parse_allowed_symbolic_vars(expression)

            # define subset of variables in the expression
            vars_subset = DotDict({
                key: variable for key, variable in self.index.variables
                if variable.symbol in var_symbols_list
            })

            # if more than one var, check equality of sets_parsing_hierarchy
            if len(vars_subset) > 1:
                self.check_variables_attribute_equality(
                    variables_subset=vars_subset,
                    attribute='sets_parsing_hierarchy'
                )

            # look for intra-problem set (only one allowed for now)
            set_intra_problem = self.find_common_sets_intra_problem(
                variables_subset=vars_subset,
            )

            if set_intra_problem:
                set_key = list(set_intra_problem.keys())[0]
                set_header = list(set_intra_problem.values())[0]
                set_data = self.index.sets[set_key].data

                # parse values in intra-problem-set
                for value in set_data[set_header]:

                    # fetch allowed cvxpy variables
                    allowed_variables = self.fetch_allowed_cvxpy_variables(
                        variables_set_dict=vars_subset,
                        problem_filter=problem_filter,
                        set_intra_problem_header=set_header,
                        set_intra_problem_value=value,
                    )

                    # define constraint
                    cvxpy_expression = self.execute_cvxpy_code(
                        expression=expression,
                        allowed_variables=allowed_variables,
                    )

                    expressions.append(cvxpy_expression)

            else:
                allowed_variables = self.fetch_allowed_cvxpy_variables(
                    variables_set_dict=vars_subset,
                    problem_filter=problem_filter,
                )

                cvxpy_expression = self.execute_cvxpy_code(
                    expression=expression,
                    allowed_variables=allowed_variables,
                )

                expressions.append(cvxpy_expression)

        return expressions

    def solve_problem(
            self,
            problem: cp.Problem,
            solver: str = None,
            verbose: bool = True,
            **kwargs: Any,
    ) -> None:

        problem.solve(
            solver=solver,
            verbose=verbose,
            **kwargs
        )

    def solve_all_problems(
            self,
            solver: str,
            verbose: bool,
            **kwargs: Any,
    ) -> None:

        if self.numeric_problems is None or \
                self.numeric_problems[constants._PROBLEM_HEADER].isna().all():
            msg = "Numeric problems have to be defined first"
            self.logger.warning(msg)
            raise exc.OperationalError(msg)

        if self.model_run:
            self.logger.warning("Numeric problems already run.")
            user_input = input("Solve again numeric problems? (y/[n]): ")

            if user_input.lower() != 'y':
                self.logger.info(
                    "Numeric problem NOT solved.")
                return
            else:
                self.logger.info(
                    "Solving numeric problem and overwriting existing results.")

        for problem_num in self.numeric_problems.index:

            problem_info = self.numeric_problems.at[
                problem_num, constants._PROBLEM_INFO_HEADER]

            self.logger.debug(f"Solving problem: {problem_info}.")

            problem = self.numeric_problems.at[
                problem_num, constants._PROBLEM_HEADER]

            self.solve_problem(
                problem=problem,
                solver=solver,
                verbose=verbose,
                **kwargs,
            )

            problem_status = getattr(problem, 'status', None)
            self.numeric_problems.at[
                problem_num,
                constants._PROBLEM_STATUS_HEADER] = problem_status

        self.model_run = True
