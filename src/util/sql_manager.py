import sqlite3
import pandas as pd

from typing import List, Dict
from pathlib import Path
from log_exc.logger import Logger


class DatabaseSQL:

    def __init__(
        self,
        logger: Logger,
        database_sql_path: Path,
    ) -> None:

        self.logger = logger.getChild(__name__)
        self.logger.info(f"Generation of '{self}' object.")

        self.database_sql_path = database_sql_path

        self.connection = None
        self.cursor = None
        self.open_connection()

    def __repr__(self):
        class_name = type(self).__name__
        return f'{class_name}'

    def open_connection(self) -> None:
        self.connection = sqlite3.connect(f'{self.database_sql_path}')
        self.cursor = self.connection.cursor()
        self.logger.info(f"'{self}' connection opened.")

    def close_connection(self) -> None:
        self.connection.close()
        self.logger.info(f"'{self}' connection closed.")

    def drop_table(self, table_name: str) -> None:
        query = f"DROP TABLE {table_name}"
        self.cursor.execute(query)
        self.connection.commit()
        self.logger.debug(f"Table '{table_name}' deleted.")

    def create_table(
            self,
            table_name: str,
            table_fields: Dict[str, List[str]],
    ) -> None:

        if table_name in self.get_existing_tables():
            self.logger.debug(
                f"Table '{table_name}' already exists.")

            user_input = input(
                f"Delete and overwrite '{table_name}'? (y/[n]): ")
            if user_input.lower() != 'y':
                self.logger.debug(f"Table '{table_name}' not owerwritten.")
                return
            else:
                self.drop_table(table_name=table_name)
                fields_str = ", ".join(
                    [f'{field_name} {field_type}'
                     for field_name, field_type in table_fields.values()])

                query = f'CREATE TABLE {table_name}({fields_str})'

                try:
                    self.cursor.execute(query)
                    self.connection.commit()
                    self.logger.debug(f"Table '{table_name}' created.")
                except sqlite3.OperationalError as error_msg:
                    self.logger.error(error_msg)
                    raise sqlite3.OperationalError(error_msg)

    def get_existing_tables(self) -> List[str]:
        query = "SELECT name FROM sqlite_master WHERE type='table'"
        self.cursor.execute(query)
        tables = self.cursor.fetchall()
        return [table[0] for table in tables]

    def get_table_fields(
            self,
            table_name: str
    ) -> Dict[str, str]:

        table_fields = {}
        query = f"PRAGMA table_info('{table_name}')"
        self.cursor.execute(query)
        result = self.cursor.fetchall()
        table_fields['labels'] = [row[1] for row in result]
        table_fields['types'] = [row[2] for row in result]
        return table_fields

    def count_table_data_entries(
            self,
            table_name: str
    ) -> int:
        self.cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
        return self.cursor.fetchone()[0]

    def delete_table_entries(self, table_name: str) -> None:
        num_entries = self.count_table_data_entries(table_name=table_name)
        self.cursor.execute(f"DELETE FROM {table_name}")
        self.logger.debug(
            f"{num_entries} rows deleted from table '{table_name}'")

    def dataframe_to_table(
            self,
            table_name: str,
            dataframe: pd.DataFrame,
    ) -> None:

        table_fields = self.get_table_fields(table_name=table_name)

        if not dataframe.columns.tolist() == table_fields['labels']:
            error = f"Dataframe and table {table_name} headers mismatch."
            self.logger.error(error)
            raise ValueError(error)

        num_entries = self.count_table_data_entries(table_name=table_name)
        if num_entries > 0:
            confirm = input(
                f"Table {table_name} already has {num_entries} rows. Do you \
                    want to delete existing data and insert new data? (y/[n])"
            )
            if confirm.lower() != 'y':
                return
            else:
                self.delete_table_entries(table_name=table_name)

        data = [tuple(row) for row in dataframe.values.tolist()]

        placeholders = ', '.join(['?'] * len(dataframe.columns))
        query = f"INSERT INTO {table_name} VALUES ({placeholders})"

        try:
            self.cursor.executemany(query, data)
            self.logger.debug(
                f"{len(data)} rows inserted into table '{table_name}'"
            )
        except sqlite3.IntegrityError as error:
            if str(error).startswith('UNIQUE'):
                error = f"Data already exists in database {table_name}."
            self.logger.error(error)

        self.connection.commit()

    def table_to_dataframe(
            self,
            table_name: str,
    ) -> pd.DataFrame:

        query = f"SELECT * FROM {table_name}"
        self.cursor.execute(query)
        data = self.cursor.fetchall()

        table_fields = self.get_table_fields(table_name=table_name)
        df = pd.DataFrame(data, columns=table_fields['labels'])

        return df