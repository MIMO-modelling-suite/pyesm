import logging


class Logger:
    """Class defined for logging Model class and subclasses."""

    def __init__(
            self,
            logger_name: str = 'default_logger',
            log_level: str = 'DEBUG',
            log_format: str = 'minimal',
    ) -> None:

        formats = {
            'standard': '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            'minimal': '%(levelname)s | %(name)s | %(message)s'
        }

        self.log_format = log_format
        self.str_format = formats[log_format]

        self.logger = logging.getLogger(logger_name)

        if not self.logger.handlers:
            self.logger.setLevel(log_level)
            formatter = logging.Formatter(self.str_format)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(log_level)
            stream_handler.setFormatter(formatter)
            stream_handler.propagate = False
            self.logger.addHandler(stream_handler)

    def getChild(self, name: str) -> 'Logger':
        """Return a child Logger class with the specified name, inheriting
        properties of parent Logger class.

        Args:
            name (str): module __name__ where logger is generated. Notice that
                only last part of the module name is taken to display 
                consistent module path in log messages.

        Returns:
            Logger: instance of the child Logger class with same log_level, 
                log_file_path and settings of the parent Logger.
        """
        child_logger = self.logger.getChild(name.split('.')[-1])

        new_logger = Logger(
            logger_name=child_logger.name,
            log_level=child_logger.level,
            log_format=self.log_format,
        )

        new_logger.logger.propagate = False
        return new_logger

    def log(self,
            message: str,
            level: str = logging.INFO):
        """Basic log message. 

        Args:
            message (str): message to be displayed.
            level (str, optional): level of the log message. Defaults 
                to logging.INFO.
        """
        self.logger.log(msg=message, level=level)

    def info(self, message: str):
        """INFO log message

        Args:
            message (str): message to be displayed.
        """
        self.logger.log(msg=message, level=logging.INFO)

    def debug(self, message: str):
        """DEBUG log message

        Args:
            message (str): message to be displayed.
        """
        self.logger.log(msg=message, level=logging.DEBUG)

    def warning(self, message: str):
        """WARNING log message

        Args:
            message (str): message to be displayed.
        """
        self.logger.log(msg=message, level=logging.WARNING)

    def error(self, message: str):
        """ERROR log message

        Args:
            message (str): message to be displayed.
        """
        self.logger.log(msg=message, level=logging.ERROR)

    def critical(self, message: str):
        """CRITICAL log message

        Args:
            message (str): message to be displayed.
        """
        self.logger.log(msg=message, level=logging.CRITICAL)
