# g:\Projects\backtesting\logging_config.py
import logging
import sys

def setup_logging(level=logging.INFO, 
                  log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                  date_format="%Y-%m-%d %H:%M:%S"
                ):
    """
    Configures logging for the entire application.
    """
   
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to prevent duplication.
    # This is important if this function is ever called more than once.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(log_format, datefmt=date_format)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


