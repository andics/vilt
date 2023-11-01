import logging
import os
import sys
import time

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s | %(levelname)s | %(module)s | %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }


    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class loggerObj():
    '''
    This class initiates a logger with a given name. It prints in a colorful format and
    stores the log-files at a particular parent-folder.
    '''

    def __init__(self, logs_subdir, log_file_name, utils_helper, log_level, name=None):
        self.logs_path = logs_subdir
        self.log_file_name = log_file_name + "_" + time.strftime("%Y_%m_%d_%H-%M-%S") + '.log'
        self.utils_helper = utils_helper
        self.log_level = log_level
        self.logger_name = name

    def factory_reset_logger(self):
        logger_ref = logging.getLogger(self.logger_name)
        logger_ref.handlers.clear()

    def setup_logger(self):
        if self.utils_helper.check_dir_and_make_if_na(self.logs_path):
            print("General log dir exists; proceeding...")
        else:
            print("General log dir did not exist; created one!")

        #Create a logger
        logger_main = logging.getLogger(self.logger_name)
        logger_main.handlers.clear()
        logger_main.propagate = False
        self.log_file_current = os.path.join(self.logs_path, self.log_file_name)
        self.main_log_file_handler = logging.FileHandler(self.log_file_current)
        stream_handler = logging.StreamHandler(stream=sys.stdout)

        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(module)s | %(message)s')
        self.main_log_file_handler.setFormatter(formatter)
        self.main_log_file_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(CustomFormatter())
        stream_handler.setLevel(logging.DEBUG)
        logger_main.addHandler(self.main_log_file_handler)
        logger_main.addHandler(stream_handler)
        logger_main.setLevel(self.log_level)

        return logger_main


    def add_temp_file_handler_and_remove_main_file_handler(self, dir_to_store_tmp_log):
        #The following function adds a temporary file-handler to the logger:
        #this is used to log the evaluation progress inside the folder of each
        #and every bin, instead of in our general log file
        _tmp_log_file_path = os.path.join(dir_to_store_tmp_log, self.log_file_name)
        self._tmp_log_file_handler = logging.FileHandler(_tmp_log_file_path)

        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(module)s | %(message)s')
        self._tmp_log_file_handler.setFormatter(formatter)
        self._tmp_log_file_handler.setLevel(logging.DEBUG)

        logger_main = logging.getLogger(self.logger_name)
        logger_main.removeHandler(self.main_log_file_handler)
        logger_main.addHandler(self._tmp_log_file_handler)


    def remove_temp_file_handler_and_add_main_file_handler(self):
        logger_main = logging.getLogger(self.logger_name)
        logger_main.removeHandler(self._tmp_log_file_handler)
        logger_main.addHandler(self.main_log_file_handler)