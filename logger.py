import logging
import os
import sys
import datetime
class Logger():
    """ log object print log at stdout and save it to local disk\n
    Args:
        log_dir: local directory for save log
        name: log file's name
    Returns:
        logger: logger
    """

    def __init__(self, log_dir, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level=logging.INFO)
        formater = logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s', '%m-%d %H:%M:%S')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        now = datetime.datetime.now()
        log_path = os.path.join(
            log_dir, '{}{:%Y%m%dT%H%M}.log'.format(name, now))
        handler = logging.FileHandler(log_path)
        handler.setFormatter(formater)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)
        self.logger.info("Start print log")

    def __call__(self, log):
        self.logger.info(log)