import logging


def configure_logger(log_level=logging.INFO):
    logging.basicConfig(level=log_level, datefmt='%d.%m.%Y %I:%M:%S %p',
                        format='[%(asctime)s] [%(levelname)8s] ---'
                               ' %(threadName)-10s --- %(message)s --- (%(name)s:%(funcName)s:%('
                               'lineno)s)')
