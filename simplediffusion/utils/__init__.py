import logging
import time


class Timer:
    def __init__(self, name="Time taken"):
        self.start = time.time()
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type or exc_value or traceback:
            logging.info(f"{self.name}: Exception occurred.")

        self.end = time.time()
        self.interval = self.end - self.start
        logging.info(f"{self.name}: {self.interval:.2f} seconds.")
